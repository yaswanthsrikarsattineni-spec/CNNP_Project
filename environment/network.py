"""Geometric network topology for the MAHDRL replication.

Implements the spatial layout described in Section II-B and Section V of the
paper:
    - BSs and RISs uniformly deployed in a 300 x 300 m^2 outdoor area.
    - Each BS owns `ris_per_bs` RISs and serves `users_per_bs` Metaverse users
      "under its own cellular coverage".
    - N WiFi APs, each serving `wifi_devices_per_ap` WiFi devices.

⚠ Constants marked "ASSUMPTION" are NOT in the paper. They are engineering
choices we must make to produce a runnable simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import SimulationConfig


# ---------------------------------------------------------------------------
# Implementation assumptions (NOT in the paper)
# ---------------------------------------------------------------------------
BS_HEIGHT_M = 25.0            # ASSUMPTION: typical outdoor macro-cell height
RIS_HEIGHT_M = 15.0           # ASSUMPTION: wall / pole mounted RIS
USER_HEIGHT_M = 1.7           # ASSUMPTION: human HMD height
WIFI_AP_HEIGHT_M = 5.0        # ASSUMPTION: small outdoor AP
RIS_OFFSET_M = 50.0           # ASSUMPTION: distance from BS to its RIS
USER_RADIUS_M = 150.0         # ASSUMPTION: "cellular coverage" radius
WIFI_DEVICE_RADIUS_M = 50.0   # ASSUMPTION: AP service radius


# ---------------------------------------------------------------------------
# Small dataclasses — one per physical entity
# ---------------------------------------------------------------------------
@dataclass
class Position3D:
    """A point in 3-D space, in metres."""
    x: float
    y: float
    z: float

    def distance_to(self, other: "Position3D") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def horizontal_distance_to(self, other: "Position3D") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return float(np.sqrt(dx * dx + dy * dy))


@dataclass
class BaseStation:
    id: int
    position: Position3D


@dataclass
class RIS:
    id: int
    bs_id: int                    # which BS this RIS belongs to
    position: Position3D
    num_elements: int


@dataclass
class MetaverseUser:
    id: int
    bs_id: int                    # which BS serves this user
    position: Position3D


@dataclass
class WiFiAP:
    id: int
    position: Position3D


@dataclass
class WiFiDevice:
    id: int
    ap_id: int                    # which WiFi AP this device is attached to
    position: Position3D


# ---------------------------------------------------------------------------
# Main topology class
# ---------------------------------------------------------------------------
class NetworkTopology:
    """Generate and store every physical entity's position once per seed.

    All downstream modules (channel model, SINR, rate, queue, QoE) will read
    positions and distances from this object. It is intentionally NOT a
    simulator — it only describes *where things are*.
    """

    def __init__(self, cfg: SimulationConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed if seed is None else seed)

        self.base_stations: List[BaseStation] = []
        self.ris_list: List[RIS] = []
        self.users: List[MetaverseUser] = []
        self.wifi_aps: List[WiFiAP] = []
        self.wifi_devices: List[WiFiDevice] = []

        self._build()

    # ------------------------------------------------------------------
    # Public helper API
    # ------------------------------------------------------------------
    def ris_for_bs(self, bs_id: int) -> List[RIS]:
        return [r for r in self.ris_list if r.bs_id == bs_id]

    def users_for_bs(self, bs_id: int) -> List[MetaverseUser]:
        return [u for u in self.users if u.bs_id == bs_id]

    def devices_for_ap(self, ap_id: int) -> List[WiFiDevice]:
        return [d for d in self.wifi_devices if d.ap_id == ap_id]

    # ------------------------------------------------------------------
    # Private placement routines
    # ------------------------------------------------------------------
    def _build(self) -> None:
        cfg = self.cfg
        W, H = cfg.area_width_m, cfg.area_height_m

        # 1) BS positions
        for m, (x, y) in enumerate(self._grid_positions(cfg.num_bs, W, H)):
            self.base_stations.append(
                BaseStation(id=m, position=Position3D(x, y, BS_HEIGHT_M))
            )

        # 2) RIS positions — offset from owning BS at distinct angles
        for bs in self.base_stations:
            for s in range(cfg.ris_per_bs):
                angle = 2.0 * np.pi * s / max(cfg.ris_per_bs, 1) + np.pi / 4.0
                rx = float(np.clip(bs.position.x + RIS_OFFSET_M * np.cos(angle), 0, W))
                ry = float(np.clip(bs.position.y + RIS_OFFSET_M * np.sin(angle), 0, H))
                self.ris_list.append(
                    RIS(
                        id=len(self.ris_list),
                        bs_id=bs.id,
                        position=Position3D(rx, ry, RIS_HEIGHT_M),
                        num_elements=cfg.ris_elements,
                    )
                )

        # 3) Metaverse users — uniform inside a disk around their serving BS
        for bs in self.base_stations:
            self._place_uniform_disk(
                center=bs.position,
                radius=USER_RADIUS_M,
                count=cfg.users_per_bs,
                W=W,
                H=H,
                make=lambda pos: MetaverseUser(
                    id=len(self.users), bs_id=bs.id, position=pos
                ),
                sink=self.users,
            )

        # 4) Wi-Fi APs — uniform random in the whole area
        for n in range(cfg.num_wifi_aps):
            self.wifi_aps.append(
                WiFiAP(
                    id=n,
                    position=Position3D(
                        float(self.rng.uniform(0, W)),
                        float(self.rng.uniform(0, H)),
                        WIFI_AP_HEIGHT_M,
                    ),
                )
            )

        # 5) Wi-Fi devices — uniform inside a disk around their AP
        for ap in self.wifi_aps:
            self._place_uniform_disk(
                center=ap.position,
                radius=WIFI_DEVICE_RADIUS_M,
                count=cfg.wifi_devices_per_ap,
                W=W,
                H=H,
                make=lambda pos, ap_id=ap.id: WiFiDevice(
                    id=len(self.wifi_devices), ap_id=ap_id, position=pos
                ),
                sink=self.wifi_devices,
            )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _place_uniform_disk(
        self,
        center: Position3D,
        radius: float,
        count: int,
        W: float,
        H: float,
        make,
        sink: list,
    ) -> None:
        """Place `count` points uniformly inside a disk around `center`.

        Uses rejection sampling so that points stay inside the disk radius and
        are then clipped to the rectangular area. Expected cost: ~1-2 attempts
        per point.
        """
        placed = 0
        safety = count * 50
        attempts = 0
        while placed < count and attempts < safety:
            attempts += 1
            dx = self.rng.uniform(-radius, radius)
            dy = self.rng.uniform(-radius, radius)
            if dx * dx + dy * dy > radius * radius:
                continue
            x = float(np.clip(center.x + dx, 0.0, W))
            y = float(np.clip(center.y + dy, 0.0, H))
            sink.append(make(Position3D(x, y, center.z if center.z else USER_HEIGHT_M)))
            placed += 1
        if placed < count:
            raise RuntimeError(
                f"Could only place {placed}/{count} points — check radius/area."
            )

    @staticmethod
    def _grid_positions(num: int, W: float, H: float) -> list[tuple[float, float]]:
        """Evenly spaced BS positions.

        For num=1 we return the area centre — matching a single-BS deployment.
        For num>1 we build a roughly square grid.
        """
        if num == 1:
            return [(W / 2.0, H / 2.0)]
        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        xs = np.linspace(W / (2 * cols), W - W / (2 * cols), cols)
        ys = np.linspace(H / (2 * rows), H - H / (2 * rows), rows)
        positions: list[tuple[float, float]] = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= num:
                    break
                positions.append((float(xs[c]), float(ys[r])))
        return positions
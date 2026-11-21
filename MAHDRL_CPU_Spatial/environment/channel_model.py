"""Channel model for the MAHDRL replication.

Implements Section II-B of the paper:

    Eq. (1)   RIS reflection matrix  Θ = diag{A_l exp(j θ_l)}
    Eq. (3)   Direct BS→user channel  h̃ = sqrt(L/(κ+1)) (sqrt(κ) h_LOS + h_NLOS)
    Eq. (5)   BS→RIS channel F  and  RIS→user channel g  (same Rician form)
    Eq. (2)   Composite user channel  h = h̃ + Σ g Θ F
    Eq. (9)   Composite WiFi-device channel  v = ṽ + Σ g Θ F

⚠ ASSUMPTIONS (not in the paper):
    - Antenna array at BS is a Uniform Linear Array (ULA) along x with λ/2 spacing.
    - RIS reflecting elements form a ULA along x with λ/2 spacing.
    - The Rician K-factor is the same for all links (paper's Table I value).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from config import SimulationConfig
from environment.network import (
    NetworkTopology,
    Position3D,
    USER_HEIGHT_M,
)

SPEED_OF_LIGHT = 3.0e8            # m/s
ANTENNA_SPACING_FACTOR = 0.5      # ASSUMPTION: λ/2 element spacing (ULA)
RIS_SPACING_FACTOR = 0.5          # ASSUMPTION: λ/2 element spacing (ULA)


class ChannelModel:
    """Generates small-scale fading realizations for every link in the topology.

    Public API:
        direct_channel(bs_id, user_id)                -> shape (1, N_m)
        bs_ris_channel(bs_id, ris_id)                 -> shape (L, N_m)
        ris_user_channel(ris_id, user_id)             -> shape (1, L)
        bs_wifi_channel(bs_id, wifi_device_id)        -> shape (1, N_m)
        ris_wifi_channel(ris_id, wifi_device_id)      -> shape (1, L)
        ris_phase_matrix(ris_id, phases)              -> shape (L, L)
        composite_user_channel(bs_id, user_id, phases)   -> shape (1, N_m)
        composite_wifi_channel(bs_id, dev_id, phases)    -> shape (1, N_m)

    "phases" is a dict mapping ris_id -> np.ndarray of length L, phase in [0, 2π].
    """

    def __init__(
        self,
        cfg: SimulationConfig,
        topology: NetworkTopology,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.topology = topology
        self.wavelength = SPEED_OF_LIGHT / cfg.carrier_frequency_hz
        self.antenna_spacing = ANTENNA_SPACING_FACTOR * self.wavelength
        self.ris_spacing = RIS_SPACING_FACTOR * self.wavelength
        self.rng = np.random.default_rng(
            cfg.random_seed if seed is None else seed
        )

        # Pre-compute ULA positions for every BS (does not change over time).
        self._bs_antennas: Dict[int, List[Position3D]] = {
            bs.id: self._ula_positions(
                bs.position, cfg.antennas_per_bs, self.antenna_spacing
            )
            for bs in topology.base_stations
        }
        # Pre-compute ULA positions for every RIS.
        self._ris_elements: Dict[int, List[Position3D]] = {
            r.id: self._ula_positions(
                r.position, r.num_elements, self.ris_spacing
            )
            for r in topology.ris_list
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_bs_antennas(self, bs_id: int) -> List[Position3D]:
        return self._bs_antennas[bs_id]

    def get_ris_elements(self, ris_id: int) -> List[Position3D]:
        return self._ris_elements[ris_id]

    def direct_channel(self, bs_id: int, user_id: int) -> np.ndarray:
        """Eq. (3): direct BS→user channel, shape (1, N_m)."""
        bs = self.topology.base_stations[bs_id]
        user = self.topology.users[user_id]
        d = bs.position.distance_to(user.position)
        pl = self._path_loss(d)
        return self._rician_channel(
            self._bs_antennas[bs_id], [user.position], pl
        )

    def bs_ris_channel(self, bs_id: int, ris_id: int) -> np.ndarray:
        """Eq. (5) BS→RIS: shape (L, N_m)."""
        bs = self.topology.base_stations[bs_id]
        ris = self.topology.ris_list[ris_id]
        d = bs.position.distance_to(ris.position)
        pl = self._path_loss(d)
        return self._rician_channel(
            self._bs_antennas[bs_id],
            self._ris_elements[ris.id],
            pl,
        )

    def ris_user_channel(self, ris_id: int, user_id: int) -> np.ndarray:
        """Eq. (5) RIS→user: shape (1, L)."""
        ris = self.topology.ris_list[ris_id]
        user = self.topology.users[user_id]
        d = ris.position.distance_to(user.position)
        pl = self._path_loss(d)
        return self._rician_channel(
            self._ris_elements[ris.id], [user.position], pl
        )

    def bs_wifi_channel(self, bs_id: int, wifi_device_id: int) -> np.ndarray:
        """Direct BS→WiFi-device channel, shape (1, N_m)."""
        bs = self.topology.base_stations[bs_id]
        dev = self.topology.wifi_devices[wifi_device_id]
        d = bs.position.distance_to(dev.position)
        pl = self._path_loss(d)
        return self._rician_channel(
            self._bs_antennas[bs_id], [dev.position], pl
        )

    def ris_wifi_channel(self, ris_id: int, wifi_device_id: int) -> np.ndarray:
        """RIS→WiFi-device channel, shape (1, L)."""
        ris = self.topology.ris_list[ris_id]
        dev = self.topology.wifi_devices[wifi_device_id]
        d = ris.position.distance_to(dev.position)
        pl = self._path_loss(d)
        return self._rician_channel(
            self._ris_elements[ris.id], [dev.position], pl
        )

    def ris_phase_matrix(
        self, ris_id: int, phases: np.ndarray
    ) -> np.ndarray:
        """Eq. (1): Θ = diag{A_l exp(j θ_l)} with |A_l| = 1, shape (L, L)."""
        L = self.topology.ris_list[ris_id].num_elements
        phases = np.asarray(phases, dtype=float).reshape(-1)
        if phases.size != L:
            raise ValueError(f"Expected {L} phases for RIS {ris_id}, got {phases.size}")
        return np.diag(np.exp(1j * phases))

    def composite_user_channel(
        self,
        bs_id: int,
        user_id: int,
        ris_phases: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Eq. (2): direct + all RIS-reflected paths, shape (1, N_m)."""
        h = self.direct_channel(bs_id, user_id).astype(complex)
        for ris in self.topology.ris_for_bs(bs_id):
            if ris.id not in ris_phases:
                continue
            F = self.bs_ris_channel(bs_id, ris.id)          # (L, N_m)
            g = self.ris_user_channel(ris.id, user_id)      # (1, L)
            Theta = self.ris_phase_matrix(ris.id, ris_phases[ris.id])
            h = h + g @ Theta @ F
        return h

    def composite_wifi_channel(
        self,
        bs_id: int,
        wifi_device_id: int,
        ris_phases: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Eq. (9): direct + all RIS-reflected paths to a WiFi device, shape (1, N_m)."""
        v = self.bs_wifi_channel(bs_id, wifi_device_id).astype(complex)
        for ris in self.topology.ris_for_bs(bs_id):
            if ris.id not in ris_phases:
                continue
            F = self.bs_ris_channel(bs_id, ris.id)
            g = self.ris_wifi_channel(ris.id, wifi_device_id)
            Theta = self.ris_phase_matrix(ris.id, ris_phases[ris.id])
            v = v + g @ Theta @ F
        return v

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _path_loss(self, distance_m: float) -> float:
        """Free-space path loss in linear power units: L = (c / (4π f d))^n."""
        c = SPEED_OF_LIGHT
        f = self.cfg.carrier_frequency_hz
        n = self.cfg.path_loss_exponent
        d = max(distance_m, 1e-3)
        return float((c / (4.0 * np.pi * f * d)) ** n)

    def _rician_channel(
        self,
        tx_positions: List[Position3D],
        rx_positions: List[Position3D],
        path_loss: float,
    ) -> np.ndarray:
        """Eq. (3) / (5): Rician channel, shape (n_rx, n_tx)."""
        n_tx = len(tx_positions)
        n_rx = len(rx_positions)
        H_los = np.zeros((n_rx, n_tx), dtype=complex)
        for i_rx, p_rx in enumerate(rx_positions):
            for j_tx, p_tx in enumerate(tx_positions):
                d = p_rx.distance_to(p_tx)
                H_los[i_rx, j_tx] = np.exp(
                    -1j * 2.0 * np.pi * d / self.wavelength
                )
        H_nlos = (
            self.rng.standard_normal((n_rx, n_tx))
            + 1j * self.rng.standard_normal((n_rx, n_tx))
        ) / np.sqrt(2.0)
        kappa = self.cfg.rician_k_factor
        return np.sqrt(path_loss / (kappa + 1.0)) * (
            np.sqrt(kappa) * H_los + H_nlos
        )

    @staticmethod
    def _ula_positions(
        center: Position3D, n: int, spacing: float
    ) -> List[Position3D]:
        """Uniform Linear Array along x-axis, centered at `center`."""
        if n == 1:
            return [Position3D(center.x, center.y, center.z)]
        length = (n - 1) * spacing
        start_x = center.x - length / 2.0
        return [
            Position3D(start_x + i * spacing, center.y, center.z)
            for i in range(n)
        ]
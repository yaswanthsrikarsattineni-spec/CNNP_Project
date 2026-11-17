"""Tests for the network topology module.

Verifies entity counts, that every entity lies inside the 300 x 300 m^2 area,
and saves a PNG so you can visually inspect the layout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — saves to file, no pop-up window
import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology


def _make_topo() -> NetworkTopology:
    return NetworkTopology(SimulationConfig())


def test_counts_match_config():
    cfg = SimulationConfig()
    topo = _make_topo()
    assert len(topo.base_stations) == cfg.num_bs
    assert len(topo.ris_list) == cfg.num_bs * cfg.ris_per_bs
    assert len(topo.users) == cfg.num_bs * cfg.users_per_bs
    assert len(topo.wifi_aps) == cfg.num_wifi_aps
    assert len(topo.wifi_devices) == cfg.num_wifi_aps * cfg.wifi_devices_per_ap


def test_all_positions_inside_area():
    cfg = SimulationConfig()
    topo = _make_topo()
    for group in (topo.base_stations, topo.ris_list, topo.users,
                  topo.wifi_aps, topo.wifi_devices):
        for e in group:
            assert 0.0 <= e.position.x <= cfg.area_width_m, e
            assert 0.0 <= e.position.y <= cfg.area_height_m, e


def test_deterministic_with_same_seed():
    cfg = SimulationConfig()
    a = NetworkTopology(cfg, seed=cfg.random_seed)
    b = NetworkTopology(cfg, seed=cfg.random_seed)
    for ea, eb in zip(a.users, b.users):
        assert np.isclose(ea.position.x, eb.position.x)
        assert np.isclose(ea.position.y, eb.position.y)


def test_plot_topology_png_saved():
    cfg = SimulationConfig()
    topo = _make_topo()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cfg.area_width_m)
    ax.set_ylim(0, cfg.area_height_m)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.scatter([b.position.x for b in topo.base_stations],
               [b.position.y for b in topo.base_stations],
               marker="^", s=240, c="red", edgecolors="black", label="BS")
    ax.scatter([r.position.x for r in topo.ris_list],
               [r.position.y for r in topo.ris_list],
               marker="s", s=180, c="purple", edgecolors="black", label="RIS")
    ax.scatter([u.position.x for u in topo.users],
               [u.position.y for u in topo.users],
               marker="o", s=40, c="tab:blue", label="Metaverse user")
    ax.scatter([a.position.x for a in topo.wifi_aps],
               [a.position.y for a in topo.wifi_aps],
               marker="*", s=220, c="green", edgecolors="black", label="WiFi AP")
    ax.scatter([d.position.x for d in topo.wifi_devices],
               [d.position.y for d in topo.wifi_devices],
               marker=".", s=25, c="tab:orange", label="WiFi device")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Step 2 — Network topology (single snapshot)")
    ax.legend(loc="upper right", fontsize=9)

    out = Path("results") / "graphs" / "topology_step2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert out.exists(), f"Plot was not saved to {out}"
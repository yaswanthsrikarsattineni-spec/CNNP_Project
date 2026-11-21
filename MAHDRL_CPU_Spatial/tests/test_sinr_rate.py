"""Tests for beamforming, SINR, and data rate."""
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel
from environment.beamforming import mrt_beamformer, zf_beamformers, equal_power_split
from environment.sinr import compute_all_user_sinrs, noise_power_w
from environment.data_rate import total_rate_bps


def _build(num_selected: int):
    cfg = SimulationConfig()
    topo = NetworkTopology(cfg)
    ch = ChannelModel(cfg, topo)
    phases = {0: np.zeros(cfg.ris_elements)}
    channels = [ch.composite_user_channel(0, k, phases) for k in range(num_selected)]
    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)
    powers = equal_power_split(p_tx_w, num_selected)
    noise = noise_power_w(cfg)
    return cfg, channels, powers, noise


def test_noise_power_matches_paper_order_of_magnitude():
    cfg = SimulationConfig()
    n_w = noise_power_w(cfg)
    n_dbm = 10.0 * np.log10(n_w) + 30.0
    # Paper: -174 dBm/Hz * 40 MHz ≈ -98 dBm
    assert -100.0 < n_dbm < -95.0, f"noise={n_dbm:.2f} dBm"


def test_mrt_gives_positive_rate():
    cfg, channels, powers, noise = _build(5)
    bf = [mrt_beamformer(h, p) for h, p in zip(channels, powers)]
    sinrs = compute_all_user_sinrs(channels, bf, noise)
    rate = total_rate_bps(sinrs, cfg.bandwidth_hz)
    assert all(s >= 0 for s in sinrs)
    assert rate > 0, "MRT total rate must be positive"


def test_zf_eliminates_interference_between_users():
    cfg, channels, powers, noise = _build(5)
    H = np.vstack(channels)
    W = zf_beamformers(H, powers)
    # For each pair (k, j != k): h_k @ W[:, j] must be ~0
    for k in range(5):
        for j in range(5):
            if k == j:
                continue
            prod = np.asarray(channels[k] @ W[:, j].reshape(-1, 1)).item()
            assert abs(prod) < 1e-6, f"ZF residual interference h_{k}·w_{j} = {prod}"


def test_zf_rate_positive():
    cfg, channels, powers, noise = _build(5)
    H = np.vstack(channels)
    W = zf_beamformers(H, powers)
    bf = [W[:, k].reshape(-1, 1) for k in range(5)]
    sinrs = compute_all_user_sinrs(channels, bf, noise)
    rate = total_rate_bps(sinrs, cfg.bandwidth_hz)
    assert rate > 0, "ZF total rate must be positive"


def test_rate_order_of_magnitude():
    """Sanity: total rate should be in [1, 500] Mbps range."""
    cfg, channels, powers, noise = _build(10)
    bf = [mrt_beamformer(h, p) for h, p in zip(channels, powers)]
    sinrs = compute_all_user_sinrs(channels, bf, noise)
    rate_mbps = total_rate_bps(sinrs, cfg.bandwidth_hz) / 1e6
    assert 1.0 < rate_mbps < 500.0, (
        f"Rate {rate_mbps:.2f} Mbps is outside [1, 500]. "
        "The paper's Fig. 3(b) shows ~60 Mbps; if we're off by >10x, "
        "the beamforming/rate calibration needs review."
    )
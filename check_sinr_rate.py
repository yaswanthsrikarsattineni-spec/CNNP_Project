"""Diagnostic: compute SINR and total rate for one BS snapshot.

Compares MRT vs ZF. This is the FIRST number we can validate against the
paper's Figure 3(b), which shows ~60 Mbps per BS after convergence.
"""
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel
from environment.beamforming import (
    mrt_beamformer, zf_beamformers, equal_power_split,
)
from environment.sinr import compute_all_user_sinrs, noise_power_w
from environment.data_rate import total_rate_bps, per_user_rates_bps


def run_scenario(num_selected: int) -> None:
    cfg = SimulationConfig()
    topo = NetworkTopology(cfg)
    ch = ChannelModel(cfg, topo)

    phases = {0: np.zeros(cfg.ris_elements)}
    channels = [ch.composite_user_channel(0, k, phases) for k in range(num_selected)]

    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)   # 1 W total
    powers = equal_power_split(p_tx_w, num_selected)
    noise = noise_power_w(cfg)

    # ---- MRT ----
    bf_mrt = [mrt_beamformer(h, p) for h, p in zip(channels, powers)]
    sinrs_mrt = compute_all_user_sinrs(channels, bf_mrt, noise)
    rate_mrt = total_rate_bps(sinrs_mrt, cfg.bandwidth_hz)

    # ---- ZF ----
    H = np.vstack(channels)
    W = zf_beamformers(H, powers)
    bf_zf = [W[:, k].reshape(-1, 1) for k in range(num_selected)]
    sinrs_zf = compute_all_user_sinrs(channels, bf_zf, noise)
    rate_zf = total_rate_bps(sinrs_zf, cfg.bandwidth_hz)

    print("=" * 74)
    print(f"Selected users: {num_selected}   (BS has {cfg.antennas_per_bs} antennas)")
    print(f"Per-user power: {10*np.log10(powers[0]) + 30:.1f} dBm")
    print(f"Noise floor:    {10*np.log10(noise) + 30:.1f} dBm")
    print("-" * 74)
    print(f"{'MRT':<10} SINRs (dB): {[f'{10*np.log10(s+1e-12):7.1f}' for s in sinrs_mrt]}")
    print(f"{'MRT':<10} rate per user (Mbps): {[f'{r/1e6:6.2f}' for r in per_user_rates_bps(sinrs_mrt, cfg.bandwidth_hz)]}")
    print(f"{'MRT':<10} total rate (Mbps): {rate_mrt/1e6:.2f}")
    print("-" * 74)
    print(f"{'ZF':<10} SINRs (dB): {[f'{10*np.log10(s+1e-12):7.1f}' for s in sinrs_zf]}")
    print(f"{'ZF':<10} rate per user (Mbps): {[f'{r/1e6:6.2f}' for r in per_user_rates_bps(sinrs_zf, cfg.bandwidth_hz)]}")
    print(f"{'ZF':<10} total rate (Mbps): {rate_zf/1e6:.2f}")
    print("=" * 74)
    print("Paper Fig. 3(b) shows ~55-70 Mbps per BS after training.")
    print("Pick whichever baseline is closest to that and explain in your report.")
    print()


if __name__ == "__main__":
    for k in (5, 10):
        run_scenario(k)
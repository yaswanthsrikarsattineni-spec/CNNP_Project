"""Quick diagnostic: print channel magnitudes for a single BS/user.

Useful to confirm the physics is sensible before we build SINR and rate on top
of the channel model. Not a unit test — just a human-readable printout.
"""
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel


def main() -> None:
    cfg = SimulationConfig()
    topo = NetworkTopology(cfg)
    ch = ChannelModel(cfg, topo)

    bs = topo.base_stations[0]
    user = topo.users[0]
    d = bs.position.distance_to(user.position)
    pl_linear = ch._path_loss(d)
    pl_db = 10.0 * np.log10(pl_linear)

    # ---- Direct channel ----
    h = ch.direct_channel(0, 0)
    avg_power = float(np.mean(np.abs(h) ** 2))
    avg_power_db = 10.0 * np.log10(avg_power)

    # ---- Composite channel with a random RIS phase configuration ----
    phases = {0: np.random.default_rng(0).uniform(0, 2 * np.pi, cfg.ris_elements)}
    h_comp = ch.composite_user_channel(0, 0, phases)
    avg_power_comp = float(np.mean(np.abs(h_comp) ** 2))
    avg_power_comp_db = 10.0 * np.log10(avg_power_comp)

    # ---- Expected RX power if BS transmits at Pmax ----
    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)     # 30 dBm → 1 W
    rx_w = p_tx_w * avg_power_comp
    rx_dbm = 10.0 * np.log10(rx_w) + 30.0
    noise_dbm = cfg.noise_psd_dbm_per_hz + 10.0 * np.log10(cfg.bandwidth_hz)

    print("=" * 60)
    print(f"BS #0 at ({bs.position.x:6.1f}, {bs.position.y:6.1f}) m")
    print(f"User #0 at ({user.position.x:6.1f}, {user.position.y:6.1f}) m")
    print(f"Distance: {d:.1f} m")
    print("-" * 60)
    print(f"Path loss:              {pl_db:8.2f} dB")
    print(f"Direct |h|^2 (dB):      {avg_power_db:8.2f} dB")
    print(f"Composite |h|^2 (dB):   {avg_power_comp_db:8.2f} dB")
    print(f"  (RIS should make composite |h|^2 differ from direct)")
    print("-" * 60)
    print(f"TX power (Pmax):        {cfg.max_bs_power_dbm:8.2f} dBm")
    print(f"RX power:               {rx_dbm:8.2f} dBm")
    print(f"Noise floor (40 MHz):   {noise_dbm:8.2f} dBm")
    print(f"Bulk SNR:               {rx_dbm - noise_dbm:8.2f} dB")
    print("=" * 60)
    print("Interpretation:")
    print("  - Path loss should be roughly -85 to -105 dB for 10-150 m distance.")
    print("  - Bulk SNR should be positive (e.g. +10 to +40 dB), not negative.")
    print("  - If SNR is negative, no communication is possible and RL will fail.")


if __name__ == "__main__":
    main()
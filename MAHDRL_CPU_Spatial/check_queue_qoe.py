"""Diagnostic: queue + QoE with MRT (all users selected) for one episode."""
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel
from environment.beamforming import mrt_beamformer, equal_power_split
from environment.sinr import compute_all_user_sinrs, noise_power_w
from environment.data_rate import per_user_rates_bps
from environment.queue_model import QueueModel
from environment.qoe import compute_qoe, compute_reliability


def main():
    cfg = SimulationConfig()
    topo = NetworkTopology(cfg)
    ch = ChannelModel(cfg, topo)
    K = cfg.users_per_bs
    qm = QueueModel(cfg, num_users=K)

    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)
    powers = equal_power_split(p_tx_w, K)
    noise = noise_power_w(cfg)
    phases = {0: np.zeros(cfg.ris_elements)}

    total_qoe = 0.0
    for t in range(cfg.time_slots_per_episode):
        channels = [ch.composite_user_channel(0, k, phases) for k in range(K)]
        bf = [mrt_beamformer(h, p) for h, p in zip(channels, powers)]
        sinrs = compute_all_user_sinrs(channels, bf, noise)
        rates = np.array(per_user_rates_bps(sinrs, cfg.bandwidth_hz))

        selected = np.ones(K, dtype=bool)
        arrivals = qm.sample_arrivals()
        latencies = qm.total_latency(rates, selected)
        qoe = compute_qoe(rates, latencies, cfg)
        total_qoe += qoe
        qm.step(selected, rates, arrivals)

        if t % 20 == 0 or t == cfg.time_slots_per_episode - 1:
            print(
                f"t={t:3d}  "
                f"mean queue={np.mean(qm.queues):9.0f} bits  "
                f"mean latency={1e3*np.mean(latencies):7.3f} ms  "
                f"mean rate={np.mean(rates)/1e6:6.2f} Mbps  "
                f"QoE={qoe/1e6:8.2f} Mbps-eq  "
                f"reliability={compute_reliability(qm.queues, cfg):.2f}"
            )
    print("-" * 92)
    print(f"Average QoE per slot: {total_qoe/cfg.time_slots_per_episode/1e6:.2f} Mbps-eq")
    print(f"Final mean queue:     {np.mean(qm.queues):.0f} bits  (max={cfg.max_queue_bits:.0f})")
    print(f"Final reliability:    {compute_reliability(qm.queues, cfg):.2f}")


if __name__ == "__main__":
    main()
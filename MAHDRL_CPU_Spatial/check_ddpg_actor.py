"""Diagnostic: feed DDPG actor output into the environment for one episode.

An UNTRAINED actor produces random beamformers. Expect rate to be lower than
the MRT baseline but still meaningful (SINR positive thanks to full-power
transmission). This verifies the actor → environment pipeline works.
"""
import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from agents.ddpg_actor import DDPGActor


def main():
    cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(cfg)
    obs = env.reset()

    p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)
    actor = DDPGActor(
        obs_dim=obs.shape[0],
        num_users=cfg.users_per_bs,
        num_ant=cfg.antennas_per_bs,
        num_ris=cfg.ris_per_bs,
        ris_elements=cfg.ris_elements,
        max_power_w=p_tx_w,
        agent_cfg=agent_cfg,
        device="cpu",
        seed=0,
    )

    print(f"Actor: input={actor.input_dim}, output={actor.output_dim} "
          f"(beam={actor.beam_dim}, phase={actor.phase_dim})")
    print()

    total_reward = 0.0
    total_rate = 0.0
    total_wifi_excess = 0.0
    N_steps = cfg.time_slots_per_episode
    for t in range(N_steps):
        # Deterministic user selection (first 10), random-continuous beamformer
        sel = np.zeros(cfg.users_per_bs, dtype=np.float32)
        sel[: min(cfg.antennas_per_bs, cfg.users_per_bs)] = 1.0

        action = actor.select_action(obs, sel, noise_std=0.0)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        total_rate += info["total_rate_bps"]
        wifi_excess_db = np.maximum(
            0.0, info["wifi_psi_dbm"] - cfg.wifi_energy_detection_dbm
        )
        total_wifi_excess += float(np.sum(wifi_excess_db))

        if t % 20 == 0 or done:
            print(
                f"t={t:3d}  "
                f"rate={info['total_rate_bps']/1e6:6.2f} Mbps  "
                f"wifi_max={info['wifi_psi_dbm'].max():7.2f} dBm  "
                f"mean_q={np.mean(info['queue_bits']):7.0f} bits  "
                f"reward={reward/1e6:7.2f} Mbps-eq  "
                f"done={done}"
            )

    print("-" * 100)
    print(f"Mean rate       (untrained actor): {total_rate/N_steps/1e6:.2f} Mbps")
    print(f"Mean reward     (untrained actor): {total_reward/N_steps/1e6:.2f} Mbps-eq")
    print(f"Mean WiFi excess per step: {total_wifi_excess/N_steps:.2f} dB (sum over devices)")
    print()
    print("Interpretation:")
    print("  - Rate should be NON-ZERO (>1 Mbps). If 0, actor is mis-wired.")
    print("  - Rate may be LOWER than MRT baseline (~20 Mbps) since the actor is")
    print("    random and does not yet know how to beamform. That's expected.")
    print("  - Untrained actor is expected to violate WiFi (random directions).")
    print("  - After Step 11-12 (full training), rate should exceed 60 Mbps.")


if __name__ == "__main__":
    main()
"""Diagnostic: run one full episode with round-robin MRT baseline.

Now also shows the Lyapunov virtual queue and bonus per step.
"""
import numpy as np

from config import SimulationConfig
from environment.simulation import SimulationEnvironment


def main():
    cfg = SimulationConfig()
    env = SimulationEnvironment(cfg)
    obs = env.reset()
    print(f"Observation dim: {obs.shape[0]}")

    total_reward = 0.0
    rewards = []
    lyap_bonuses = []

    for t in range(cfg.time_slots_per_episode):
        offset = t % cfg.users_per_bs
        action = env.make_mrt_action(num_selected=10, offset=offset)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        lyap_bonuses.append(info["lyapunov_bonus_bps"])

        if t % 20 == 0 or done:
            print(
                f"t={t:3d}  "
                f"rate={info['total_rate_bps']/1e6:6.2f} Mbps  "
                f"wifi_max={info['wifi_psi_dbm'].max():7.2f} dBm  "
                f"max_q={info['queue_bits'].max():9.0f} bits  "
                f"max_Y={info['virtual_queue_bits'].max():9.0f} bits  "
                f"mean_lat={1e3*np.mean(info['latencies_s']):6.2f} ms  "
                f"base={info['base_reward_bps']/1e6:6.2f}  "
                f"lyap={info['lyapunov_bonus_bps']/1e6:6.3f}  "
                f"total={reward/1e6:6.2f} Mbps-eq  "
                f"rel={info['reliability']:.2f}  done={done}"
            )

    print("-" * 120)
    print(f"Episode total reward:       {total_reward/1e6:.2f} Mbps-eq × {len(rewards)} steps")
    print(f"Mean reward per step:       {np.mean(rewards)/1e6:.2f} Mbps-eq")
    print(f"Mean Lyapunov bonus:        {np.mean(lyap_bonuses)/1e6:.3f} Mbps-eq "
          f"(should be tiny when queues are healthy)")
    print(f"Max Wi-Fi interference:     {info['wifi_psi_dbm'].max():.2f} dBm "
          f"(threshold = {cfg.wifi_energy_detection_dbm:.0f} dBm)")
    print(f"Final max virtual queue:    {info['virtual_queue_bits'].max():.0f} bits")
    print(f"Final reliability:          {info['reliability']:.2f}")
    print()
    print("Health checks:")
    print("  - max_Y should be exactly 0 (queues far below q_max = 3e6)")
    print("  - lyap bonus should be ≈ 0 Mbps (tiny relative to base reward)")
    print("  - wifi_max still above -62 dBm (baseline violates; RL must fix)")
    print("  - reward still positive O(10 Mbps-eq)")


if __name__ == "__main__":
    main()
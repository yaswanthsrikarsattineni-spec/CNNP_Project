"""Training loop for the CC-MAHDRL baseline (M=1: == MAHDRL with new seed)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from training.cc_mahdrl import CCMAHDRLAgent


def train_cc_mahdrl(
    num_episodes: int = 100,
    output_tag: str = "cc_mahdrl",
    seed: int = 1,
    log_every: int = 10,
) -> dict:
    sim_cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = CCMAHDRLAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )

    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"results/logs/{output_tag}_training.csv")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "total_step", "episode_reward_bps", "episode_rate_bps",
        "episode_wifi_excess_db", "episode_latency_ms", "reliability",
    ])

    total_steps = 0
    eps_decay = 3000
    noise_decay = 3000
    warmup = agent_cfg.ddpg_warmup_steps

    print(f"[cc_mahdrl] CC-MAHDRL (M=1 degenerate) for {num_episodes} episodes")
    t_start = time.time()

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = ep_rate = ep_wifi = ep_lat = 0.0
        done = False
        while not done:
            eps = max(
                agent_cfg.dqn_epsilon_start * (1.0 - total_steps / eps_decay),
                agent_cfg.dqn_epsilon_end,
            )
            noise = max(
                agent_cfg.exploration_noise_std * (1.0 - total_steps / noise_decay),
                agent_cfg.dqn_epsilon_end,
            )
            disc, raw, action = agent.select_action(obs, epsilon=eps, noise_std=noise)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, disc, raw, reward, next_obs, float(done))

            ep_reward += reward
            ep_rate += info["total_rate_bps"]
            ep_wifi += float(np.sum(np.maximum(
                0.0, info["wifi_psi_dbm"] - sim_cfg.wifi_energy_detection_dbm
            )))
            ep_lat += float(np.mean(info["latencies_s"]))

            if (agent.buffer.size >= warmup) and (total_steps % agent_cfg.ddpg_update_every == 0):
                agent.train_step()

            obs = next_obs
            total_steps += 1

        T = sim_cfg.time_slots_per_episode
        csv_writer.writerow([
            ep, total_steps, f"{ep_reward/T:.2f}", f"{ep_rate/T:.2f}",
            f"{ep_wifi/T:.2f}", f"{ep_lat/T*1e3:.3f}", f"{info['reliability']:.3f}",
        ])
        csv_file.flush()

        if ep % log_every == 0:
            elapsed = time.time() - t_start
            print(f"[cc_mahdrl] ep {ep:3d}/{num_episodes}  "
                  f"r={ep_reward/T/1e6:6.2f} Mbps-eq  "
                  f"rate={ep_rate/T/1e6:5.2f} Mbps  ({elapsed:.0f}s)")

    csv_file.close()
    print(f"[cc_mahdrl] Done in {time.time() - t_start:.1f}s")
    return {"log": str(csv_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    train_cc_mahdrl(args.episodes, seed=args.seed)
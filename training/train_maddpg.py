"""Training loop for the MADDPG baseline."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from training.maddpg import MADDPGAgent


def train_maddpg(
    num_episodes: int,
    output_tag: str = "maddpg",
    seed: int = 0,
    log_every: int = 10,
) -> dict:
    sim_cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MADDPGAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )

    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"results/logs/{output_tag}_training.csv")
    ckpt_path = Path(f"results/checkpoints/{output_tag}_final.pt")

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

    print(f"[maddpg] {num_episodes} episodes")
    t_start = time.time()

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = ep_rate = ep_wifi = ep_lat = 0.0
        done = False
        while not done:
            noise = max(
                agent_cfg.exploration_noise_std * (1.0 - total_steps / noise_decay),
                agent_cfg.dqn_epsilon_end,
            )
            raw, action = agent.select_action(obs, noise_std=noise)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, raw, reward, next_obs, float(done))

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
            print(f"[maddpg] ep {ep:3d}/{num_episodes}  "
                  f"r={ep_reward/T/1e6:6.2f} Mbps-eq  "
                  f"rate={ep_rate/T/1e6:5.2f} Mbps  "
                  f"({elapsed:.0f}s)")

    agent.save(str(ckpt_path))
    csv_file.close()
    print(f"[maddpg] Done in {time.time() - t_start:.1f}s")
    return {"checkpoint": str(ckpt_path), "log": str(csv_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--tag", type=str, default="maddpg")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_maddpg(args.episodes, args.tag, args.seed)
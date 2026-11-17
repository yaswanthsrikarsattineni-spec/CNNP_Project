"""Full MAHDRL training loop — paper Algorithm 1 at scale.

Paper's training budget: 400 episodes × 100 slots = 40,000 env steps.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from agents.mahdrl import MAHDRLAgent


def train(
    num_episodes: int,
    output_tag: str,
    seed: int = 0,
    checkpoint_every: int = 100,
    log_every: int = 10,
    disable_ris: bool = False,
) -> dict:
    sim_cfg = SimulationConfig()
    if disable_ris:
        # Copy with RIS removed. All downstream components auto-adjust.
        sim_cfg = replace(sim_cfg, ris_per_bs=0)
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MAHDRLAgent(
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

    tag_desc = "MAHDRL (no RIS)" if disable_ris else "MAHDRL"
    print(f"[{output_tag}] {tag_desc}: {num_episodes} episodes")
    if disable_ris:
        print(f"[{output_tag}] RIS disabled: ris_per_bs = 0")

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
            print(f"[{output_tag}] ep {ep:3d}/{num_episodes}  "
                  f"r={ep_reward/T/1e6:6.2f} Mbps-eq  "
                  f"rate={ep_rate/T/1e6:5.2f} Mbps  "
                  f"({elapsed:.0f}s)")

        if ep % checkpoint_every == 0:
            agent.save(str(Path(f"results/checkpoints/{output_tag}_ep{ep:03d}.pt")))

    agent.save(str(ckpt_path))
    csv_file.close()
    print(f"[{output_tag}] Done in {time.time() - t_start:.1f}s")
    return {"checkpoint": str(ckpt_path), "log": str(csv_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--tag", type=str, default="mahdrl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ris", action="store_true")
    args = p.parse_args()
    train(
        num_episodes=args.episodes, output_tag=args.tag, seed=args.seed,
        disable_ris=args.no_ris,
    )
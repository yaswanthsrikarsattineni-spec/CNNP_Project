"""Batch-run the Figure 5 sweep: WiFi interference & reliability vs # APs.

⚠ This takes ~60 minutes on CPU (10 AP values × 2 algorithms × 100 episodes).
You can lower EPISODES_TRAIN to 50 to halve the time.
"""
from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment
from agents.mahdrl import MAHDRLAgent
from training.maddpg import MADDPGAgent


AP_COUNTS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
EPISODES_TRAIN = 100
EVAL_EPISODES = 5


def _eval_mahdrl(agent, env, n_episodes=EVAL_EPISODES):
    """Return (wifi_dbm, reliability)."""
    wifis, rels = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_w = 0.0
        ep_rel = 0.0
        steps = 0
        while not done:
            _, _, action = agent.select_action(obs, epsilon=0.0, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            ep_rel += float(info["reliability"])
            steps += 1
        wifis.append(ep_w / steps)
        rels.append(ep_rel / steps)
    return float(np.mean(wifis)), float(np.mean(rels))


def _eval_maddpg(agent, env, n_episodes=EVAL_EPISODES):
    """Return (wifi_dbm, reliability)."""
    wifis, rels = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_w = 0.0
        ep_rel = 0.0
        steps = 0
        while not done:
            _, action = agent.select_action(obs, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            ep_rel += float(info["reliability"])
            steps += 1
        wifis.append(ep_w / steps)
        rels.append(ep_rel / steps)
    return float(np.mean(wifis)), float(np.mean(rels))


def _train_mahdrl(num_aps: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), num_wifi_aps=num_aps)
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MAHDRLAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )
    warmup = agent_cfg.ddpg_warmup_steps
    total = 0
    for _ in range(train_eps):
        obs = env.reset()
        done = False
        while not done:
            eps = max(agent_cfg.dqn_epsilon_start * (1.0 - total / 3000),
                      agent_cfg.dqn_epsilon_end)
            noise = max(agent_cfg.exploration_noise_std * (1.0 - total / 3000),
                        agent_cfg.dqn_epsilon_end)
            disc, raw, action = agent.select_action(obs, epsilon=eps, noise_std=noise)
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, disc, raw, reward, next_obs, float(done))
            if agent.buffer.size >= warmup and total % agent_cfg.ddpg_update_every == 0:
                agent.train_step()
            obs = next_obs
            total += 1
    return _eval_mahdrl(agent, env)


def _train_maddpg(num_aps: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), num_wifi_aps=num_aps)
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MADDPGAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )
    warmup = agent_cfg.ddpg_warmup_steps
    total = 0
    for _ in range(train_eps):
        obs = env.reset()
        done = False
        while not done:
            noise = max(agent_cfg.exploration_noise_std * (1.0 - total / 3000),
                        agent_cfg.dqn_epsilon_end)
            raw, action = agent.select_action(obs, noise_std=noise)
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, raw, reward, next_obs, float(done))
            if agent.buffer.size >= warmup and total % agent_cfg.ddpg_update_every == 0:
                agent.train_step()
            obs = next_obs
            total += 1
    return _eval_maddpg(agent, env)


def main():
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    csv_path = Path("results/logs/figure5_sweep.csv")

    rows = []
    t_start = time.time()

    for num_aps in AP_COUNTS:
        print(f"\n[sweep] num_aps = {num_aps}")

        t0 = time.time()
        wifi_m, rel_m = _train_mahdrl(num_aps, EPISODES_TRAIN, seed=0)
        print(f"  MAHDRL  : wifi={wifi_m:6.1f} dBm  reliability={rel_m:.3f}  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MAHDRL", "num_aps": num_aps,
            "mean_wifi_dbm": wifi_m, "reliability": rel_m,
        })

        t0 = time.time()
        wifi_d, rel_d = _train_maddpg(num_aps, EPISODES_TRAIN, seed=0)
        print(f"  MADDPG  : wifi={wifi_d:6.1f} dBm  reliability={rel_d:.3f}  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MADDPG", "num_aps": num_aps,
            "mean_wifi_dbm": wifi_d, "reliability": rel_d,
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "num_aps", "mean_wifi_dbm", "reliability",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n[sweep] Done in {time.time()-t_start:.0f}s")
    print(f"[sweep] CSV: {csv_path}")

    from experiments.figure5 import plot_figure5
    plot_figure5(str(csv_path))


if __name__ == "__main__":
    main()
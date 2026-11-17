"""Batch-run the Figure 6 sweep: rate & WiFi interference vs # RIS elements.

⚠ This takes ~30-40 min on CPU (5 element counts × 2 algorithms × 100 episodes).
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


ELEMENT_COUNTS = [10, 20, 30, 40, 50]
EPISODES_TRAIN = 100
EVAL_EPISODES = 5


def _eval_mahdrl(agent, env, n_episodes=EVAL_EPISODES):
    """Return (total_rate_mbps, wifi_dbm)."""
    rates, wifis = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_w = 0.0
        steps = 0
        while not done:
            _, _, action = agent.select_action(obs, epsilon=0.0, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_r += info["total_rate_bps"]
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            steps += 1
        rates.append(ep_r / steps)
        wifis.append(ep_w / steps)
    return float(np.mean(rates)) / 1e6, float(np.mean(wifis))


def _eval_maddpg(agent, env, n_episodes=EVAL_EPISODES):
    rates, wifis = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_w = 0.0
        steps = 0
        while not done:
            _, action = agent.select_action(obs, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_r += info["total_rate_bps"]
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            steps += 1
        rates.append(ep_r / steps)
        wifis.append(ep_w / steps)
    return float(np.mean(rates)) / 1e6, float(np.mean(wifis))


def _train_mahdrl(num_elements: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), ris_elements=num_elements)
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


def _train_maddpg(num_elements: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), ris_elements=num_elements)
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
    csv_path = Path("results/logs/figure6_sweep.csv")

    rows = []
    t_start = time.time()

    for num_elements in ELEMENT_COUNTS:
        print(f"\n[sweep] num_elements = {num_elements}")

        t0 = time.time()
        rate_m, wifi_m = _train_mahdrl(num_elements, EPISODES_TRAIN, seed=0)
        print(f"  MAHDRL  : rate={rate_m:5.2f} Mbps  wifi={wifi_m:6.1f} dBm  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MAHDRL", "num_elements": num_elements,
            "total_rate_mbps": rate_m, "mean_wifi_dbm": wifi_m,
        })

        t0 = time.time()
        rate_d, wifi_d = _train_maddpg(num_elements, EPISODES_TRAIN, seed=0)
        print(f"  MADDPG  : rate={rate_d:5.2f} Mbps  wifi={wifi_d:6.1f} dBm  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MADDPG", "num_elements": num_elements,
            "total_rate_mbps": rate_d, "mean_wifi_dbm": wifi_d,
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "num_elements", "total_rate_mbps", "mean_wifi_dbm",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n[sweep] Done in {time.time()-t_start:.0f}s")
    print(f"[sweep] CSV: {csv_path}")

    from experiments.figure6 import plot_figure6
    plot_figure6(str(csv_path))


if __name__ == "__main__":
    main()
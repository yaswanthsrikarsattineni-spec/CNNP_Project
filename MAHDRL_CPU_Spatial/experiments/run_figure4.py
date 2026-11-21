"""Batch-run the Figure 4 sweep: train one agent per user count, evaluate,
then plot. ~12 minutes on CPU for 6 K values × 2 algorithms.
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


USER_COUNTS = [6, 8, 10, 12, 14, 16]
EPISODES_TRAIN = 100
EVAL_EPISODES = 5


def _evaluate_mahdrl(agent, env, n_episodes=EVAL_EPISODES):
    rates, lats, wifi = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = ep_l = ep_w = 0.0
        steps = 0
        while not done:
            _, _, action = agent.select_action(obs, epsilon=0.0, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_r += float(np.mean(info["per_user_rates_bps"]))
            ep_l += float(np.mean(info["latencies_s"]))
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            steps += 1
        rates.append(ep_r / steps)
        lats.append(ep_l / steps)
        wifi.append(ep_w / steps)
    return float(np.mean(rates)), float(np.mean(lats)), float(np.mean(wifi))


def _evaluate_maddpg(agent, env, n_episodes=EVAL_EPISODES):
    rates, lats, wifi = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_r = ep_l = ep_w = 0.0
        steps = 0
        while not done:
            _, action = agent.select_action(obs, noise_std=0.0)
            obs, _, done, info = env.step(action)
            ep_r += float(np.mean(info["per_user_rates_bps"]))
            ep_l += float(np.mean(info["latencies_s"]))
            ep_w += float(np.mean(info["wifi_psi_dbm"]))
            steps += 1
        rates.append(ep_r / steps)
        lats.append(ep_l / steps)
        wifi.append(ep_w / steps)
    return float(np.mean(rates)), float(np.mean(lats)), float(np.mean(wifi))


def _train_and_eval_mahdrl(K: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), users_per_bs=K)
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MAHDRLAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )
    warmup = agent_cfg.ddpg_warmup_steps
    total = 0
    for ep in range(1, train_eps + 1):
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
    return _evaluate_mahdrl(agent, env)


def _train_and_eval_maddpg(K: int, train_eps: int, seed: int):
    sim_cfg = replace(SimulationConfig(), users_per_bs=K)
    agent_cfg = AgentConfig()
    env = SimulationEnvironment(sim_cfg)
    obs = env.reset()

    agent = MADDPGAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=obs.shape[0], device="cpu", seed=seed,
    )
    warmup = agent_cfg.ddpg_warmup_steps
    total = 0
    for ep in range(1, train_eps + 1):
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
    return _evaluate_maddpg(agent, env)


def main():
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    csv_path = Path("results/logs/figure4_sweep.csv")

    rows = []
    t_start = time.time()

    for K in USER_COUNTS:
        print(f"\n[sweep] K = {K} users")

        # ---- MAHDRL ----
        t0 = time.time()
        rate_m, lat_m, wifi_m = _train_and_eval_mahdrl(K, EPISODES_TRAIN, seed=0)
        print(f"  MAHDRL  : rate/user={rate_m/1e6:5.2f} Mbps  "
              f"latency={lat_m*1e3:5.2f} ms  wifi={wifi_m:6.1f} dBm  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MAHDRL", "K": K,
            "per_user_rate_mbps": rate_m / 1e6,
            "mean_latency_ms": lat_m * 1e3,
            "mean_wifi_dbm": wifi_m,
        })

        # ---- MADDPG ----
        t0 = time.time()
        rate_d, lat_d, wifi_d = _train_and_eval_maddpg(K, EPISODES_TRAIN, seed=0)
        print(f"  MADDPG  : rate/user={rate_d/1e6:5.2f} Mbps  "
              f"latency={lat_d*1e3:5.2f} ms  wifi={wifi_d:6.1f} dBm  "
              f"({time.time()-t0:.0f}s)")
        rows.append({
            "algo": "MADDPG", "K": K,
            "per_user_rate_mbps": rate_d / 1e6,
            "mean_latency_ms": lat_d * 1e3,
            "mean_wifi_dbm": wifi_d,
        })

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "K", "per_user_rate_mbps", "mean_latency_ms", "mean_wifi_dbm",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n[sweep] Done in {time.time()-t_start:.0f}s")
    print(f"[sweep] CSV: {csv_path}")

    # Plot
    from experiments.figure4 import plot_figure4
    plot_figure4(str(csv_path))


if __name__ == "__main__":
    main()
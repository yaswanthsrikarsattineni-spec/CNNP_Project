"""Diagnostic: verify DQN can learn a toy task.

Toy task:
    - Random obs vector in R^20.
    - First 12 dims = per-user signal; last 8 dims = extra noise features.
    - Pick `num_select` = 5 of 12 users.
    - Reward = sum(obs[:12][selected]) * 1e7 (bps-like scale).
    - Optimal policy: pick users 0..4 (highest obs, by construction).
    - Stateless bandit (next_obs = obs).

If DQN works, after training the fraction of the time it selects exactly
{0,1,2,3,4} should rise from near-random (≈1%) to > 80%.
"""
import numpy as np

from config import AgentConfig
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer


def make_env_obs(rng):
    """20-dim obs: obs[0..11] is per-user signal, obs[12..19] is noise."""
    obs = rng.normal(size=20).astype(np.float32)
    obs[:5] += 3.0        # boost users 0..4 (the optimal set)
    return obs


def compute_reward(obs, selected):
    """Reward uses only the per-user signal (first 12 dims of obs)."""
    return float(np.sum(obs[:12] * selected) * 1e7)


def main():
    rng = np.random.default_rng(0)
    cfg = AgentConfig()
    agent = DQNAgent(obs_dim=20, num_users=12, num_select=5,
                     agent_cfg=cfg, device="cpu", seed=0)
    buffer = ReplayBuffer(capacity=10_000, obs_dim=20,
                          action_shape=(12,), seed=0)

    print("Filling buffer with random actions...")
    for _ in range(500):
        obs = make_env_obs(rng)
        selected = np.zeros(12, dtype=np.float32)
        selected[rng.choice(12, size=5, replace=False)] = 1.0
        reward = compute_reward(obs, selected)
        buffer.add(obs, selected, reward, obs, 0.0)

    print("Training DQN on toy task...")
    last_loss = float("nan")
    for step in range(2000):
        obs = make_env_obs(rng)
        if rng.random() < 0.3:
            selected = agent.select_users(obs, epsilon=0.3)
        else:
            selected = np.zeros(12, dtype=np.float32)
            selected[rng.choice(12, size=5, replace=False)] = 1.0
        reward = compute_reward(obs, selected)
        buffer.add(obs, selected, reward, obs, 0.0)

        if buffer.is_ready(cfg.batch_size):
            last_loss = agent.update(buffer.sample(cfg.batch_size))["loss"]

        if (step + 1) % 400 == 0:
            hits = 0
            for _ in range(100):
                obs = make_env_obs(rng)
                sel = agent.select_users(obs)
                if set(np.where(sel == 1)[0]).issubset({0, 1, 2, 3, 4}):
                    hits += 1
            print(f"  step {step+1:4d}  loss={last_loss:8.4f}  "
                  f"optimal-selection-rate={hits:3d}%")

    # Final evaluation
    hits = 0
    for _ in range(200):
        obs = make_env_obs(rng)
        sel = agent.select_users(obs)
        if set(np.where(sel == 1)[0]).issubset({0, 1, 2, 3, 4}):
            hits += 1
    rate = hits / 200 * 100
    print("-" * 60)
    print(f"Final optimal-selection rate: {rate:.1f}%")
    print(f"  (random baseline ≈ 1.0% for exactly matching {{0,1,2,3,4}})")
    print(f"  (ideal DQN ≈ 100%)")
    if rate > 80:
        print("✅ DQN learned the toy task.")
    else:
        print("⚠ DQN did not learn — investigate.")


if __name__ == "__main__":
    main()
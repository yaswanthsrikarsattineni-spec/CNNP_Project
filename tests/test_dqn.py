"""Tests for the DQN agent."""
import numpy as np
import torch

from config import AgentConfig
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer


def _make_agent(obs_dim=20, num_users=12, num_select=10):
    cfg = AgentConfig()
    return DQNAgent(obs_dim, num_users, num_select, cfg, device="cpu", seed=0)


def test_action_has_exactly_num_select_ones():
    agent = _make_agent()
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    selected = agent.select_users(obs, epsilon=0.0)
    assert selected.shape == (12,)
    assert int(selected.sum()) == 10
    assert set(np.unique(selected)).issubset({0.0, 1.0})


def test_greedy_picks_top_n_by_q():
    agent = _make_agent()
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        q = agent.q_net(obs_t).squeeze(0).numpy()
    expected = np.argsort(q)[-10:]
    selected = agent.select_users(obs, epsilon=0.0)
    assert set(np.where(selected == 1)[0]) == set(expected)


def test_epsilon_exploration_can_differ_from_greedy():
    agent = _make_agent()
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    greedies = [
        tuple(np.where(agent.select_users(obs, epsilon=0.0))[0]) for _ in range(5)
    ]
    assert len(set(greedies)) == 1, "greedy must be deterministic"

    randoms = [
        tuple(np.where(agent.select_users(obs, epsilon=1.0))[0]) for _ in range(50)
    ]
    assert len(set(randoms)) > 1, "epsilon=1 must randomize"


def test_loss_decreases_on_fixed_batch():
    """Loss should trend downward over training.

    With REWARD_SCALE=1e-7, the initial loss is already small (~1) — after
    enough iterations on a fixed 64-sample batch, DQN starts overfitting and
    loss can drift back up. We compare windowed averages rather than single
    endpoints to catch genuine learning without being sensitive to the
    overfitting tail.
    """
    agent = _make_agent()
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, 20)).astype(np.float32)
    next_obs = rng.normal(size=(64, 20)).astype(np.float32)
    action = np.zeros((64, 12), dtype=np.float32)
    for b in range(64):
        action[b, rng.choice(12, size=10, replace=False)] = 1.0
    reward = rng.normal(size=64).astype(np.float32) * 1e8   # 10x larger reward
    done = np.zeros(64, dtype=np.float32)
    batch = (obs, action, reward, next_obs, done)

    losses = [agent.update(batch)["loss"] for _ in range(200)]
    first_window = float(np.mean(losses[:20]))
    middle_window = float(np.mean(losses[80:100]))   # skip initial transient
    assert middle_window < first_window * 0.5, (
        f"Loss did not decrease enough: first20_avg={first_window:.4f}, "
        f"mid80-100_avg={middle_window:.4f}"
    )


def test_reward_scale_prevents_explosion():
    """Loss on bps-scale reward should be O(1e6)-ish, not 1e14."""
    agent = _make_agent()
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, 20)).astype(np.float32)
    action = np.zeros((32, 12), dtype=np.float32)
    action[:, :10] = 1.0
    next_obs = rng.normal(size=(32, 20)).astype(np.float32)
    reward = np.full(32, 2e7, dtype=np.float32)   # 20 Mbps per step
    done = np.zeros(32, dtype=np.float32)
    loss = agent.update((obs, action, reward, next_obs, done))["loss"]
    assert 1e-3 < loss < 1e6, f"Loss out of scale: {loss}"


def test_save_load_roundtrip(tmp_path):
    agent = _make_agent()
    path = tmp_path / "dqn.pt"
    agent.save(str(path))

    agent2 = _make_agent()
    agent2.load(str(path))
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    a1 = agent.select_users(obs)
    a2 = agent2.select_users(obs)
    assert np.array_equal(a1, a2)
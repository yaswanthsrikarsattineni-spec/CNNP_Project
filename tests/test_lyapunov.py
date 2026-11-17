"""Tests for the Lyapunov virtual queue (Eq. 21) and bonus (Eq. 26)."""
import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment


def _env():
    cfg = SimulationConfig()
    return cfg, SimulationEnvironment(cfg)


def test_virtual_queue_starts_at_zero():
    cfg, env = _env()
    env.reset()
    assert np.all(env.virtual_queues == 0.0)


def test_virtual_queue_stays_zero_when_queue_small():
    cfg, env = _env()
    env.reset()
    for t in range(50):
        action = env.make_mrt_action(num_selected=10, offset=t % cfg.users_per_bs)
        env.step(action)
    # All queues should be far below q_max=3e6, so Y must be exactly 0
    assert np.all(env.virtual_queues == 0.0), (
        f"Expected Y=0 (queues small), got {env.virtual_queues}"
    )


def test_virtual_queue_grows_when_queue_exceeds_max():
    cfg, env = _env()
    env.reset()
    # Force a queue above q_max
    env.queue.queues = np.full(cfg.users_per_bs, 4e6)
    # Do 3 steps — Y should grow by (4e6 − 3e6) = 1e6 each step
    for _ in range(3):
        env.queue.queues = np.full(cfg.users_per_bs, 4e6)  # re-force each step
        env.step(env.make_mrt_action(num_selected=10))
    # After 3 forced updates, Y ≈ 3e6 per user (minus queue depletion each step)
    assert np.all(env.virtual_queues > 1e6), (
        f"Expected Y > 1e6, got {env.virtual_queues}"
    )


def test_lyapunov_bonus_is_positive_when_queues_nonzero():
    cfg, env = _env()
    env.reset()
    # Force some non-zero queues
    env.queue.queues = np.full(cfg.users_per_bs, 1e5)
    _, _, _, info = env.step(env.make_mrt_action(num_selected=10))
    assert info["lyapunov_bonus_bps"] > 0, (
        f"Expected positive Lyapunov bonus, got {info['lyapunov_bonus_bps']}"
    )


def test_lyapunov_bonus_is_zero_when_rates_zero():
    cfg, env = _env()
    env.reset()
    env.queue.queues = np.full(cfg.users_per_bs, 1e5)
    # Zero action → all rates = 0 → no bonus
    _, _, _, info = env.step(env._zero_action())
    assert info["lyapunov_bonus_bps"] == 0.0


def test_lambda_equals_q_plus_Y():
    cfg, env = _env()
    env.reset()
    env.queue.queues = np.full(cfg.users_per_bs, 5e4)
    env.virtual_queues = np.full(cfg.users_per_bs, 2e5)
    _, _, _, info = env.step(env.make_mrt_action(num_selected=10))
    expected = 5e4 + 2e5
    assert np.allclose(info["lambda_bits"], expected)


def test_total_reward_equals_base_plus_lyapunov():
    cfg, env = _env()
    env.reset()
    env.queue.queues = np.full(cfg.users_per_bs, 1e5)
    _, reward, _, info = env.step(env.make_mrt_action(num_selected=10))
    expected = info["base_reward_bps"] + info["lyapunov_bonus_bps"]
    assert np.isclose(reward, expected), (
        f"reward={reward}, base+lyap={expected}"
    )
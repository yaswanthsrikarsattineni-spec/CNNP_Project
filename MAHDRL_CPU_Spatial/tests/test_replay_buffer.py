"""Tests for the replay buffer."""
import numpy as np
import pytest

from agents.replay_buffer import ReplayBuffer


def test_starts_empty():
    buf = ReplayBuffer(capacity=100, obs_dim=8, action_shape=(5,))
    assert len(buf) == 0
    assert not buf.is_ready(1)


def test_add_and_len():
    buf = ReplayBuffer(capacity=100, obs_dim=8, action_shape=(5,))
    for _ in range(10):
        buf.add(
            obs=np.zeros(8),
            action=np.zeros(5),
            reward=1.0,
            next_obs=np.zeros(8),
            done=0.0,
        )
    assert len(buf) == 10


def test_capacity_wrap():
    """Ring buffer must overwrite the oldest entries."""
    buf = ReplayBuffer(capacity=5, obs_dim=4, action_shape=(2,))
    for i in range(10):
        buf.add(
            obs=np.full(4, float(i)),
            action=np.array([float(i), float(i)]),
            reward=float(i),
            next_obs=np.full(4, float(i + 1)),
            done=0.0,
        )
    assert len(buf) == 5
    # After 10 writes to a 5-slot buffer, we should hold i = 5..9.
    # Check the stored reward array directly (not a random sample, which
    # may repeat indices with replacement).
    assert sorted(buf.reward_buf.tolist()) == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_sample_shape():
    buf = ReplayBuffer(capacity=100, obs_dim=6, action_shape=(3,), seed=0)
    for _ in range(20):
        buf.add(np.zeros(6), np.zeros(3), 0.0, np.zeros(6), 0.0)
    obs, action, reward, next_obs, done = buf.sample(8)
    assert obs.shape == (8, 6)
    assert action.shape == (8, 3)
    assert reward.shape == (8,)
    assert next_obs.shape == (8, 6)
    assert done.shape == (8,)


def test_sample_raises_when_insufficient():
    buf = ReplayBuffer(capacity=100, obs_dim=4, action_shape=(2,))
    for _ in range(3):
        buf.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), 0.0)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_is_ready():
    buf = ReplayBuffer(capacity=100, obs_dim=4, action_shape=(2,))
    for _ in range(5):
        buf.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), 0.0)
    assert buf.is_ready(5)
    assert not buf.is_ready(6)
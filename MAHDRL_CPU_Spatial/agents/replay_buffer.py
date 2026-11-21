"""Generic replay buffer for DQN / DDPG / MAHDRL training.

Stores transitions (obs, action, reward, next_obs, done) in fixed-size numpy
arrays. Ring-buffer pointer, uniform random sampling. Deterministic.

The `action_shape` is a tuple. Examples:
    DQN   : (K,)              — binary user-selection mask
    DDPG  : (N_ant*K + S*L,)  — continuous beamformer + phases
    Combined MAHDRL uses two separate buffers.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_shape: tuple,
        seed: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_shape = tuple(action_shape)
        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.reward_buf = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = np.asarray(action, dtype=np.float32)
        self.reward_buf[self.ptr] = float(reward)
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: {self.size} < {batch_size}"
            )
        idxs = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.action_buf[idxs],
            self.reward_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
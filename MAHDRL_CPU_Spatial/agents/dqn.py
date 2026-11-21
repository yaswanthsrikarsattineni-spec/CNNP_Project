"""DQN agent — upper-level user selection (paper Eq. 18, x_m).

Design:
    - Input : state observation vector.
    - Output: K Q-values q_1 ... q_K, one per user.
    - Action: top-N_ant users by Q-value.
    - Update: Bellman TD error on the aggregate Q of the selected set.

⚠ Implementation choices (not in paper):
    - Aggregate-Q formulation for multi-binary actions.
    - Reward scaling factor 1e-6 so Q-values are in Mbps units.
    - Epsilon-greedy as "swap one bit" rather than "random action".
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from config import AgentConfig


REWARD_SCALE = 1e-7   # DQN learns on Mbps units, not bps.


class QNetwork(nn.Module):
    """MLP: obs → hidden → K Q-values."""

    def __init__(self, obs_dim: int, num_users: int, hidden_dims: tuple) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_users))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN agent for discrete user selection."""

    def __init__(
        self,
        obs_dim: int,
        num_users: int,
        num_select: int,
        agent_cfg: AgentConfig,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_users = num_users
        self.num_select = num_select
        self.cfg = agent_cfg

        # Fall back to CPU if CUDA requested but unavailable.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, num_users, agent_cfg.hidden_dims).to(self.device)
        self.target_net = QNetwork(obs_dim, num_users, agent_cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=agent_cfg.learning_rate
        )
        self.loss_fn = nn.MSELoss()

        self.global_step = 0
        self.rng = np.random.default_rng(seed)

    # ==================================================================
    def select_users(self, obs: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Return a binary mask of length K selecting `num_select` users."""
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(obs_t).squeeze(0).cpu().numpy()

        if self.rng.random() < epsilon:
            # Swap one selected ↔ one unselected user (keeps top-N constraint).
            order = np.argsort(q)[::-1]
            top = order[: self.num_select].tolist()
            bottom = order[self.num_select:].tolist()
            if bottom:
                to_deselect = self.rng.choice(top)
                to_select = self.rng.choice(bottom)
                top.remove(to_deselect)
                top.append(int(to_select))
            idx = np.array(top)
        else:
            idx = np.argsort(q)[-self.num_select:]

        selected = np.zeros(self.num_users, dtype=np.float32)
        selected[idx] = 1.0
        return selected

    # ==================================================================
    def update(self, batch) -> Dict[str, float]:
        """One gradient step. Batch = (obs, action, reward, next_obs, done)."""
        obs, action, reward, next_obs, done = batch

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device) * REWARD_SCALE
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # Current Q of the actually-taken action.
        q_all = self.q_net(obs_t)                              # (B, K)
        q_action = (q_all * action_t).sum(dim=1)               # (B,)

        # Greedy target over next state.
        with torch.no_grad():
            next_q = self.target_net(next_obs_t)               # (B, K)
            top_n_q, _ = torch.topk(next_q, self.num_select, dim=1)
            q_next_target = top_n_q.sum(dim=1)                 # (B,)
            target = reward_t + self.cfg.discount_factor * (1.0 - done_t) * q_next_target

        loss = self.loss_fn(q_action, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item())}

    # ==================================================================
    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
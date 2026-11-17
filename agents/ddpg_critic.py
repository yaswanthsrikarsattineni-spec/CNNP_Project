"""DDPG Critic — Q-value network for continuous + discrete actions.

Training stabilizers:
  - slower target soft-update (ddpg_target_tau)
  - L2 weight decay (critic_weight_decay)
  - tight gradient clipping (max_norm=1.0)
  - **NEW** Bellman target clipping (ddpg_target_clip) — prevents Q blow-up
    on out-of-distribution states the buffer has not seen.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from config import AgentConfig


REWARD_SCALE = 1e-7


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DDPGCritic:
    def __init__(
        self,
        obs_dim: int,
        num_users: int,
        continuous_dim: int,
        agent_cfg: AgentConfig,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_users = num_users
        self.continuous_dim = continuous_dim
        self.cfg = agent_cfg
        self.input_dim = obs_dim + num_users + continuous_dim

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.q_net = CriticNetwork(self.input_dim, agent_cfg.hidden_dims).to(self.device)
        self.target_q_net = CriticNetwork(self.input_dim, agent_cfg.hidden_dims).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=agent_cfg.learning_rate,
            weight_decay=agent_cfg.critic_weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    def compute_q_from_tensors(
        self,
        obs: torch.Tensor,
        discrete: torch.Tensor,
        continuous: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        inp = torch.cat([obs, discrete, continuous], dim=-1)
        net = self.target_q_net if use_target else self.q_net
        return net(inp)

    def compute_q(
        self, obs: np.ndarray, discrete: np.ndarray, continuous: np.ndarray
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        disc_t = torch.as_tensor(discrete, dtype=torch.float32, device=self.device)
        cont_t = torch.as_tensor(continuous, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.compute_q_from_tensors(obs_t, disc_t, cont_t)
        return q.cpu().numpy()

    def update(self, actor, batch) -> Dict[str, float]:
        obs, action, reward, next_obs, done = batch

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device) * REWARD_SCALE
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        disc_t = act_t[:, : self.num_users]
        cont_t = act_t[:, self.num_users :]

        q_current = self.compute_q_from_tensors(obs_t, disc_t, cont_t)

        with torch.no_grad():
            next_disc_t = disc_t
            inp_next = torch.cat([next_obs_t, next_disc_t], dim=-1)
            next_raw = actor.target_actor_net(inp_next)
            q_next = self.compute_q_from_tensors(
                next_obs_t, next_disc_t, next_raw, use_target=True
            )
            target = reward_t + self.cfg.discount_factor * (1.0 - done_t) * q_next
            # Clipping prevents runaway Q values on unseen states.
            target = torch.clamp(
                target, -self.cfg.ddpg_target_clip, self.cfg.ddpg_target_clip
            )

        loss = self.loss_fn(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update_target()

        return {
            "critic_loss": float(loss.item()),
            "q_abs_mean": float(q_current.abs().mean().item()),
        }

    def _soft_update_target(self) -> None:
        tau = self.cfg.ddpg_target_tau
        with torch.no_grad():
            for tp, p in zip(
                self.target_q_net.parameters(), self.q_net.parameters()
            ):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_q_net.load_state_dict(ckpt["target_q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
"""QMIX mixing network — paper Eq. 34, 35, 41.

Combines per-agent Q-values {Q_m} into a team Q-value Q_tot via a hypernetwork
whose weights are forced non-negative (guaranteeing Eq. 35: ∂Q_tot/∂Q_m ≥ 0).

⚠ Implementation choices (not in paper):
    - No tanh on the output. tanh saturation kills gradients; we observed
      actor_loss = -0.000 exactly when the raw mixer output exceeded 4.
    - Last layer of hyper_w2 / hyper_b2 is small-initialized (std=0.001)
      so Q_tot starts near 0 and grows toward the TD target naturally.
    - L2 regularization on Q_tot is added in the training loss (see MAHDRL).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """QMIX monotonic mixing network (no output tanh)."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # Small init on output layer → Q_tot starts near 0, grows naturally.
        nn.init.normal_(self.hyper_w2[-1].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.hyper_w2[-1].bias, mean=0.0, std=0.001)
        nn.init.normal_(self.hyper_b2[-1].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.hyper_b2[-1].bias, mean=0.0, std=0.001)

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = agent_qs.shape[0]
        q_vec = agent_qs.view(B, 1, self.num_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(B, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_vec, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        raw = torch.bmm(hidden, w2) + b2

        return raw.view(B)


class QMixer:
    """Wrapper with target network, optimizer, save/load."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        learning_rate: float = 1e-4,
        tau: float = 0.0005,
        embed_dim: int = 32,
        hypernet_hidden: int = 64,
        device: str = "cpu",
    ) -> None:
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.tau = tau

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.mix_net = MixingNetwork(
            num_agents, state_dim, embed_dim, hypernet_hidden
        ).to(self.device)
        self.target_mix_net = MixingNetwork(
            num_agents, state_dim, embed_dim, hypernet_hidden
        ).to(self.device)
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())
        self.target_mix_net.eval()

        self.optimizer = torch.optim.Adam(
            self.mix_net.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.MSELoss()

    def compute_q_tot(
        self, agent_qs: np.ndarray, state: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        aq_t = torch.as_tensor(agent_qs, dtype=torch.float32, device=self.device)
        st_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        squeeze = False
        if aq_t.dim() == 1:
            aq_t = aq_t.unsqueeze(0)
            st_t = st_t.unsqueeze(0)
            squeeze = True
        net = self.target_mix_net if use_target else self.mix_net
        with torch.no_grad():
            q = net(aq_t, st_t)
        return (q.squeeze(0) if squeeze else q).cpu().numpy()

    def update(
        self,
        agent_qs: np.ndarray,
        state: np.ndarray,
        reward: np.ndarray,
        next_agent_qs: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        discount: float,
        reward_scale: float = 1e-7,
        l2_penalty: float = 0.01,
    ) -> dict:
        aq_t = torch.as_tensor(agent_qs, dtype=torch.float32, device=self.device)
        st_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        r_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device) * reward_scale
        naq_t = torch.as_tensor(next_agent_qs, dtype=torch.float32, device=self.device)
        nst_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q_tot = self.mix_net(aq_t, st_t)
        with torch.no_grad():
            q_tot_next = self.target_mix_net(naq_t, nst_t)
            target = r_t + discount * (1.0 - d_t) * q_tot_next

        loss = self.loss_fn(q_tot, target) + l2_penalty * (q_tot ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.soft_update_target()
        return {"mixer_loss": float(loss.item())}

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for tp, p in zip(
                self.target_mix_net.parameters(), self.mix_net.parameters()
            ):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

    def save(self, path: str) -> None:
        torch.save(
            {
                "mix_net": self.mix_net.state_dict(),
                "target_mix_net": self.target_mix_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.mix_net.load_state_dict(ckpt["mix_net"])
        self.target_mix_net.load_state_dict(ckpt["target_mix_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
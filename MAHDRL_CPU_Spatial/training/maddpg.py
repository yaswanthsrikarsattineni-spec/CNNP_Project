"""MADDPG baseline — paper's comparison algorithm.

MADDPG relaxes the discrete user selection into a continuous space. So the
whole action (user scores, power logits, RIS phases) is output by a single
DDPG-style actor. No DQN, no QMIX mixer.

⚠ Implementation assumption:
    Paper does not specify the exact continuous relaxation. We use
    "continuous scores + top-N" — the most common relaxation in the DRL-for-
    wireless literature (used e.g. by Mao et al. 2018, "A survey on mobile
    edge computing").

Actor output (dim = 44 for K=12, S·L=20):
    [0:K]     user scores    →  top-N_ant selection
    [K:2K]    power logits   →  softmax over selected
    [2K:2K+SL] phase raws    →  (raw + 1) · π
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from config import SimulationConfig, AgentConfig
from agents.replay_buffer import ReplayBuffer


class MADDPGActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))


class MADDPGCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MADDPGAgent:
    """MADDPG baseline for MAHDRL comparison."""

    def __init__(
        self,
        simulation_cfg: SimulationConfig,
        agent_cfg: AgentConfig,
        obs_dim: int = 320,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = simulation_cfg
        self.agent_cfg = agent_cfg
        self.obs_dim = obs_dim
        self.device = device
        self.seed = seed

        K = simulation_cfg.users_per_bs
        N_ant = simulation_cfg.antennas_per_bs
        S = simulation_cfg.ris_per_bs
        L = simulation_cfg.ris_elements
        self.K, self.N_ant = K, N_ant
        self.num_select = min(K, N_ant)
        self.S, self.L = S, L
        self.max_power_w = 10.0 ** ((simulation_cfg.max_bs_power_dbm - 30.0) / 10.0)
        self.wifi_device_count = simulation_cfg.num_wifi_aps * simulation_cfg.wifi_devices_per_ap
        self.channel_obs_scale = simulation_cfg.channel_obs_scale
        self.channel_obs_start = 4 * K + self.wifi_device_count

        # Total continuous output dim
        self.output_dim = K + K + S * L  # 44

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # Networks
        hidden = agent_cfg.hidden_dims
        self.actor = MADDPGActorNet(obs_dim, self.output_dim, hidden).to(self.device)
        self.actor_target = MADDPGActorNet(obs_dim, self.output_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_learning_rate
        )

        self.critic = MADDPGCriticNet(
            obs_dim + self.output_dim, hidden
        ).to(self.device)
        self.critic_target = MADDPGCriticNet(
            obs_dim + self.output_dim, hidden
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.learning_rate,
            weight_decay=agent_cfg.critic_weight_decay,
        )

        # Buffer
        self.buffer = ReplayBuffer(
            capacity=agent_cfg.replay_capacity,
            obs_dim=obs_dim,
            action_shape=(self.output_dim,),
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)
        self.global_step = 0

    # ==================================================================
    def _raw_to_env_action(self, raw: np.ndarray, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert raw continuous vector → env action dict."""
        raw = np.clip(raw, -1.0, 1.0)
        scores = raw[: self.K]
        power_logits = raw[self.K : 2 * self.K]
        phase_raw = raw[2 * self.K :].reshape(self.S, self.L)

        # Top-N selection
        top_idx = np.argsort(scores)[-self.num_select:]
        selected = np.zeros(self.K, dtype=bool)
        selected[top_idx] = True

        # Softmax power over selected
        power = np.zeros(self.K, dtype=np.float64)
        if selected.any():
            logits = power_logits[selected] - power_logits[selected].max()
            exp = np.exp(logits)
            power[selected] = exp / (exp.sum() + 1e-12)

        # Extract channel from obs (same layout as DDPGActor)
        K, N = self.K, self.N_ant
        start = self.channel_obs_start
        size = K * N
        c_re = obs[start : start + size].reshape(K, N).astype(np.float64)
        c_im = obs[start + size : start + 2 * size].reshape(K, N).astype(np.float64)
        channels = (c_re + 1j * c_im) * self.channel_obs_scale

        # MRT direction
        norms = np.maximum(np.linalg.norm(channels, axis=1, keepdims=True), 1e-12)
        mrt = channels.conj() / norms

        beamformers = np.zeros((K, N), dtype=np.complex128)
        for k in range(K):
            if selected[k] and power[k] > 0:
                beamformers[k] = np.sqrt(self.max_power_w * power[k]) * mrt[k]

        phases = (phase_raw + 1.0) * np.pi

        return {
            "user_selection": selected,
            "beamformers": beamformers,
            "ris_phases": phases,
        }

    def select_action(
        self, obs: np.ndarray, noise_std: float = 0.0
    ):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            raw = np.clip(raw + self.rng.normal(0.0, noise_std, size=raw.shape), -1.0, 1.0)
        return raw, self._raw_to_env_action(raw, obs)

    def store(self, obs, raw, reward, next_obs, done):
        self.buffer.add(obs, raw, reward, next_obs, done)

    # ==================================================================
    def train_step(self) -> Optional[Dict[str, float]]:
        if not self.buffer.is_ready(self.agent_cfg.batch_size):
            return None

        obs, act, reward, next_obs, done = self.buffer.sample(self.agent_cfg.batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        r_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device) * 1e-7
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_raw = self.actor_target(next_obs_t)
            q_next = self.critic_target(torch.cat([next_obs_t, next_raw], dim=-1))
            target = r_t + self.agent_cfg.discount_factor * (1.0 - d_t) * q_next
            target = torch.clamp(target, -self.agent_cfg.ddpg_target_clip,
                                 self.agent_cfg.ddpg_target_clip)
        q_current = self.critic(torch.cat([obs_t, act_t], dim=-1))
        critic_loss = nn.functional.mse_loss(q_current, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ---- Actor update ----
        raw_new = self.actor(obs_t)
        q_actor = self.critic(torch.cat([obs_t, raw_new], dim=-1))
        actor_loss = -q_actor.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ---- Target soft updates ----
        with torch.no_grad():
            tau = self.agent_cfg.ddpg_target_tau
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

        self.global_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(q_current.mean().item()),
        }

    # ==================================================================
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
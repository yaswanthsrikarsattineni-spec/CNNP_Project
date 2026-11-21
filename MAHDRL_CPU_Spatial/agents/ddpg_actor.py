"""DDPG Actor — power-weight parameterization with MRT direction (Option A).

The paper (Eq. 18) defines the action as (w_m, θ_m) with complex beamformers.
Learning raw complex coefficients from scratch is sample-inefficient; we adopt
the standard DRL-for-wireless parameterization (also used in IEEE references
[16], [20] of the paper):

    w_k = sqrt(P_max · softmax(power_logits)_k) · h_k^* / ||h_k||
                       ↑ learned                       ↑ MRT (fixed by physics)

The actor's 32-dim output:
    [0:12]    power_logits  →  softmax over selected users
    [12:32]   phase_raw     →  (raw + 1) · π  ∈ [0, 2π]

The channel is extracted from the observation (paper Eq. 17 includes it),
so the actor can compute MRT for each user without extra API.

This preserves Eq. 18's action space exactly — only the parameterization
of w_m differs.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from config import AgentConfig


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class DDPGActor:
    def __init__(
        self,
        obs_dim: int,
        num_users: int,
        num_ant: int,
        num_ris: int,
        ris_elements: int,
        max_power_w: float,
        wifi_device_count: int,
        channel_obs_scale: float,
        agent_cfg: AgentConfig,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        self.obs_dim = obs_dim
        self.num_users = num_users
        self.num_ant = num_ant
        self.num_ris = num_ris
        self.ris_elements = ris_elements
        self.max_power_w = max_power_w
        self.wifi_device_count = wifi_device_count
        self.channel_obs_scale = channel_obs_scale
        self.cfg = agent_cfg

        # Layout of the channel block inside obs (must match env._build_observation).
        # obs = [4K+J summary features, channel_re (K·N), channel_im (K·N)]
        self.channel_obs_start = 4 * num_users + wifi_device_count

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.input_dim = obs_dim + num_users          # obs ⊕ discrete
        self.power_dim = num_users                    # power logits
        self.phase_dim = num_ris * ris_elements
        self.output_dim = self.power_dim + self.phase_dim  # = 32 for K=12, S·L=20

        self.actor_net = ActorNetwork(
            self.input_dim, self.output_dim, agent_cfg.hidden_dims
        ).to(self.device)
        self.target_actor_net = ActorNetwork(
            self.input_dim, self.output_dim, agent_cfg.hidden_dims
        ).to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_actor_net.eval()

        self.optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=agent_cfg.actor_learning_rate
        )
        self.rng = np.random.default_rng(seed)

    # ==================================================================
    def _raw_output(
        self, obs: np.ndarray, discrete_action: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        disc_t = torch.as_tensor(discrete_action, dtype=torch.float32, device=self.device)
        inp = torch.cat([obs_t, disc_t], dim=-1).unsqueeze(0)
        net = self.target_actor_net if use_target else self.actor_net
        with torch.no_grad():
            return net(inp).squeeze(0).cpu().numpy()

    # ==================================================================
    def _extract_channel(self, obs: np.ndarray) -> np.ndarray:
        """Recover (K, N_ant) complex channel from the observation."""
        K, N = self.num_users, self.num_ant
        start = self.channel_obs_start
        size = K * N
        re = obs[start : start + size].reshape(K, N).astype(np.float64)
        im = obs[start + size : start + 2 * size].reshape(K, N).astype(np.float64)
        return (re + 1j * im) * self.channel_obs_scale

    def _mrt_direction(self, channels: np.ndarray) -> np.ndarray:
        """MRT: w_dir_k = h_k^* / ||h_k||, shape (K, N_ant), unit-norm rows."""
        norms = np.linalg.norm(channels, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return channels.conj() / norms

    # ==================================================================
    def _postprocess(
        self, raw: np.ndarray, discrete_action: np.ndarray, obs: np.ndarray
    ) -> dict:
        raw = np.clip(raw, -1.0, 1.0)

        power_logits = raw[: self.power_dim]                              # (K,)
        phase_raw = raw[self.power_dim:].reshape(self.num_ris, self.ris_elements)

        # Softmax power allocation over selected users only.
        selected = discrete_action.astype(bool)
        power_weights = np.zeros(self.num_users, dtype=np.float64)
        if selected.any():
            logits_sel = power_logits[selected]
            logits_sel = logits_sel - logits_sel.max()                    # stability
            exp_sel = np.exp(logits_sel)
            power_weights[selected] = exp_sel / (exp_sel.sum() + 1e-12)

        # MRT direction from channel (physics, not learned).
        channels = self._extract_channel(obs)                             # (K, N)
        mrt = self._mrt_direction(channels)                               # (K, N)

        beamformers = np.zeros((self.num_users, self.num_ant), dtype=np.complex128)
        for k in range(self.num_users):
            if selected[k] and power_weights[k] > 0.0:
                amp = np.sqrt(self.max_power_w * power_weights[k])
                beamformers[k] = amp * mrt[k]

        phases = (phase_raw + 1.0) * np.pi

        return {
            "user_selection": selected,
            "beamformers": beamformers,
            "ris_phases": phases,
        }

    # ==================================================================
    def select_action(
        self, obs: np.ndarray, discrete_action: np.ndarray, noise_std: float = 0.0
    ) -> dict:
        raw = self._raw_output(obs, discrete_action)
        if noise_std > 0.0:
            raw = raw + self.rng.normal(0.0, noise_std, size=raw.shape)
            raw = np.clip(raw, -1.0, 1.0)
        return self._postprocess(raw, discrete_action, obs)

    def select_action_and_raw(
        self, obs: np.ndarray, discrete_action: np.ndarray, noise_std: float = 0.0
    ):
        raw = self._raw_output(obs, discrete_action)
        if noise_std > 0.0:
            raw = raw + self.rng.normal(0.0, noise_std, size=raw.shape)
            raw = np.clip(raw, -1.0, 1.0)
        return raw, self._postprocess(raw, discrete_action, obs)

    def select_target_action(
        self, obs: np.ndarray, discrete_action: np.ndarray
    ) -> dict:
        raw = self._raw_output(obs, discrete_action, use_target=True)
        return self._postprocess(raw, discrete_action, obs)

    # ==================================================================
    def soft_update_target(self) -> None:
        tau = self.cfg.ddpg_target_tau
        with torch.no_grad():
            for tp, p in zip(
                self.target_actor_net.parameters(), self.actor_net.parameters()
            ):
                tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    # ==================================================================
    def update(self, critic, batch) -> Dict[str, float]:
        obs, action, reward, next_obs, done = batch

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        disc_t = act_t[:, : self.num_users]

        inp = torch.cat([obs_t, disc_t], dim=-1)
        raw_new = self.actor_net(inp)
        q = critic.compute_q_from_tensors(obs_t, disc_t, raw_new)
        loss = -q.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.soft_update_target()
        return {"actor_loss": float(loss.item())}

    # ==================================================================
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor_net.state_dict(),
                "target_actor": self.target_actor_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor_net.load_state_dict(ckpt["actor"])
        self.target_actor_net.load_state_dict(ckpt["target_actor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
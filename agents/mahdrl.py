"""Full MAHDRL agent — implements paper Algorithm 1.

⚠ M=1 bypass: for single agent, QMIX is mathematically degenerate
(Q_tot = Q_m trivially satisfies Eq. 34/35). Training through a redundant
mixer caused Q_m drift (observed −90 in practice). We bypass the mixer
for M=1 and keep it for future multi-BS experiments.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import SimulationConfig, AgentConfig
from agents.dqn import DQNAgent
from agents.ddpg_actor import DDPGActor
from agents.ddpg_critic import DDPGCritic
from agents.mixing_network import QMixer
from agents.replay_buffer import ReplayBuffer


class MAHDRLAgent:
    def __init__(
        self,
        simulation_cfg: SimulationConfig,
        agent_cfg: AgentConfig | None = None,
        obs_dim: int = 320,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        self.cfg = simulation_cfg
        self.agent_cfg = agent_cfg or AgentConfig()
        self.obs_dim = obs_dim
        self.device = device
        self.seed = seed

        K = simulation_cfg.users_per_bs
        N_ant = simulation_cfg.antennas_per_bs
        S = simulation_cfg.ris_per_bs
        L = simulation_cfg.ris_elements
        wifi_dev = simulation_cfg.num_wifi_aps * simulation_cfg.wifi_devices_per_ap

        # ---- Upper level: DQN for discrete user selection ----
        self.dqn = DQNAgent(
            obs_dim=obs_dim,
            num_users=K,
            num_select=min(K, N_ant),
            agent_cfg=self.agent_cfg,
            device=device,
            seed=seed,
        )

        # ---- Lower level: DDPG actor + critic ----
        p_tx_w = 10.0 ** ((simulation_cfg.max_bs_power_dbm - 30.0) / 10.0)
        self.actor = DDPGActor(
            obs_dim=obs_dim,
            num_users=K,
            num_ant=N_ant,
            num_ris=S,
            ris_elements=L,
            max_power_w=p_tx_w,
            wifi_device_count=wifi_dev,
            channel_obs_scale=simulation_cfg.channel_obs_scale,
            agent_cfg=self.agent_cfg,
            device=device,
            seed=seed,
        )
        self.critic = DDPGCritic(
            obs_dim=obs_dim,
            num_users=K,
            continuous_dim=self.actor.output_dim,
            agent_cfg=self.agent_cfg,
            device=device,
        )

        # ---- QMIX mixer (bypassed for M=1) ----
        self.num_agents = 1
        self.mixer: QMixer | None = None

        # ---- Replay buffer ----
        self.buffer = ReplayBuffer(
            capacity=self.agent_cfg.replay_capacity,
            obs_dim=obs_dim,
            action_shape=(K + self.actor.output_dim,),
            seed=seed,
        )

        self.num_users = K
        self.global_step = 0

    # ==================================================================
    def select_action(
        self, obs: np.ndarray, epsilon: float = 0.0, noise_std: float = 0.0
    ):
        disc = self.dqn.select_users(obs, epsilon=epsilon)
        raw, env_action = self.actor.select_action_and_raw(
            obs, disc, noise_std=noise_std
        )
        return disc, raw, env_action

    def store(
        self, obs, disc, raw, reward, next_obs, done
    ) -> None:
        action = np.concatenate([disc, raw], axis=0).astype(np.float32)
        self.buffer.add(obs, action, reward, next_obs, done)

    # ==================================================================
    def train_step(self) -> Optional[Dict[str, float]]:
        if not self.buffer.is_ready(self.agent_cfg.batch_size):
            return None

        batch = self.buffer.sample(self.agent_cfg.batch_size)
        obs, action, reward, next_obs, done = batch
        K = self.num_users

        dqn_batch = (
            obs,
            action[:, :K].astype(np.float32),
            reward,
            next_obs,
            done,
        )
        dqn_info = self.dqn.update(dqn_batch)
        cm_info = self._train_critic_and_mixer(batch)
        actor_info = self.actor.update(self.critic, batch)

        self.global_step += 1
        return {
            "dqn_loss": dqn_info["loss"],
            "critic_mixer_loss": cm_info["critic_mixer_loss"],
            "q_tot_mean": cm_info["q_tot_mean"],
            "actor_loss": actor_info["actor_loss"],
        }

    def _train_critic_and_mixer(self, batch) -> Dict[str, float]:
        obs, action, reward, next_obs, done = batch
        K = self.num_users

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.critic.device)
        disc_t = torch.as_tensor(action[:, :K], dtype=torch.float32, device=self.critic.device)
        cont_t = torch.as_tensor(action[:, K:], dtype=torch.float32, device=self.critic.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.critic.device) * 1e-7
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.critic.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.critic.device)

        q_m = self.critic.compute_q_from_tensors(obs_t, disc_t, cont_t)
        q_tot = q_m if self.mixer is None else self.mixer.mix_net(q_m.unsqueeze(-1), obs_t)

        with torch.no_grad():
            next_q_dqn = self.dqn.target_net(next_obs_t)
            top_idx = torch.topk(next_q_dqn, self.dqn.num_select, dim=1).indices
            next_disc_t = torch.zeros_like(next_q_dqn)
            next_disc_t.scatter_(1, top_idx, 1.0)

            inp_next = torch.cat([next_obs_t, next_disc_t], dim=-1)
            next_raw = self.actor.target_actor_net(inp_next)

            next_q_m = self.critic.compute_q_from_tensors(
                next_obs_t, next_disc_t, next_raw, use_target=True
            )
            q_tot_next = next_q_m if self.mixer is None else self.mixer.target_mix_net(
                next_q_m.unsqueeze(-1), next_obs_t
            )

            target = (
                reward_t
                + self.agent_cfg.discount_factor * (1.0 - done_t) * q_tot_next
            )
            target = torch.clamp(
                target,
                -self.agent_cfg.ddpg_target_clip,
                self.agent_cfg.ddpg_target_clip,
            )

        loss = F.mse_loss(q_tot, target)
        if self.mixer is not None:
            loss = loss + self.agent_cfg.mixer_l2_penalty * (q_tot ** 2).mean()

        self.critic.optimizer.zero_grad()
        if self.mixer is not None:
            self.mixer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.q_net.parameters(), max_norm=1.0)
        if self.mixer is not None:
            torch.nn.utils.clip_grad_norm_(self.mixer.mix_net.parameters(), max_norm=1.0)
        self.critic.optimizer.step()
        if self.mixer is not None:
            self.mixer.optimizer.step()

        self.critic._soft_update_target()
        if self.mixer is not None:
            self.mixer.soft_update_target()

        return {
            "critic_mixer_loss": float(loss.item()),
            "q_tot_mean": float(q_tot.mean().item()),
        }

    # ==================================================================
    def save(self, path: str) -> None:
        data = {
            "dqn_q": self.dqn.q_net.state_dict(),
            "dqn_target": self.dqn.target_net.state_dict(),
            "dqn_optimizer": self.dqn.optimizer.state_dict(),
            "actor": self.actor.actor_net.state_dict(),
            "actor_target": self.actor.target_actor_net.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic": self.critic.q_net.state_dict(),
            "critic_target": self.critic.target_q_net.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.mixer is not None:
            data["mixer"] = self.mixer.mix_net.state_dict()
            data["mixer_target"] = self.mixer.target_mix_net.state_dict()
            data["mixer_optimizer"] = self.mixer.optimizer.state_dict()
        torch.save(data, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=torch.device(self.device))
        self.dqn.q_net.load_state_dict(ckpt["dqn_q"])
        self.dqn.target_net.load_state_dict(ckpt["dqn_target"])
        self.dqn.optimizer.load_state_dict(ckpt["dqn_optimizer"])
        self.actor.actor_net.load_state_dict(ckpt["actor"])
        self.actor.target_actor_net.load_state_dict(ckpt["actor_target"])
        self.actor.optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic.q_net.load_state_dict(ckpt["critic"])
        self.critic.target_q_net.load_state_dict(ckpt["critic_target"])
        self.critic.optimizer.load_state_dict(ckpt["critic_optimizer"])
        if self.mixer is not None and "mixer" in ckpt:
            self.mixer.mix_net.load_state_dict(ckpt["mixer"])
            self.mixer.target_mix_net.load_state_dict(ckpt["mixer_target"])
            self.mixer.optimizer.load_state_dict(ckpt["mixer_optimizer"])
        self.global_step = ckpt["global_step"]
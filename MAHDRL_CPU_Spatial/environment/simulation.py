"""SimulationEnvironment — full MAHDRL environment.

Observation (paper Eq. 17): includes the composite channel matrix so that
the DDPG actor can learn MRT-like beamforming from scratch.
    obs = [channel_norms, queues, rates, wifi, last_selection,
           channel_re, channel_im]
    dims = 4K + J + 2·K·N_ant = 320  (with K=12, N_ant=10, J=32)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from config import SimulationConfig, AgentConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel
from environment.beamforming import mrt_beamformer, equal_power_split
from environment.sinr import compute_all_user_sinrs, noise_power_w
from environment.data_rate import per_user_rates_bps
from environment.queue_model import QueueModel
from environment.qoe import compute_qoe, compute_reliability


WIFI_PENALTY_BPS_PER_DB = 1e4
LATENCY_PENALTY_BPS_PER_S = 1e6


class SimulationEnvironment:
    """Full MAHDRL environment (one BS, S RISs, K users, J Wi-Fi devices)."""

    def __init__(
        self,
        cfg: SimulationConfig,
        agent_cfg: AgentConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg or AgentConfig()
        self.seed = cfg.random_seed if seed is None else seed

        self.topology = NetworkTopology(cfg, seed=self.seed)
        self.channel = ChannelModel(cfg, self.topology, seed=self.seed)

        self.K = cfg.users_per_bs
        self.N_ant = cfg.antennas_per_bs
        self.S = cfg.ris_per_bs
        self.L = cfg.ris_elements
        self.num_wifi_dev = cfg.num_wifi_aps * cfg.wifi_devices_per_ap

        self.p_tx_w = 10.0 ** ((cfg.max_bs_power_dbm - 30.0) / 10.0)
        self.noise_w = noise_power_w(cfg)
        self.psi_max_w = 10.0 ** ((cfg.wifi_energy_detection_dbm - 30.0) / 10.0)

        self.queue: QueueModel | None = None
        self.virtual_queues = np.zeros(self.K)
        self.timestep: int = 0
        self.last_rates = np.zeros(self.K)
        self.last_selected = np.zeros(self.K, dtype=bool)

    @property
    def obs_dim(self) -> int:
        """Total observation dimension. Auto-computed from config."""
        return 4 * self.K + self.num_wifi_dev + 2 * self.K * self.N_ant

    # ==================================================================
    def reset(self) -> np.ndarray:
        self.queue = QueueModel(
            self.cfg, num_users=self.K, seed=self.seed + self.timestep
        )
        self.virtual_queues = np.zeros(self.K)
        self.timestep = 0
        self.last_rates = np.zeros(self.K)
        self.last_selected = np.zeros(self.K, dtype=bool)
        obs, _, _, _ = self._apply_and_observe(self._zero_action())
        return obs

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, _, reward, info = self._apply_and_observe(action)
        self.timestep += 1
        done = self.timestep >= self.cfg.time_slots_per_episode
        return obs, reward, done, info

    # ==================================================================
    def make_mrt_action(
        self, num_selected: int | None = None, offset: int = 0
    ) -> Dict[str, np.ndarray]:
        if num_selected is None:
            num_selected = min(self.K, self.N_ant)
        idx = [(offset + i) % self.K for i in range(num_selected)]
        selected = np.zeros(self.K, dtype=bool)
        selected[idx] = True

        ris_phases_dict = {s: np.zeros(self.L) for s in range(self.S)}
        powers = equal_power_split(self.p_tx_w, num_selected)

        beamformers = np.zeros((self.K, self.N_ant), dtype=complex)
        for i, k in enumerate(idx):
            h_k = self.channel.composite_user_channel(0, k, ris_phases_dict)
            beamformers[k] = mrt_beamformer(h_k, powers[i]).reshape(-1)

        return {
            "user_selection": selected,
            "beamformers": beamformers,
            "ris_phases": np.zeros((self.S, self.L)),
        }

    # ==================================================================
    def _zero_action(self) -> Dict[str, np.ndarray]:
        return {
            "user_selection": np.zeros(self.K, dtype=bool),
            "beamformers": np.zeros((self.K, self.N_ant), dtype=complex),
            "ris_phases": np.zeros((self.S, self.L)),
        }

    def _validate_action(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = np.asarray(action["user_selection"], dtype=bool).reshape(-1)
        beamformers = np.asarray(action["beamformers"], dtype=complex)
        ris_phases = np.asarray(action["ris_phases"], dtype=float)

        if selected.shape != (self.K,):
            raise ValueError(f"user_selection must have shape ({self.K},)")
        if beamformers.shape != (self.K, self.N_ant):
            raise ValueError(
                f"beamformers must have shape ({self.K}, {self.N_ant}), "
                f"got {beamformers.shape}"
            )
        if ris_phases.shape != (self.S, self.L):
            raise ValueError(f"ris_phases must have shape ({self.S}, {self.L})")

        if np.sum(selected) > self.N_ant:
            norms = np.linalg.norm(beamformers, axis=1)
            scores = np.where(selected, norms, -np.inf)
            keep = np.argsort(scores)[-self.N_ant:]
            selected = np.zeros(self.K, dtype=bool)
            selected[keep] = True

        total_power = float(np.sum(np.abs(beamformers) ** 2))
        if total_power > self.p_tx_w and total_power > 0:
            beamformers = beamformers * np.sqrt(self.p_tx_w / total_power)

        ris_phases = np.mod(ris_phases, 2.0 * np.pi)

        return selected, beamformers, ris_phases

    def _apply_and_observe(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        selected, beamformers, ris_phases_matrix = self._validate_action(action)

        ris_phases_dict = {s: ris_phases_matrix[s] for s in range(self.S)}
        channels = [
            self.channel.composite_user_channel(0, k, ris_phases_dict)
            for k in range(self.K)
        ]

        idx = np.where(selected)[0]
        if idx.size == 0:
            sinrs: list[float] = []
            rates_selected = np.zeros(0)
        else:
            channels_sel = [channels[k] for k in idx]
            bf_sel = [beamformers[k].reshape(-1, 1) for k in idx]
            sinrs = compute_all_user_sinrs(channels_sel, bf_sel, self.noise_w)
            rates_selected = np.array(
                per_user_rates_bps(sinrs, self.cfg.bandwidth_hz)
            )

        rates_all = np.zeros(self.K)
        for i, k in enumerate(idx):
            rates_all[k] = rates_selected[i]

        latencies = self.queue.total_latency(rates_all, selected)
        qoe = compute_qoe(rates_all, latencies, self.cfg)

        wifi_psi_w = self._compute_wifi_interference(ris_phases_dict, beamformers)

        base_reward = self._compute_base_reward(
            rates_all, latencies, wifi_psi_w, qoe
        )

        Lambda = self.queue.queues + self.virtual_queues
        lyap_bonus = self.agent_cfg.lyapunov_zeta * float(
            np.sum(Lambda * rates_all)
        )
        reward = base_reward + lyap_bonus

        arrivals = self.queue.sample_arrivals()
        self.queue.step(selected, rates_all, arrivals)
        self.virtual_queues = np.maximum(
            self.virtual_queues + self.queue.queues - self.cfg.max_queue_bits,
            0.0,
        )

        self.last_rates = rates_all
        self.last_selected = selected

        obs = self._build_observation(channels, rates_all, wifi_psi_w)

        wifi_dbm = 10.0 * np.log10(wifi_psi_w + 1e-30) + 30.0
        info = {
            "qoe_bps": qoe,
            "base_reward_bps": base_reward,
            "lyapunov_bonus_bps": lyap_bonus,
            "total_rate_bps": float(np.sum(rates_all)),
            "per_user_rates_bps": rates_all.copy(),
            "sinrs": list(sinrs),
            "latencies_s": latencies.copy(),
            "wifi_psi_w": wifi_psi_w.copy(),
            "wifi_psi_dbm": wifi_dbm,
            "queue_bits": self.queue.queues.copy(),
            "virtual_queue_bits": self.virtual_queues.copy(),
            "lambda_bits": Lambda.copy(),
            "selected": selected.copy(),
            "num_selected": int(idx.size),
            "reliability": compute_reliability(self.queue.queues, self.cfg),
            "timestep": self.timestep,
        }
        return obs, float(np.sum(rates_all)), reward, info

    def _compute_wifi_interference(
        self,
        ris_phases_dict: Dict[int, np.ndarray],
        beamformers: np.ndarray,
    ) -> np.ndarray:
        psi = np.zeros(self.num_wifi_dev)
        for j in range(self.num_wifi_dev):
            v_j = self.channel.composite_wifi_channel(0, j, ris_phases_dict)
            for k in range(self.K):
                if np.linalg.norm(beamformers[k]) < 1e-12:
                    continue
                w_k = beamformers[k].reshape(-1, 1)
                gain = abs(np.asarray(v_j @ w_k).item()) ** 2
                psi[j] += gain
        return psi

    def _compute_base_reward(
        self,
        rates_all: np.ndarray,
        latencies: np.ndarray,
        wifi_psi_w: np.ndarray,
        qoe: float,
    ) -> float:
        reward = qoe
        wifi_dbm = 10.0 * np.log10(wifi_psi_w + 1e-30) + 30.0
        excess_db = np.maximum(0.0, wifi_dbm - self.cfg.wifi_energy_detection_dbm)
        reward -= (
            self.agent_cfg.reward_lambda_wifi
            * WIFI_PENALTY_BPS_PER_DB
            * float(np.sum(excess_db))
        )
        rate_excess = np.maximum(0.0, rates_all - self.cfg.min_user_rate_bps)
        reward += self.agent_cfg.reward_lambda_rate * float(np.sum(rate_excess))
        lat_excess = np.maximum(0.0, latencies - self.cfg.max_latency_s)
        reward -= (
            self.agent_cfg.reward_lambda_latency
            * LATENCY_PENALTY_BPS_PER_S
            * float(np.sum(lat_excess))
        )
        return float(reward)

    def _build_observation(
        self,
        channels: list[np.ndarray],
        rates_all: np.ndarray,
        wifi_psi_w: np.ndarray,
    ) -> np.ndarray:
        """Paper Eq. 17: includes full composite channel matrix (re + im)."""
        # 1) Channel norms (K)
        channel_norms = np.array(
            [float(np.linalg.norm(channels[k]) ** 2) for k in range(self.K)]
        )
        channel_norm_feat = channel_norms / 1e-9

        # 2) Queue lengths (K)
        queue_feat = self.queue.queues / self.cfg.max_queue_bits

        # 3) Current rates (K)
        rate_feat = rates_all / 1e7

        # 4) WiFi interference (J)
        wifi_dbm = 10.0 * np.log10(wifi_psi_w + 1e-30) + 30.0
        wifi_feat = wifi_dbm / -60.0

        # 5) Last selection (K)
        select_feat = self.last_selected.astype(float)

        # 6) NEW — composite channel matrix, real and imaginary parts
        channel_re = np.zeros((self.K, self.N_ant), dtype=np.float32)
        channel_im = np.zeros((self.K, self.N_ant), dtype=np.float32)
        scale = self.cfg.channel_obs_scale
        for k in range(self.K):
            h_k = channels[k].reshape(-1)          # (N_ant,)
            channel_re[k] = np.real(h_k) / scale
            channel_im[k] = np.imag(h_k) / scale

        return np.concatenate(
            [
                channel_norm_feat,
                queue_feat,
                rate_feat,
                wifi_feat,
                select_feat,
                channel_re.reshape(-1),
                channel_im.reshape(-1),
            ]
        ).astype(np.float32)
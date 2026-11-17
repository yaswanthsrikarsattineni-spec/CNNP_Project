"""Central configuration for the MAHDRL paper replication."""
from dataclasses import dataclass, asdict
import json


@dataclass
class SimulationConfig:
    # ---------- Paper values (Table I + Section V) ----------
    area_width_m: float = 300.0
    area_height_m: float = 300.0
    carrier_frequency_hz: float = 5e9
    bandwidth_hz: float = 40e6
    rician_k_factor: float = 1.0
    max_bs_power_dbm: float = 30.0
    wifi_energy_detection_dbm: float = -62.0
    max_queue_bits: float = 3e6
    min_user_rate_bps: float = 3e6
    max_latency_s: float = 30e-3
    noise_psd_dbm_per_hz: float = -174.0

    antennas_per_bs: int = 10
    users_per_bs: int = 12
    ris_per_bs: int = 1
    ris_elements: int = 20
    num_wifi_aps: int = 4
    wifi_devices_per_ap: int = 8
    user_arrival_rate_bps: float = 0.2e6

    time_slots_per_episode: int = 100
    slot_duration_s: float = 10e-3
    discount_factor: float = 0.95
    soft_update_tau: float = 0.01
    batch_size: int = 128
    learning_rate: float = 1e-3

    num_episodes: int = 400

    # ---------- Implementation assumptions ----------
    num_bs: int = 1
    path_loss_exponent: float = 2.0
    qoe_latency_penalty: float = 1e6
    random_seed: int = 7
    device: str = "cuda"

    # Observation layout: channel_re and channel_im are stored in obs at
    # positions [4K+J : 4K+J+K·N_ant] and [4K+J+K·N_ant : 4K+J+2K·N_ant],
    # divided by this scale (so obs values are ~O(1)).
    channel_obs_scale: float = 1e-4


@dataclass
class AgentConfig:
    # ---------- Paper Table I values (mirrored) ----------
    discount_factor: float = 0.95
    soft_update_tau: float = 0.01
    batch_size: int = 128
    learning_rate: float = 1e-3

    # ---------- DDPG stabilizers ----------
    ddpg_target_tau: float = 0.0005
    actor_learning_rate: float = 1e-4
    critic_weight_decay: float = 1e-5
    ddpg_warmup_steps: int = 4000
    ddpg_update_every: int = 4
    ddpg_target_clip: float = 20.0

    # ---------- Mixer stabilizers ----------
    mixer_learning_rate: float = 1e-4
    mixer_l2_penalty: float = 0.01

    # ---------- Implementation choices ----------
    hidden_dims: tuple = (256, 256)
    replay_capacity: int = 100_000
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay_steps: int = 20_000
    exploration_noise_std: float = 0.1
    target_update_interval: int = 10

    lyapunov_zeta: float = 1e-6

    reward_lambda_wifi: float = 1.0
    reward_lambda_rate: float = 0.01
    reward_lambda_latency: float = 1.0


@dataclass
class ExperimentConfig:
    user_counts: tuple = (6, 8, 10, 12, 14, 16)
    wifi_ap_counts: tuple = (2, 4, 6, 8, 10)
    ris_element_counts: tuple = (10, 20, 30, 40, 50)
    ris_counts: tuple = (1, 2, 3, 4)
    min_rate_sweep_mbps: tuple = (3, 4, 5)


def config_as_dict():
    return {
        "simulation": asdict(SimulationConfig()),
        "agent": asdict(AgentConfig()),
        "experiments": asdict(ExperimentConfig()),
    }


if __name__ == "__main__":
    print(json.dumps(config_as_dict(), indent=2))
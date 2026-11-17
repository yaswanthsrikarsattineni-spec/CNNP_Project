"""Tests for the MADDPG baseline."""
import numpy as np

from config import SimulationConfig, AgentConfig
from training.maddpg import MADDPGAgent


OBS_DIM = 320
K, N_ANT, N_RIS, L = 12, 10, 1, 20
WIFI_DEV = 32
SCALE = 1e-4


def _make():
    sim_cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    return MADDPGAgent(
        simulation_cfg=sim_cfg, agent_cfg=agent_cfg,
        obs_dim=OBS_DIM, device="cpu", seed=0,
    )


def _obs(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=OBS_DIM).astype(np.float32)
    c = (rng.normal(size=(K, N_ANT)) + 1j * rng.normal(size=(K, N_ANT))) * 1e-4
    start = 4 * K + WIFI_DEV
    obs[start : start + K * N_ANT] = np.real(c).reshape(-1) / SCALE
    obs[start + K * N_ANT : start + 2 * K * N_ANT] = np.imag(c).reshape(-1) / SCALE
    return obs


def test_action_shapes():
    agent = _make()
    raw, action = agent.select_action(_obs())
    assert raw.shape == (44,)
    assert action["user_selection"].shape == (K,)
    assert action["beamformers"].shape == (K, N_ANT)
    assert action["ris_phases"].shape == (N_RIS, L)


def test_top_n_selection():
    agent = _make()
    _, action = agent.select_action(_obs())
    assert int(np.sum(action["user_selection"])) == 10


def test_power_equals_max():
    agent = _make()
    _, action = agent.select_action(_obs())
    power = float(np.sum(np.abs(action["beamformers"]) ** 2))
    assert np.isclose(power, 1.0, atol=1e-9)


def test_noise_adds_variation():
    agent = _make()
    obs = _obs()
    raw1, _ = agent.select_action(obs, noise_std=0.2)
    raw2, _ = agent.select_action(obs, noise_std=0.2)
    assert not np.allclose(raw1, raw2)


def test_train_step_returns_losses():
    agent = _make()
    agent_cfg = AgentConfig()
    rng = np.random.default_rng(0)
    for _ in range(agent_cfg.batch_size + 10):
        obs = _obs(seed=int(rng.integers(0, 10_000)))
        raw, _ = agent.select_action(obs)
        agent.store(obs, raw, float(rng.normal() * 1e7), obs, 0.0)
    info = agent.train_step()
    assert info is not None
    for k in ("critic_loss", "actor_loss", "q_mean"):
        assert k in info
        assert np.isfinite(info[k])


def test_save_load_roundtrip(tmp_path):
    agent = _make()
    obs = _obs()
    raw1, _ = agent.select_action(obs)
    path = tmp_path / "maddpg.pt"
    agent.save(str(path))

    agent2 = _make()
    agent2.load(str(path))
    raw2, _ = agent2.select_action(obs)
    assert np.allclose(raw1, raw2, atol=1e-5)
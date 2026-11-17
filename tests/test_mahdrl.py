"""Tests for the full MAHDRL agent (paper Algorithm 1).

⚠ obs_dim = 320 matches the environment's expanded observation (see
environment/simulation.py::obs_dim) — a reflection of paper Eq. 17 which
includes channel matrices in the state.
"""
import numpy as np
import pytest

from config import SimulationConfig, AgentConfig
from agents.mahdrl import MAHDRLAgent


OBS_DIM = 320   # matches SimulationEnvironment.obs_dim for K=12, N_ant=10, J=32


def _make_agent(obs_dim=OBS_DIM):
    sim_cfg = SimulationConfig()
    agent_cfg = AgentConfig()
    return sim_cfg, agent_cfg, MAHDRLAgent(
        simulation_cfg=sim_cfg,
        agent_cfg=agent_cfg,
        obs_dim=obs_dim,
        device="cpu",
        seed=0,
    )


def _make_obs(obs_dim=OBS_DIM, seed=0):
    return np.random.default_rng(seed).normal(size=obs_dim).astype(np.float32)


# ---------------------------------------------------------------- construction
def test_agent_constructs():
    sim_cfg, _, agent = _make_agent()
    assert agent.num_users == sim_cfg.users_per_bs
    assert agent.num_agents == 1
    assert agent.buffer.size == 0


# ---------------------------------------------------------------- action selection
def test_select_action_returns_three_objects():
    _, _, agent = _make_agent()
    obs = _make_obs()
    disc, raw, env_action = agent.select_action(obs)
    assert disc.shape == (12,)
    assert int(disc.sum()) == 10
    assert raw.shape == (agent.actor.output_dim,)
    assert env_action["beamformers"].shape == (12, 10)
    assert env_action["ris_phases"].shape == (1, 20)


def test_epsilon_changes_discrete_action():
    _, _, agent = _make_agent()
    obs = _make_obs()
    greedies = [tuple(np.where(agent.select_action(obs)[0])[0]) for _ in range(5)]
    assert len(set(greedies)) == 1
    randoms = [tuple(np.where(agent.select_action(obs, epsilon=1.0)[0])[0])
               for _ in range(50)]
    assert len(set(randoms)) > 1


def test_noise_changes_continuous_action():
    _, _, agent = _make_agent()
    obs = _make_obs()
    a1 = agent.select_action(obs, noise_std=0.0)[2]["beamformers"]
    a2 = agent.select_action(obs, noise_std=0.2)[2]["beamformers"]
    assert not np.allclose(a1, a2)


# ---------------------------------------------------------------- store
def test_store_adds_to_buffer():
    _, _, agent = _make_agent()
    obs = _make_obs()
    disc, raw, _ = agent.select_action(obs)
    agent.store(obs, disc, raw, reward=1.0, next_obs=obs, done=0.0)
    assert agent.buffer.size == 1
    expected_action_dim = agent.num_users + agent.actor.output_dim
    assert agent.buffer.action_buf.shape[1] == expected_action_dim


# ---------------------------------------------------------------- training
def test_train_step_returns_none_until_ready():
    _, _, agent = _make_agent()
    obs = _make_obs()
    disc, raw, _ = agent.select_action(obs)
    agent.store(obs, disc, raw, 1.0, obs, 0.0)
    assert agent.train_step() is None


def test_train_step_returns_losses_when_ready():
    _, agent_cfg, agent = _make_agent()
    rng = np.random.default_rng(0)
    for _ in range(agent_cfg.batch_size + 10):
        obs = _make_obs(seed=int(rng.integers(0, 10_000)))
        disc, raw, _ = agent.select_action(obs)
        reward = float(rng.normal() * 1e7)
        agent.store(obs, disc, raw, reward, obs, 0.0)
    info = agent.train_step()
    assert info is not None
    for key in ("dqn_loss", "critic_mixer_loss", "q_tot_mean", "actor_loss"):
        assert key in info
        assert np.isfinite(info[key])


def test_multiple_train_steps_run_without_error():
    _, agent_cfg, agent = _make_agent()
    rng = np.random.default_rng(0)
    for _ in range(agent_cfg.batch_size + 10):
        obs = _make_obs(seed=int(rng.integers(0, 10_000)))
        disc, raw, _ = agent.select_action(obs)
        reward = float(rng.normal() * 1e7)
        agent.store(obs, disc, raw, reward, obs, 0.0)
    for _ in range(20):
        info = agent.train_step()
        assert info is not None
        assert np.isfinite(info["critic_mixer_loss"])


# ---------------------------------------------------------------- persistence
def test_save_load_roundtrip(tmp_path):
    _, _, agent = _make_agent()
    obs = _make_obs()
    disc1, raw1, _ = agent.select_action(obs)

    path = tmp_path / "mahdrl.pt"
    agent.save(str(path))

    _, _, agent2 = _make_agent()
    agent2.load(str(path))
    disc2, raw2, _ = agent2.select_action(obs)

    assert np.array_equal(disc1, disc2)
    assert np.allclose(raw1, raw2, atol=1e-5)


# ---------------------------------------------------------------- distributed
def test_global_step_increments():
    _, agent_cfg, agent = _make_agent()
    rng = np.random.default_rng(0)
    for _ in range(agent_cfg.batch_size + 10):
        obs = _make_obs(seed=int(rng.integers(0, 10_000)))
        disc, raw, _ = agent.select_action(obs)
        agent.store(obs, disc, raw, float(rng.normal() * 1e7), obs, 0.0)
    before = agent.global_step
    agent.train_step()
    assert agent.global_step == before + 1
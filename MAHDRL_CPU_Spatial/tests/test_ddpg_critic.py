"""Tests for DDPG critic and joint actor-critic training."""
import numpy as np

from config import AgentConfig
from agents.ddpg_actor import DDPGActor
from agents.ddpg_critic import DDPGCritic


OBS_DIM = 320
K, N_ANT, N_RIS, L = 12, 10, 1, 20
WIFI_DEV = 32
SCALE = 1e-4


def _make_agents():
    cfg = AgentConfig()
    actor = DDPGActor(
        obs_dim=OBS_DIM, num_users=K, num_ant=N_ANT, num_ris=N_RIS,
        ris_elements=L, max_power_w=1.0, wifi_device_count=WIFI_DEV,
        channel_obs_scale=SCALE, agent_cfg=cfg, device="cpu", seed=0,
    )
    critic = DDPGCritic(
        obs_dim=OBS_DIM, num_users=K, continuous_dim=actor.output_dim,
        agent_cfg=cfg, device="cpu",
    )
    return actor, critic, cfg


def _make_batch(batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32)
    disc = np.zeros((batch_size, K), dtype=np.float32)
    for b in range(batch_size):
        disc[b, rng.choice(K, size=10, replace=False)] = 1.0
    cont_dim = K + N_RIS * L      # = actor.output_dim = 32
    cont = rng.uniform(-1.0, 1.0, size=(batch_size, cont_dim)).astype(np.float32)
    action = np.concatenate([disc, cont], axis=1)
    reward = rng.normal(size=batch_size).astype(np.float32) * 1e7
    next_obs = rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32)
    done = np.zeros(batch_size, dtype=np.float32)
    return (obs, action, reward, next_obs, done)


def test_critic_input_dim():
    actor, critic, _ = _make_agents()
    assert critic.input_dim == OBS_DIM + K + actor.output_dim


def test_critic_compute_q_shape():
    actor, critic, _ = _make_agents()
    batch = _make_batch(8)
    obs, action, _, _, _ = batch
    disc = action[:, :K]
    cont = action[:, K:]
    q = critic.compute_q(obs, disc, cont)
    assert q.shape == (8,)
    assert np.all(np.isfinite(q))


def test_critic_loss_decreases_on_fixed_batch():
    actor, critic, _ = _make_agents()
    batch = _make_batch(32)
    losses = [critic.update(actor, batch)["critic_loss"] for _ in range(200)]
    first_avg = np.mean(losses[:20])
    last_avg = np.mean(losses[-20:])
    assert last_avg < first_avg


def test_critic_loss_reasonable_scale():
    actor, critic, _ = _make_agents()
    batch = _make_batch(32)
    loss = critic.update(actor, batch)["critic_loss"]
    assert 1e-3 < loss < 1e6


def test_actor_update_returns_loss():
    actor, critic, _ = _make_agents()
    batch = _make_batch(32)
    info = actor.update(critic, batch)
    assert "actor_loss" in info
    assert np.isfinite(info["actor_loss"])


def test_actor_update_increases_q_on_frozen_critic():
    actor, critic, _ = _make_agents()
    batch = _make_batch(32)
    obs, action, _, _, _ = batch
    disc = action[:, :K]

    def eval_q():
        raw = actor._raw_output(obs, disc, use_target=False)
        q = critic.compute_q(obs, disc, raw)
        return float(np.mean(q))

    q_before = eval_q()
    for _ in range(50):
        actor.update(critic, batch)
    q_after = eval_q()
    assert q_after > q_before


def test_joint_training_runs_without_error():
    actor, critic, _ = _make_agents()
    batch = _make_batch(32)
    for _ in range(20):
        critic_info = critic.update(actor, batch)
        actor_info = actor.update(critic, batch)
    assert np.isfinite(critic_info["critic_loss"])
    assert np.isfinite(actor_info["actor_loss"])


def test_critic_save_load(tmp_path):
    actor, critic, _ = _make_agents()
    batch = _make_batch(8)
    obs, action, _, _, _ = batch
    q1 = critic.compute_q(obs, action[:, :K], action[:, K:])

    path = tmp_path / "critic.pt"
    critic.save(str(path))
    actor2, critic2, _ = _make_agents()
    critic2.load(str(path))
    q2 = critic2.compute_q(obs, action[:, :K], action[:, K:])
    assert np.allclose(q1, q2)
"""Tests for the DDPG actor with power-weight + MRT parameterization."""
import numpy as np

from config import AgentConfig
from agents.ddpg_actor import DDPGActor
from agents.ddpg_critic import DDPGCritic


OBS_DIM = 320
K, N_ANT, N_RIS, L = 12, 10, 1, 20
WIFI_DEV = 32
SCALE = 1e-4


def _make_actor():
    cfg = AgentConfig()
    return DDPGActor(
        obs_dim=OBS_DIM, num_users=K, num_ant=N_ANT, num_ris=N_RIS,
        ris_elements=L, max_power_w=1.0, wifi_device_count=WIFI_DEV,
        channel_obs_scale=SCALE, agent_cfg=cfg, device="cpu", seed=0,
    )


def _sample_obs_selection(num_selected=10, seed=0):
    """Build a realistic obs with a valid channel block."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=OBS_DIM).astype(np.float32)
    # Inject random channel at obs[80:320] scaled so obs ≈ O(1)
    channel = (rng.normal(size=(K, N_ANT)) + 1j * rng.normal(size=(K, N_ANT))) * 1e-4
    start = 4 * K + WIFI_DEV                                  # 80
    obs[start : start + K * N_ANT] = np.real(channel).reshape(-1) / SCALE
    obs[start + K * N_ANT : start + 2 * K * N_ANT] = np.imag(channel).reshape(-1) / SCALE
    disc = np.zeros(K, dtype=np.float32)
    disc[:num_selected] = 1.0
    return obs, disc


def test_action_shapes():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    action = actor.select_action(obs, disc)
    assert action["user_selection"].shape == (K,)
    assert action["beamformers"].shape == (K, N_ANT)
    assert action["beamformers"].dtype == np.complex128
    assert action["ris_phases"].shape == (N_RIS, L)


def test_input_output_dims():
    actor = _make_actor()
    assert actor.input_dim == OBS_DIM + K          # 332
    assert actor.power_dim == K                    # 12
    assert actor.phase_dim == N_RIS * L            # 20
    assert actor.output_dim == K + N_RIS * L       # 32


def test_power_equals_max_over_selected():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    action = actor.select_action(obs, disc)
    total_power = float(np.sum(np.abs(action["beamformers"]) ** 2))
    assert np.isclose(total_power, 1.0, atol=1e-9), f"power={total_power}"


def test_ris_phases_in_range():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    action = actor.select_action(obs, disc)
    assert np.all(action["ris_phases"] >= 0.0)
    assert np.all(action["ris_phases"] <= 2.0 * np.pi + 1e-9)


def test_non_selected_users_have_zero_beamformers():
    actor = _make_actor()
    obs, disc = _sample_obs_selection(num_selected=5)
    action = actor.select_action(obs, disc)
    for k in range(5, K):
        assert np.all(action["beamformers"][k] == 0.0 + 0.0j)


def test_beamformer_direction_is_mrt():
    """For each selected user, w_k must be parallel to h_k^*."""
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    action = actor.select_action(obs, disc)
    channels = actor._extract_channel(obs)                        # (K, N)
    for k in range(K):
        if not disc[k]:
            continue
        w = action["beamformers"][k]
        h = channels[k]
        if np.linalg.norm(w) < 1e-12 or np.linalg.norm(h) < 1e-12:
            continue
        # w should be parallel to h^*  ⇔  w × h has zero imaginary ratio
        # Check via normalized inner product ≈ 1 (real, positive)
        cross = np.vdot(h.conj(), w)                               # <h^*, w>
        nrm = np.linalg.norm(h) * np.linalg.norm(w)
        assert np.isclose(np.abs(cross / nrm), 1.0, atol=1e-6), (
            f"user {k} beamformer not aligned with MRT"
        )


def test_deterministic_without_noise():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    a1 = actor.select_action(obs, disc, noise_std=0.0)
    a2 = actor.select_action(obs, disc, noise_std=0.0)
    assert np.allclose(a1["beamformers"], a2["beamformers"])
    assert np.allclose(a1["ris_phases"], a2["ris_phases"])


def test_noise_adds_variation():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    a1 = actor.select_action(obs, disc, noise_std=0.2)
    a2 = actor.select_action(obs, disc, noise_std=0.2)
    assert not np.allclose(a1["beamformers"], a2["beamformers"])


def test_target_action_is_deterministic():
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    a1 = actor.select_target_action(obs, disc)
    a2 = actor.select_target_action(obs, disc)
    assert np.allclose(a1["beamformers"], a2["beamformers"])


def test_soft_update_moves_target():
    import torch
    actor = _make_actor()
    before = [p.clone() for p in actor.target_actor_net.parameters()]
    with torch.no_grad():
        for p in actor.actor_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    actor.soft_update_target()
    after = [p.clone() for p in actor.target_actor_net.parameters()]
    moved = sum((a - b).abs().sum().item() for a, b in zip(after, before))
    assert moved > 0.0


def test_save_load_roundtrip(tmp_path):
    actor = _make_actor()
    obs, disc = _sample_obs_selection()
    a1 = actor.select_action(obs, disc)
    path = tmp_path / "actor.pt"
    actor.save(str(path))
    actor2 = _make_actor()
    actor2.load(str(path))
    a2 = actor2.select_action(obs, disc)
    assert np.allclose(a1["beamformers"], a2["beamformers"])


def test_actor_update_works_with_critic():
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
    rng = np.random.default_rng(0)
    B = 16
    obs = rng.normal(size=(B, OBS_DIM)).astype(np.float32)
    disc = np.zeros((B, K), dtype=np.float32); disc[:, :10] = 1.0
    cont = rng.uniform(-1, 1, size=(B, actor.output_dim)).astype(np.float32)
    action = np.concatenate([disc, cont], axis=1)
    reward = rng.normal(size=B).astype(np.float32) * 1e7
    next_obs = rng.normal(size=(B, OBS_DIM)).astype(np.float32)
    done = np.zeros(B, dtype=np.float32)
    batch = (obs, action, reward, next_obs, done)

    info = actor.update(critic, batch)
    assert "actor_loss" in info
    assert np.isfinite(info["actor_loss"])
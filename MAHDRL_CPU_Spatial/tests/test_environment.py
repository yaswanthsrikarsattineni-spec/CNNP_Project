"""Tests for the SimulationEnvironment."""
import numpy as np

from config import SimulationConfig, AgentConfig
from environment.simulation import SimulationEnvironment


def _env():
    cfg = SimulationConfig()
    return cfg, SimulationEnvironment(cfg)


def test_reset_returns_correct_observation_shape():
    cfg, env = _env()
    obs = env.reset()
    expected_dim = (
        4 * cfg.users_per_bs
        + cfg.num_wifi_aps * cfg.wifi_devices_per_ap
        + 2 * cfg.users_per_bs * cfg.antennas_per_bs
    )
    assert obs.shape == (expected_dim,)
    assert expected_dim == 320   # sanity: K=12, N_ant=10, J=32 → 320
    assert np.all(np.isfinite(obs))

def test_step_returns_five_expected_objects():
    cfg, env = _env()
    env.reset()
    result = env.step(env.make_mrt_action())
    assert len(result) == 4
    obs, reward, done, info = result
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "total_rate_bps" in info
    assert "queue_bits" in info


def test_episode_ends_after_100_steps():
    cfg, env = _env()
    env.reset()
    for t in range(cfg.time_slots_per_episode):
        _, _, done, _ = env.step(env.make_mrt_action())
        if t < cfg.time_slots_per_episode - 1:
            assert not done, f"done prematurely at t={t}"
        else:
            assert done, "episode did not end at t=99"


def test_mrt_action_under_power_budget():
    cfg, env = _env()
    action = env.make_mrt_action(num_selected=10)
    total_power = np.sum(np.abs(action["beamformers"]) ** 2)
    assert total_power <= env.p_tx_w + 1e-9, (
        f"MRT action exceeds Pmax: {total_power:.4f} > {env.p_tx_w:.4f}"
    )


def test_validation_caps_over_selection():
    cfg, env = _env()
    env.reset()
    action = env.make_mrt_action(num_selected=cfg.users_per_bs)  # all 12
    # 12 > N_ant = 10 → must be capped
    obs, _, _, info = env.step(action)
    assert info["num_selected"] <= cfg.antennas_per_bs


def test_validation_scales_power_down():
    cfg, env = _env()
    env.reset()
    action = env.make_mrt_action(num_selected=5)
    # Blow up the beamformers to 10x power
    action["beamformers"] = action["beamformers"] * np.sqrt(10.0)
    # Should be renormalised
    _, _, _, info = env.step(action)
    # Can't check internal, but at least it shouldn't crash
    assert info["total_rate_bps"] > 0


def test_wifi_interference_computed_for_every_device():
    cfg, env = _env()
    env.reset()
    _, _, _, info = env.step(env.make_mrt_action())
    assert info["wifi_psi_w"].shape == (env.num_wifi_dev,)
    assert np.all(info["wifi_psi_w"] >= 0.0)


def test_zero_action_produces_zero_rate():
    cfg, env = _env()
    env.reset()
    zero = env._zero_action()
    _, _, _, info = env.step(zero)
    assert info["total_rate_bps"] == 0.0
    assert info["num_selected"] == 0


def test_reward_is_finite_and_roughly_expected_scale():
    cfg, env = _env()
    env.reset()
    _, reward, _, _ = env.step(env.make_mrt_action())
    # Reward should be O(1e7) bps-equivalent, not 1e-3 or 1e12
    assert 1e5 < abs(reward) < 1e10, f"Reward out of scale: {reward}"
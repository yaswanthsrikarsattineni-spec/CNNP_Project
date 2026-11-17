"""Tests for the QMIX mixing network (paper Eq. 34, 35, 41)."""
import numpy as np
import pytest
import torch

from agents.mixing_network import MixingNetwork, QMixer


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("num_agents", [1, 2, 3, 4])
def test_forward_shape(num_agents):
    net = MixingNetwork(num_agents, state_dim=80)
    B = 8
    qs = torch.rand(B, num_agents)
    state = torch.rand(B, 80)
    out = net(qs, state)
    assert out.shape == (B,)
    assert torch.all(torch.isfinite(out))


def test_single_agent_degenerates():
    net = MixingNetwork(num_agents=1, state_dim=80)
    B = 4
    qs = torch.rand(B, 1)
    state = torch.rand(B, 80)
    out = net(qs, state)
    assert out.shape == (B,)


# ---------------------------------------------------------------- monotonicity
def _check_monotonicity(net: MixingNetwork, B: int = 4, delta: float = 0.1):
    """Numerically verify ∂Q_tot/∂Q_m ≥ 0 for every agent m."""
    M = net.num_agents
    torch.manual_seed(0)
    qs0 = torch.rand(B, M)
    state = torch.rand(B, net.state_dim)

    with torch.no_grad():
        q_tot0 = net(qs0, state).numpy()

    for m in range(M):
        qs_perturbed = qs0.clone()
        qs_perturbed[:, m] += delta
        with torch.no_grad():
            q_tot1 = net(qs_perturbed, state).numpy()
        diff = q_tot1 - q_tot0
        assert np.all(diff >= -1e-5), (
            f"Monotonicity violated for agent {m}: "
            f"Q_tot decreased by {np.min(diff):.2e} when Q_{m} increased"
        )


def test_monotonicity_m1():
    _check_monotonicity(MixingNetwork(1, 80))


def test_monotonicity_m2():
    _check_monotonicity(MixingNetwork(2, 80))


def test_monotonicity_m3():
    _check_monotonicity(MixingNetwork(3, 80))


def test_monotonicity_m4():
    _check_monotonicity(MixingNetwork(4, 80))


def test_monotonicity_across_random_state():
    """Monotonicity must hold for ANY global state (paper Eq. 35 is universal)."""
    net = MixingNetwork(3, 80)
    for seed in range(5):
        torch.manual_seed(seed)
        _check_monotonicity(net)


# ---------------------------------------------------------------- QMixer wrapper
def test_qmixer_compute_q_tot_numpy_single():
    mixer = QMixer(num_agents=2, state_dim=80, device="cpu")
    qs = np.array([1.0, 2.0], dtype=np.float32)
    state = np.zeros(80, dtype=np.float32)
    q_tot = mixer.compute_q_tot(qs, state)
    assert q_tot.shape == ()   # scalar for single input
    assert np.isfinite(q_tot)


def test_qmixer_compute_q_tot_numpy_batch():
    mixer = QMixer(num_agents=3, state_dim=80, device="cpu")
    qs = np.random.rand(10, 3).astype(np.float32)
    state = np.random.rand(10, 80).astype(np.float32)
    q_tot = mixer.compute_q_tot(qs, state)
    assert q_tot.shape == (10,)


# ---------------------------------------------------------------- training
def test_mixer_learns_to_fit_target():
    """Toy task: Q_tot should learn to fit 2*Q_1 + 3*Q_2 + f(s) (a monotone fn)."""
    mixer = QMixer(num_agents=2, state_dim=4, learning_rate=1e-2, tau=0.01, device="cpu")
    rng = np.random.default_rng(0)

    B = 64
    losses = []
    for _ in range(500):
        aq = rng.normal(size=(B, 2)).astype(np.float32)
        st = rng.normal(size=(B, 4)).astype(np.float32)
        target = 2.0 * aq[:, 0] + 3.0 * aq[:, 1] + 0.5 * st.mean(axis=1)
        reward = target * 1e6   # unscale; mixer.update applies ×1e-6
        info = mixer.update(
            aq, st, reward, aq, st, np.zeros(B),
            discount=0.0, reward_scale=1e-6,
        )
        losses.append(info["mixer_loss"])

    first = np.mean(losses[:50])
    last = np.mean(losses[-50:])
    assert last < first * 0.1, (
        f"Mixer did not learn: first={first:.4f}, last={last:.4f}"
    )


# ---------------------------------------------------------------- save/load
def test_save_load_roundtrip(tmp_path):
    mixer = QMixer(num_agents=3, state_dim=80, device="cpu")
    qs = np.random.rand(5, 3).astype(np.float32)
    state = np.random.rand(5, 80).astype(np.float32)
    q1 = mixer.compute_q_tot(qs, state)

    path = tmp_path / "mixer.pt"
    mixer.save(str(path))

    mixer2 = QMixer(num_agents=3, state_dim=80, device="cpu")
    mixer2.load(str(path))
    q2 = mixer2.compute_q_tot(qs, state)
    assert np.allclose(q1, q2)


# ---------------------------------------------------------------- targets
def test_soft_update_moves_target():
    mixer = QMixer(num_agents=2, state_dim=80, device="cpu", tau=0.1)
    before = [p.clone() for p in mixer.target_mix_net.parameters()]
    # Perturb current weights.
    with torch.no_grad():
        for p in mixer.mix_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    mixer.soft_update_target()
    after = [p.clone() for p in mixer.target_mix_net.parameters()]
    moved = sum((a - b).abs().sum().item() for a, b in zip(after, before))
    assert moved > 0.0
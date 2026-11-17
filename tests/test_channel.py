"""Tests for the channel model.

Verifies that every channel has the correct shape, that the path-loss model
returns a sensible magnitude, that the RIS phase matrix is a unit-modulus
diagonal, and that the composite channel really includes the RIS contribution.
"""
import numpy as np

from config import SimulationConfig
from environment.network import NetworkTopology
from environment.channel_model import ChannelModel


def _make():
    cfg = SimulationConfig()
    topo = NetworkTopology(cfg)
    return cfg, topo, ChannelModel(cfg, topo)


# ---------------------------------------------------------------- shapes
def test_direct_channel_shape():
    cfg, _, ch = _make()
    h = ch.direct_channel(0, 0)
    assert h.shape == (1, cfg.antennas_per_bs)
    assert np.iscomplexobj(h)


def test_bs_ris_channel_shape():
    cfg, _, ch = _make()
    F = ch.bs_ris_channel(0, 0)
    assert F.shape == (cfg.ris_elements, cfg.antennas_per_bs)


def test_ris_user_channel_shape():
    cfg, _, ch = _make()
    g = ch.ris_user_channel(0, 0)
    assert g.shape == (1, cfg.ris_elements)


def test_ris_phase_matrix_is_unit_diagonal():
    cfg, _, ch = _make()
    phases = np.linspace(0, 2 * np.pi, cfg.ris_elements, endpoint=False)
    Theta = ch.ris_phase_matrix(0, phases)
    assert Theta.shape == (cfg.ris_elements, cfg.ris_elements)
    diag = np.diag(Theta)
    assert np.allclose(np.abs(diag), 1.0), "RIS elements must be unit-modulus"
    # Off-diagonal must be zero
    off = Theta - np.diag(diag)
    assert np.allclose(off, 0.0)


# ---------------------------------------------------------------- magnitudes
def test_path_loss_is_reasonable_at_100m():
    cfg, _, ch = _make()
    pl = ch._path_loss(100.0)
    # FSPL at 100 m, 5 GHz, n=2: ~2.28e-9 (about -86 dB)
    assert 1e-10 < pl < 1e-8, f"Unexpected path loss: {pl}"


def test_direct_channel_power_scale():
    cfg, _, ch = _make()
    # Average |h_ij|^2 over many realizations should be close to path loss L.
    N = 500
    energy = 0.0
    bs = ch.topology.base_stations[0]
    user = ch.topology.users[0]
    L_expected = ch._path_loss(bs.position.distance_to(user.position))
    for _ in range(N):
        h = ch.direct_channel(0, 0)
        energy += float(np.mean(np.abs(h) ** 2))
    avg = energy / N
    assert 0.3 * L_expected < avg < 3.0 * L_expected, (
        f"avg |h|^2 = {avg:.3e}, expected ≈ {L_expected:.3e}"
    )


# ---------------------------------------------------------------- RIS
def test_composite_channel_includes_ris():
    """Composite must differ from the direct-only channel when RIS is active."""
    cfg, _, ch = _make()
    phases = {0: np.zeros(cfg.ris_elements)}
    h_direct = ch.direct_channel(0, 0)
    h_comp = ch.composite_user_channel(0, 0, phases)
    assert h_comp.shape == h_direct.shape
    # Even a tiny RIS contribution perturbs the values; np.allclose would fail.
    assert not np.allclose(h_direct, h_comp), (
        "RIS contribution missing from composite channel"
    )


def test_ris_phases_change_composite_channel():
    """Two different phase settings must produce two different composites."""
    cfg, _, ch = _make()
    phases_a = {0: np.zeros(cfg.ris_elements)}
    phases_b = {0: np.full(cfg.ris_elements, np.pi)}
    h_a = ch.composite_user_channel(0, 0, phases_a)
    h_b = ch.composite_user_channel(0, 0, phases_b)
    assert not np.allclose(h_a, h_b), (
        "RIS phases do not affect composite channel"
    )


# ---------------------------------------------------------------- Wi-Fi
def test_composite_wifi_channel_shape():
    cfg, _, ch = _make()
    v = ch.composite_wifi_channel(0, 0, {0: np.zeros(cfg.ris_elements)})
    assert v.shape == (1, cfg.antennas_per_bs)
"""Tests for queue dynamics and QoE."""
import numpy as np

from config import SimulationConfig
from environment.queue_model import QueueModel
from environment.qoe import compute_qoe, compute_reliability


def test_queue_starts_empty():
    cfg = SimulationConfig()
    qm = QueueModel(cfg, num_users=5)
    assert np.all(qm.queues == 0.0)
    assert np.all(qm.queuing_latency == 0.0)


def test_queue_update_depletes_then_arrives():
    cfg = SimulationConfig()
    qm = QueueModel(cfg, num_users=1)
    # Seed queue with 10,000 bits
    qm.queues[0] = 10_000.0
    # Serve 5,000 bits/slot equivalent → departure 5,000 bits
    selected = np.array([True])
    rates_bps = np.array([5000.0 / cfg.slot_duration_s])  # 500 kbps → 5000 bits/slot
    arrivals = np.array([0.0])
    qm.step(selected, rates_bps, arrivals)
    assert np.isclose(qm.queues[0], 5_000.0)


def test_queue_overflow_saturates_at_zero():
    cfg = SimulationConfig()
    qm = QueueModel(cfg, num_users=1)
    qm.queues[0] = 100.0
    # Departure bigger than queue → must clip to 0
    selected = np.array([True])
    rates_bps = np.array([100_000.0])
    arrivals = np.array([0.0])
    qm.step(selected, rates_bps, arrivals)
    assert qm.queues[0] == 0.0


def test_queuing_latency_resets_when_selected():
    cfg = SimulationConfig()
    qm = QueueModel(cfg, num_users=1)
    # 3 slots of not selected → latency = 3*ΔT
    for _ in range(3):
        qm.step(np.array([False]), np.array([0.0]), np.array([0.0]))
    assert np.isclose(qm.queuing_latency[0], 3.0 * cfg.slot_duration_s)
    # Now select → reset
    qm.step(np.array([True]), np.array([1e6]), np.array([0.0]))
    assert qm.queuing_latency[0] == 0.0


def test_qoe_penalises_latency():
    cfg = SimulationConfig()
    rates = np.array([50e6])           # 50 Mbps
    lat = np.array([0.030])            # 30 ms
    qoe = compute_qoe(rates, lat, cfg)
    # λ_d = 1e6 → penalty = 30,000 bps → QoE = 49.97 Mbps
    expected = 50e6 - cfg.qoe_latency_penalty * 0.030
    assert np.isclose(qoe, expected)


def test_reliability_formula():
    cfg = SimulationConfig()
    queues = np.array([1e6, 2e6, 5e6, 1e5])   # 3 healthy, 1 overflowed
    assert np.isclose(compute_reliability(queues, cfg), 0.75)


def test_poisson_mean_close_to_expected():
    cfg = SimulationConfig()
    qm = QueueModel(cfg, num_users=10_000, seed=42)
    samples = qm.sample_arrivals()
    # Mean arrival mean = 0.2e6 * 0.01 = 2000 bits/slot
    assert 1900 < samples.mean() < 2100
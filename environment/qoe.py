"""QoE and reliability metrics — paper Eq. (13).

    QoE(t) = Σ_m Σ_{k_m} (R_{k_m}(t) − λ_d · d_{k_m}(t))

⚠ IMPLEMENTATION ASSUMPTION: numeric value of λ_d. See `AgentConfig.qoe_latency_penalty`.
  Paper states λ_d is a latency-penalty factor but provides no number.
"""
from __future__ import annotations

import numpy as np

from config import SimulationConfig


def compute_qoe(
    rates_bps: np.ndarray,
    latencies_s: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    """Sum of (rate − λ_d·latency) across all users, in bps."""
    return float(
        np.sum(rates_bps - cfg.qoe_latency_penalty * latencies_s)
    )


def compute_reliability(
    queues_bits_per_slot: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    """Fraction of users with a queue below the maximum (0..1)."""
    if queues_bits_per_slot.size == 0:
        return 1.0
    healthy = queues_bits_per_slot < cfg.max_queue_bits
    return float(np.mean(healthy))
"""Data-rate calculation per paper Eq. (8).

    R_m(t) = Σ_k x_k · log₂(1 + γ_k)

⚠ IMPLEMENTATION ASSUMPTION: the paper's Eq. (8) is written in bps/Hz
(the log₂ term). To turn it into bps we multiply by the bandwidth allocated to
each user. We assume bandwidth is shared equally among selected users, i.e.
each user gets B/K_selected Hz. This is standard for multi-user scheduling.
"""
from __future__ import annotations

import numpy as np


def user_rate_bps(sinr: float, bandwidth_hz: float) -> float:
    """Shannon rate for one user with the full `bandwidth_hz` allocated."""
    return float(bandwidth_hz * np.log2(1.0 + sinr))


def total_rate_bps(sinrs: list[float], bandwidth_hz: float) -> float:
    """Total BS rate with bandwidth shared among selected users."""
    K = len(sinrs)
    if K == 0:
        return 0.0
    bw_per_user = bandwidth_hz / K
    return sum(user_rate_bps(s, bw_per_user) for s in sinrs)


def per_user_rates_bps(sinrs: list[float], bandwidth_hz: float) -> list[float]:
    """Per-user rate list (same bandwidth-sharing assumption)."""
    K = len(sinrs)
    if K == 0:
        return []
    bw_per_user = bandwidth_hz / K
    return [user_rate_bps(s, bw_per_user) for s in sinrs]
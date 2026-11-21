"""Simultaneous spatial streams occupy the full channel in the SINR model."""
from __future__ import annotations
import numpy as np


def user_rate_bps(sinr: float, bandwidth_hz: float) -> float:
    return float(bandwidth_hz * np.log2(1.0 + sinr))


def per_user_rates_bps(sinrs: list[float], bandwidth_hz: float) -> list[float]:
    return [user_rate_bps(sinr, bandwidth_hz) for sinr in sinrs]


def total_rate_bps(sinrs: list[float], bandwidth_hz: float) -> float:
    return sum(per_user_rates_bps(sinrs, bandwidth_hz))

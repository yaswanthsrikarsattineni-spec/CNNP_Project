"""Queue and latency model — paper Eq. (11), (12), (14), (15).

⚠ CONVENTION: every quantity in this file is in **bits per time slot** for
queue sizes and departures. Only when converting to latency do we turn
"bits/slot" back into "bits/s" by dividing by ΔT.

⚠ LATENCY DEFINITION (implementation choice, see below):
    For SELECTED users:     d = q/R + d_q          (transmission + wait)
    For UNSELECTED users:   d = 0   + d_q          (wait only)

The paper's Eq. (14) writes d = q/R + d_q without specifying what to do when
R = 0 (i.e. the user was not selected). Interpreting R = 0 literally makes
latency infinite and breaks the reward. Physical interpretation: an unserved
user is not currently transmitting; only their accumulated wait time matters.
"""
from __future__ import annotations

import numpy as np

from config import SimulationConfig


class QueueModel:
    """Per-user traffic queues at a single BS."""

    def __init__(self, cfg: SimulationConfig, num_users: int, seed: int | None = None) -> None:
        self.cfg = cfg
        self.num_users = num_users
        self.rng = np.random.default_rng(
            cfg.random_seed if seed is None else seed
        )
        self.queues = np.zeros(num_users, dtype=float)
        self.queuing_latency = np.zeros(num_users, dtype=float)
        self.mean_arrivals_bits_per_slot = (
            cfg.user_arrival_rate_bps * cfg.slot_duration_s
        )

    # ------------------------------------------------------------------
    def sample_arrivals(self) -> np.ndarray:
        """Poisson arrivals D_{k_m}(t), one value per user, in bits/slot."""
        return self.rng.poisson(
            self.mean_arrivals_bits_per_slot, size=self.num_users
        ).astype(float)

    def step(
        self,
        selected: np.ndarray,
        rates_bps: np.ndarray,
        arrivals_bits_per_slot: np.ndarray,
    ) -> None:
        """Advance one time slot (Eq. 11 and Eq. 15)."""
        departures_bits = rates_bps * self.cfg.slot_duration_s
        self.queues = np.maximum(self.queues - departures_bits, 0.0) + arrivals_bits_per_slot
        for k in range(self.num_users):
            if selected[k]:
                self.queuing_latency[k] = 0.0
            else:
                self.queuing_latency[k] += self.cfg.slot_duration_s

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def queue_seconds(self, rates_bps: np.ndarray, selected: np.ndarray) -> np.ndarray:
        """q/R in seconds for SELECTED users; 0 for UNSELECTED users."""
        selected = np.asarray(selected, dtype=bool)
        safe_rate = np.where(selected, np.maximum(rates_bps, 1.0), 1.0)
        q_time = np.where(selected, self.queues / safe_rate, 0.0)
        return q_time

    def total_latency(self, rates_bps: np.ndarray, selected: np.ndarray) -> np.ndarray:
        """Eq. (14) with the unselected-users convention above."""
        return self.queue_seconds(rates_bps, selected) + self.queuing_latency

    def is_stable(self) -> bool:
        """Eq. (12): E[q] < q_max (instantaneous approximation)."""
        return bool(np.all(self.queues < self.cfg.max_queue_bits))
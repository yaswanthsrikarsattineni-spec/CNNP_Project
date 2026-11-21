"""CC-MAHDRL baseline (paper Section V).

Paper: "A centralized-critic based hierarchical framework that integrates DQN
and MADDPG to respectively deal with discrete and continuous actions."

⚠ HONEST DOCUMENTATION:
    For a single-agent deployment (M=1, num_bs=1), CC-MAHDRL and MAHDRL are
    mathematically identical — no mixing network can operate over one agent.
    We therefore implement CC-MAHDRL as a MAHDRL variant (no mixer) trained
    with a different seed, and label it clearly in the figure legend.
"""
from __future__ import annotations

from agents.mahdrl import MAHDRLAgent


class CCMAHDRLAgent(MAHDRLAgent):
    """MAHDRL with mixer forcibly disabled (M=1 degenerate)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixer = None
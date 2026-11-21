"""SINR calculation per paper Eq. (7).

    γ_k = |h_k w_k|^2 / (Σ_{j≠k} |h_k w_j|^2 + σ^2)

⚠ Note: in the paper's Eq. (6) the denominator also includes inter-cell
interference from other BSs. With `num_bs=1` (our baseline) there is no inter-
cell term; if you later increase `num_bs`, add those channels to the interference
list — the formula stays identical.
"""
from __future__ import annotations

import numpy as np

from config import SimulationConfig


def noise_power_w(cfg: SimulationConfig) -> float:
    """Total noise power in the full band (linear watts).

    noise_dBm = PSD_dBm_per_Hz + 10 log10(B)
    noise_W  = 10^((noise_dBm - 30) / 10)
    """
    noise_dbm = cfg.noise_psd_dbm_per_hz + 10.0 * np.log10(cfg.bandwidth_hz)
    return 10.0 ** ((noise_dbm - 30.0) / 10.0)


def compute_sinr(
    h_serving: np.ndarray,
    w_serving: np.ndarray,
    h_interference: list[np.ndarray],
    w_interference: list[np.ndarray],
    noise_w: float,
) -> float:
    """Eq. (7) for one user.

    h_serving      : (1, N_tx) useful channel
    w_serving      : (N_tx, 1) useful beamformer
    h_interference : list of (1, N_tx) — every other user's channel
    w_interference : list of (N_tx, 1) — every other user's beamformer
    noise_w        : scalar, linear watts
    """
    useful = abs(np.asarray(h_serving @ w_serving).item()) ** 2
    interference = sum(
        abs(np.asarray(h_i @ w_i).item()) ** 2
        for h_i, w_i in zip(h_interference, w_interference)
    )
    return float(useful / (interference + noise_w))


def compute_all_user_sinrs(
    channels: list[np.ndarray],
    beamformers: list[np.ndarray],
    noise_w: float,
) -> list[float]:
    """Vectorised convenience wrapper: SINR for every selected user."""
    K = len(channels)
    sinrs: list[float] = []
    for k in range(K):
        h_k = channels[k]
        useful = abs(np.asarray(h_k @ beamformers[k]).item()) ** 2
        interference = sum(
            abs(np.asarray(h_k @ beamformers[j]).item()) ** 2
            for j in range(K) if j != k
        )
        sinrs.append(float(useful / (interference + noise_w)))
    return sinrs
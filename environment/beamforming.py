"""Beamforming strategies for the baseline evaluation.

⚠ ASSUMPTION: the paper does not state which beamformer the *initial* simulation
uses; the RL agents later learn the optimal beamformer. For a fixed baseline
before training, we provide MRT and ZF. Both are classical multi-user MIMO
precoders (see any standard textbook like Tse & Viswanath, "Fundamentals of
Wireless Communication").

    MRT:  w_k = sqrt(P_k) · h_k^H / ||h_k||
    ZF:   W   = H^H (H H^H)^{-1}  (columns normalized to unit norm, then scaled)
"""
from __future__ import annotations

import numpy as np


def mrt_beamformer(h: np.ndarray, power_w: float) -> np.ndarray:
    """Maximum Ratio Transmission for a single user.

    Parameters
    ----------
    h : np.ndarray
        Channel row of shape (1, N_tx) or (N_tx,).
    power_w : float
        Transmit power in linear watts.

    Returns
    -------
    np.ndarray of shape (N_tx, 1) — the beamforming vector w_k.
    """
    h_vec = np.asarray(h, dtype=complex).reshape(-1)
    norm = np.linalg.norm(h_vec)
    if norm < 1e-30:
        return np.zeros((h_vec.size, 1), dtype=complex)
    return np.sqrt(power_w) * (h_vec.conj() / norm).reshape(-1, 1)


def zf_beamformers(H: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Zero-Forcing beamformers for K users.

    Parameters
    ----------
    H : np.ndarray of shape (K, N_tx)
        Matrix whose k-th row is the channel of user k.
    powers : np.ndarray of shape (K,)
        Transmit power per user (linear watts).

    Returns
    -------
    np.ndarray of shape (N_tx, K). Column k is the beamformer for user k,
    scaled so that ||w_k||^2 = powers[k].
    """
    K, N = H.shape
    if K > N:
        raise ValueError(f"ZF needs K <= N_tx (antennas); got K={K}, N={N}")
    W_orig = H.conj().T @ np.linalg.inv(H @ H.conj().T)      # (N, K)
    norms = np.linalg.norm(W_orig, axis=0)                    # (K,)
    W_unit = W_orig / norms[np.newaxis, :]                    # unit columns
    return W_unit * np.sqrt(np.asarray(powers))[np.newaxis, :]


def equal_power_split(total_power_w: float, num_users: int) -> np.ndarray:
    """Split total transmit power equally among `num_users` selected users."""
    return np.full(num_users, total_power_w / max(num_users, 1), dtype=float)
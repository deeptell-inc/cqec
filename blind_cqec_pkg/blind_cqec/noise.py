"""Noise channels used in blind CQEC benchmarks.

All channels act on a density matrix expressed in the equally-spaced
energy eigenbasis $E_i = i$, for which the mode gap is $|\\Delta_{ij}| = |i-j|$.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def dephasing(rho: Array, gamma: float) -> Array:
    """Energy-gap-dependent dephasing: rho_{ij} -> exp(-gamma*|i-j|) rho_{ij}.

    Parameters
    ----------
    rho : (d, d) complex array
        Input density matrix.
    gamma : float
        Dephasing rate. gamma = 0 is identity; gamma -> infinity collapses
        to the diagonal.

    Returns
    -------
    (d, d) complex array
    """
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    idx = np.arange(d)
    mask = np.exp(-gamma * np.abs(idx[:, None] - idx[None, :]))
    return rho * mask


def depolarizing(rho: Array, p: float) -> Array:
    """Depolarizing channel: rho -> (1-p) rho + p I/d."""
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    return (1.0 - p) * rho + (p / d) * np.eye(d, dtype=complex)


def amplitude_damping(rho: Array, gamma: float) -> Array:
    """Generalized amplitude damping: cascaded decay |k> -> |k-1>, rate gamma.

    Population from excited states flows to the ground state; coherences
    between levels j, k (with j+k > 0) are multiplied by (1-gamma)^{(j+k)/2}.
    """
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    out = np.zeros_like(rho)
    for i in range(d):
        for j in range(d):
            if i == 0 and j == 0:
                out[0, 0] = rho[0, 0] + gamma * sum(
                    rho[k, k] for k in range(1, d)
                )
            elif i == j:
                out[i, j] = (1.0 - gamma) * rho[i, j]
            else:
                fac = (1.0 - gamma) ** ((i + j) / 2.0) if (i > 0 or j > 0) else 1.0
                out[i, j] = rho[i, j] * fac
    return out


def combined_noise(
    rho: Array,
    gamma: float = 1.0,
    p: float = 0.15,
    gamma_ad: float = 0.1,
) -> Array:
    """Sequential composition: dephasing -> depolarizing -> amplitude damping."""
    return amplitude_damping(depolarizing(dephasing(rho, gamma), p), gamma_ad)

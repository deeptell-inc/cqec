"""Fidelity and coherence metrics."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def fidelity(rho: Array, sigma: Array) -> float:
    """Uhlmann (squared) quantum fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Robust against small numerical negativity in eigenvalues and
    near-rank-1 states. The result is clipped to [0, 1] to absorb
    floating-point noise at the boundaries.
    """
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    # Hermitise
    rho_h = (rho + rho.conj().T) / 2
    sigma_h = (sigma + sigma.conj().T) / 2

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        eigvals_r, eigvecs_r = np.linalg.eigh(rho_h)
        eigvals_r = np.clip(eigvals_r.real, 0.0, None)
        sqrt_eigs = np.sqrt(eigvals_r)
        sqrt_rho = (eigvecs_r * sqrt_eigs[None, :]) @ eigvecs_r.conj().T

        M = sqrt_rho @ sigma_h @ sqrt_rho
        M = (M + M.conj().T) / 2
        eigvals_M = np.linalg.eigvalsh(M)
        eigvals_M = np.clip(eigvals_M.real, 0.0, None)
        f = float(np.sum(np.sqrt(eigvals_M))) ** 2
    # Clip numerical excursions outside [0, 1]
    if f > 1.0:
        f = 1.0
    elif f < 0.0:
        f = 0.0
    return f


def trace_distance(rho: Array, sigma: Array) -> float:
    """D(rho, sigma) = 0.5 * ||rho - sigma||_1."""
    diff = np.asarray(rho, dtype=complex) - np.asarray(sigma, dtype=complex)
    # Hermitian part
    diff = (diff + diff.conj().T) / 2
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def l1_coherence(rho: Array) -> float:
    """l1-norm of coherence: C_l1(rho) = sum_{i != j} |rho_{ij}|."""
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    off = np.abs(rho)
    return float(np.sum(off) - np.sum(np.diag(off)))

"""
Core utilities for quantum coherence resource theory.

Provides density matrix operations, noise channels, coherence measures,
and mode analysis used throughout the CQEC package.
"""

import numpy as np
from typing import Set, Tuple, Optional


# ============================================================
# Density matrix utilities
# ============================================================

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(ρ, σ) = (Tr √(√ρ σ √ρ))²."""
    sr = _matrix_sqrt(rho)
    M = sr @ sigma @ sr
    eigvals = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(eigvals, 0.0))) ** 2)


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def purity(rho: np.ndarray) -> float:
    """Purity Tr(ρ²)."""
    return float(np.real(np.trace(rho @ rho)))


def ensure_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Project onto valid density matrix (positive semidefinite, trace 1)."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0.0)
    rho_out = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    tr = np.trace(rho_out)
    if tr > 1e-15:
        rho_out /= tr
    return rho_out


def random_density_matrix(d: int, rng=None) -> np.ndarray:
    """Random full-rank density matrix of dimension d."""
    if rng is None:
        rng = np.random.default_rng()
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def concurrence(rho: np.ndarray) -> float:
    """Concurrence for a 2-qubit state (d=4)."""
    assert rho.shape == (4, 4), "Concurrence is defined for 2-qubit states"
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y = np.kron(sy, sy)
    rho_tilde = Y @ rho.conj() @ Y
    R = _matrix_sqrt(rho) @ rho_tilde @ _matrix_sqrt(rho)
    eigvals = np.sqrt(np.maximum(np.linalg.eigvalsh(R), 0.0))
    eigvals = np.sort(eigvals)[::-1]
    return float(max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


# ============================================================
# Coherence measures
# ============================================================

def l1_coherence(rho: np.ndarray) -> float:
    """ℓ₁-norm of coherence: Σ_{i≠j} |ρ_{ij}|."""
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def coherence_modes(rho: np.ndarray, tol: float = 1e-14) -> Set[Tuple[int, int]]:
    """Coherent modes D(ρ) = {(i,j) : |ρ_{ij}| > tol, i < j}."""
    d = rho.shape[0]
    modes = set()
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho[i, j]) > tol:
                modes.add((i, j))
    return modes


def mode_inclusion(rho_target: np.ndarray, rho_noisy: np.ndarray,
                   tol: float = 1e-14) -> bool:
    """Check C(ρ_target) ⊆ C(ρ_noisy)."""
    target_modes = coherence_modes(rho_target, tol)
    noisy_modes = coherence_modes(rho_noisy, tol)
    return target_modes.issubset(noisy_modes)


def quantum_fisher_information(rho: np.ndarray, H: np.ndarray) -> float:
    """Quantum Fisher Information F(ρ, H)."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    d = rho.shape[0]
    qfi = 0.0
    for i in range(d):
        for j in range(d):
            denom = eigvals[i] + eigvals[j]
            if denom > 1e-15:
                h_ij = eigvecs[:, i].conj() @ H @ eigvecs[:, j]
                qfi += 2 * (eigvals[i] - eigvals[j])**2 / denom * np.abs(h_ij)**2
    return float(qfi)


# ============================================================
# Noise channels
# ============================================================

def dephasing_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Energy-dependent dephasing: ρ_{ij} → ρ_{ij} e^{-γ|i-j|}."""
    d = rho.shape[0]
    mask = np.exp(-gamma * np.abs(
        np.arange(d)[:, None] - np.arange(d)[None, :]))
    return rho * mask


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing: ρ → (1-p)ρ + p I/d."""
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


def amplitude_damping_channel(rho: np.ndarray, gamma_ad: float) -> np.ndarray:
    """Per-qubit amplitude damping."""
    d = rho.shape[0]
    n_qubits = int(np.log2(d))
    result = rho.copy()
    E0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma_ad)]], dtype=complex)
    E1_1q = np.array([[0, np.sqrt(gamma_ad)], [0, 0]], dtype=complex)
    for q in range(n_qubits):
        I_pre = np.eye(2**q, dtype=complex)
        I_post = np.eye(2**(n_qubits - q - 1), dtype=complex)
        E0 = np.kron(np.kron(I_pre, E0_1q), I_post)
        E1 = np.kron(np.kron(I_pre, E1_1q), I_post)
        result = E0 @ result @ E0.conj().T + E1 @ result @ E1.conj().T
    return result


def combined_noise(rho: np.ndarray,
                   gamma_deph: float = 1.0,
                   p_depol: float = 0.15,
                   gamma_ad: float = 0.1) -> np.ndarray:
    """Sequential dephasing + depolarizing + amplitude damping."""
    out = dephasing_channel(rho, gamma_deph)
    out = depolarizing_channel(out, p_depol)
    out = amplitude_damping_channel(out, gamma_ad)
    return out

"""Density-matrix-level catalytic recovery (ICEC) primitive and PSD projection."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def psd_project(rho: Array) -> Array:
    """Project onto the PSD cone via eigenvalue clipping, then renormalize.

    Ensures Hermitian input via symmetrization.
    """
    rho = np.asarray(rho, dtype=complex)
    rho = (rho + rho.conj().T) / 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals.real, 0.0)
        out = (eigvecs * eigvals[None, :]) @ eigvecs.conj().T
    tr = np.trace(out).real
    if tr < 1e-15:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    return out / tr


def icec_recover(
    noisy: Array, target_estimate: Array, mode_tol: float = 1e-10
) -> Array:
    """Mode-inclusion-restricted catalytic recovery (Theorem 1).

    The catalytic covariant transformation can only amplify coherence in
    modes that are still present in the noisy state: the recovered state
    must satisfy the mode-inclusion constraint
    C(rho_est) subset of C(rho_noisy). Coherences (i, j) whose magnitude in
    the noisy state is below `mode_tol` correspond to modes that decoherence
    has annihilated; they cannot be regenerated and are therefore zeroed in
    the recovered state, no matter what the target estimate claims.

    This makes recovery genuinely depend on `noisy`: under deep dephasing,
    modes that have vanished are unrecoverable, so F_rec <= F_est in general.

    Parameters
    ----------
    noisy : (d, d) complex array
        The noisy state whose surviving coherence modes bound the recovery.
    target_estimate : (d, d) complex array
        The estimate of the ideal target state.
    mode_tol : float
        Magnitude below which a noisy coherence counts as an absent mode.
        The default 1e-10 matches the mode-inclusion checker of the core
        library and the convention stated in the companion paper's Methods.

    Returns
    -------
    (d, d) complex array
        The recovered state (PSD, trace one).
    """
    noisy = np.asarray(noisy, dtype=complex)
    est = np.asarray(target_estimate, dtype=complex).copy()
    d = est.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j and abs(noisy[i, j]) < mode_tol:
                est[i, j] = 0.0
    return psd_project(est)

"""Target-state estimators for blind CQEC.

All estimators accept a noisy density matrix (or list thereof for multi-copy
strategies) and return an estimate of the pre-noise target state, suitable
as input to :func:`blind_cqec.recovery.icec_recover`.
"""
from __future__ import annotations

from typing import Sequence, Optional

import numpy as np

from .recovery import psd_project

Array = np.ndarray


def estimate_naive(noisy: Array) -> Array:
    """Use the noisy state itself as the estimate. Baseline (no improvement)."""
    return psd_project(np.asarray(noisy, dtype=complex))


def estimate_coherence_max(noisy: Array) -> Array:
    """Coherence maximization: restore coherences to the physicality bound.

    $(\\rho_est)_{ij} = \\sqrt{p_i p_j}\\,e^{i \\phi_{ij}}$ for $i \\neq j$,
    where $\\phi_{ij} = \\arg(\\rho_noisy)_{ij}$ when nonzero else 0.
    """
    rho = np.asarray(noisy, dtype=complex)
    d = rho.shape[0]
    pops = np.clip(np.real(np.diag(rho)), 0.0, None)
    # Build estimate
    est = np.zeros((d, d), dtype=complex)
    for i in range(d):
        est[i, i] = pops[i]
        for j in range(d):
            if i == j:
                continue
            mag = float(np.sqrt(pops[i] * pops[j]))
            phase = float(np.angle(rho[i, j])) if abs(rho[i, j]) > 1e-15 else 0.0
            est[i, j] = mag * np.exp(1j * phase)
    return psd_project(est)


def estimate_channel_inversion(
    noisy: Array,
    gamma: float = 0.0,
    p: float = 0.0,
    gamma_ad: float = 0.0,
) -> Array:
    """Analytical inversion of the combined noise channel.

    Assumes the channel was applied in the order
    :func:`blind_cqec.noise.combined_noise`
    (dephasing -> depolarizing -> amplitude damping) and inverts in the
    reverse order.

    Parameters
    ----------
    noisy : (d, d) complex array
    gamma, p, gamma_ad : float
        Assumed noise parameters. Pass 0.0 to skip that component.
    """
    rho = np.asarray(noisy, dtype=complex)
    d = rho.shape[0]
    est = rho.copy()

    # 1) invert amplitude damping (applied last)
    if gamma_ad > 0:
        inv = np.zeros_like(est)
        # off-diagonal and excited diagonal
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    # ground recovers by removing influx from excited
                    inv[0, 0] = est[0, 0] - gamma_ad * sum(
                        est[k, k] / (1.0 - gamma_ad) for k in range(1, d)
                    )
                elif i == j:
                    inv[i, j] = est[i, j] / (1.0 - gamma_ad)
                else:
                    fac = (1.0 - gamma_ad) ** ((i + j) / 2.0) if (i > 0 or j > 0) else 1.0
                    inv[i, j] = est[i, j] / fac if fac > 1e-15 else 0.0
        est = inv

    # 2) invert depolarizing
    if p > 0 and p < 1:
        est = (est - (p / d) * np.eye(d, dtype=complex)) / (1.0 - p)

    # 3) invert dephasing
    if gamma > 0:
        idx = np.arange(d)
        mask = np.exp(gamma * np.abs(idx[:, None] - idx[None, :]))
        np.fill_diagonal(mask, 1.0)
        est = est * mask

    return psd_project(est)


def estimate_iterative(
    noisy: Array,
    max_iter: int = 10,
    damping: float = 0.5,
    tol: float = 1e-4,
) -> Array:
    """Iterative refinement starting from the coherence-maximized estimate.

    Each step: run a virtual CQEC recovery toward the current estimate
    (the density-matrix ICEC simply returns the estimate), then blend
    with the previous estimate using `damping`.

    Because the density-matrix ICEC is the identity on the estimate,
    the iteration effectively averages successive coherence-max reapplications
    on the previous estimate, providing a mild regularization.
    """
    est = estimate_coherence_max(noisy)
    for _ in range(max_iter):
        # "Recovery" on the density-matrix level is psd_project(est); we
        # re-apply coherence_max to the blend for a nontrivial iteration.
        candidate = estimate_coherence_max(est)
        new_est = damping * candidate + (1.0 - damping) * est
        new_est = psd_project(new_est)
        if np.linalg.norm(new_est - est, ord="fro") < tol:
            est = new_est
            break
        est = new_est
    return est


def estimate_multicopy_average(
    noisy_copies: Sequence[Array],
) -> Array:
    """Average `n` noisy copies, then apply coherence maximization."""
    copies = [np.asarray(c, dtype=complex) for c in noisy_copies]
    if not copies:
        raise ValueError("Need at least one noisy copy")
    mean = np.mean(copies, axis=0)
    mean = psd_project(mean)
    return estimate_coherence_max(mean)


def estimate_hybrid(
    noisy: Array,
    weight: float,
    gamma: float = 0.0,
    p: float = 0.0,
    gamma_ad: float = 0.0,
) -> Array:
    """Convex combination of channel inversion and coherence maximization.

    weight=0 reduces to coherence maximization; weight=1 to channel inversion.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    cm = estimate_coherence_max(noisy)
    if weight == 0.0:
        return cm
    ci = estimate_channel_inversion(noisy, gamma=gamma, p=p, gamma_ad=gamma_ad)
    mix = weight * ci + (1.0 - weight) * cm
    return psd_project(mix)

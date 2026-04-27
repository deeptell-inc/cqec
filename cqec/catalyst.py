"""
Catalyst preparation strategies for CQEC.

Four methods of increasing sophistication:
1. Variational circuit (zero copies)
2. Standard recursive swap test
3. Covariant recursive swap test
4. DD+Twirl+Swap Test pipeline
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from cqec.core import (
    fidelity, purity, l1_coherence, coherence_modes,
    depolarizing_channel, dephasing_channel, ensure_density_matrix,
)


# ============================================================
# Swap test
# ============================================================

def swap_test(rho: np.ndarray, sigma: np.ndarray, d: int):
    """
    Standard swap test: project ρ⊗σ onto Sym²(C^d) via Π=(I+SWAP)/2.

    Returns (purified_state, success_probability).
    """
    d2 = d * d
    rs = np.kron(rho, sigma)
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    Pi = (np.eye(d2) + S) / 2.0
    proj = Pi @ rs @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15:
        return rho.copy(), 0.0
    proj /= p
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i * d + k, j * d + k]
    return out, p


def recursive_swap(rho_noisy: np.ndarray, d: int, n_rounds: int):
    """Recursive swap test: n_rounds rounds consuming 2^n_rounds copies."""
    rho = rho_noisy.copy()
    p_total = 1.0
    for _ in range(n_rounds):
        rho, p = swap_test(rho, rho, d)
        p_total *= p
    return rho, 2**n_rounds, p_total


# ============================================================
# Clifford twirling
# ============================================================

def twirl_analytical(rho_ideal: np.ndarray, gamma_eff: float,
                     d: int) -> tuple:
    """
    Analytical Clifford twirl: dephasing(γ_eff) → depolarizing(p_eff).

    Returns (twirled_state, p_eff).
    """
    e_g = np.exp(-gamma_eff)
    F_avg = e_g + (1 - e_g) * 2 / (d * (d + 1))
    p_eff = max(0.0, 1 - (d * F_avg - 1) / (d - 1))
    return depolarizing_channel(rho_ideal, p_eff), p_eff


# ============================================================
# DD+Twirl+Swap Test pipeline
# ============================================================

def dd_effective_gamma(gamma: float, n_dd: int) -> float:
    """Effective dephasing after CPMG-N: γ_eff = γ/(N+1)."""
    return gamma / (n_dd + 1)


def dd_twirl_pipeline(rho_ideal: np.ndarray, d: int,
                      gamma: float, n_copies: int,
                      n_dd: int = 8) -> dict:
    """
    Full DD+Twirl+Swap Test pipeline.

    Parameters
    ----------
    rho_ideal : target catalyst state (maximally coherent)
    d : Hilbert space dimension
    gamma : original dephasing strength
    n_copies : number of noisy copies (must be power of 2)
    n_dd : number of CPMG DD pulses

    Returns dict with rho_cat, fidelity, coherence, p_eff, gamma_eff.
    """
    # Stage 1: DD
    gamma_eff = dd_effective_gamma(gamma, n_dd)

    # Stage 2: Twirl
    rho_twirled, p_eff = twirl_analytical(rho_ideal, gamma_eff, d)

    # Stage 3: Recursive swap test
    n_rounds = max(1, int(np.log2(n_copies)))
    rho_cat = rho_twirled.copy()
    for _ in range(n_rounds):
        rho_cat, _ = swap_test(rho_cat, rho_cat, d)

    fid = fidelity(rho_ideal, rho_cat)
    coh = l1_coherence(rho_cat)

    return {
        'rho_cat': rho_cat,
        'fidelity': fid,
        'coherence': coh,
        'p_eff': p_eff,
        'gamma_eff': gamma_eff,
        'n_copies': n_copies,
        'n_dd': n_dd,
    }


# ============================================================
# Variational catalyst
# ============================================================

def variational_catalyst(d: int, n_layers: int = 3,
                         n_restarts: int = 5,
                         maxiter: int = 500,
                         seed: int = 42) -> dict:
    """
    Prepare catalyst via variational EC circuit (zero copies).

    Returns dict with rho_cat, coherence, mode coverage, n_params.
    """
    rng = np.random.default_rng(seed)
    n_pairs = d * (d - 1) // 2
    n_params = 2 * n_layers * n_pairs

    target_modes = set()
    for i in range(d):
        for j in range(i + 1, d):
            target_modes.add((i, j))

    def build_state(params):
        U = np.eye(d, dtype=complex)
        idx = 0
        for _ in range(n_layers):
            for i in range(d):
                for j in range(i + 1, d):
                    theta = params[idx]
                    phi = params[idx + 1]
                    G = np.eye(d, dtype=complex)
                    c, s = np.cos(theta), np.sin(theta)
                    G[i, i] = c
                    G[j, j] = c
                    G[i, j] = -np.exp(1j * phi) * s
                    G[j, i] = np.exp(-1j * phi) * s
                    U = G @ U
                    idx += 2
        psi = U[:, 0]
        return np.outer(psi, psi.conj())

    def cost(params):
        rho = build_state(params)
        coh = l1_coherence(rho) / (d - 1)
        modes = coherence_modes(rho)
        missing = len(target_modes - modes) / len(target_modes)
        rho_min = np.min(np.abs(np.diag(rho))) + 1e-15
        return -1.0 * coh + 10.0 * missing - 5.0 * np.log(rho_min)

    best_cost = np.inf
    best_params = None
    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        result = minimize(cost, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter})
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x

    rho_cat = build_state(best_params)
    cat_modes = coherence_modes(rho_cat)
    coverage = len(target_modes & cat_modes) / len(target_modes)

    return {
        'rho_cat': rho_cat,
        'coherence': l1_coherence(rho_cat),
        'rho_min': float(np.min(np.abs(np.diag(rho_cat)))),
        'modes_covered': coverage,
        'n_params': n_params,
    }

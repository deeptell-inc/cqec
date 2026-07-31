#!/usr/bin/env python3
"""
covariant_purification.py — Covariant Recursive Swap Test for CQEC

The standard swap test projects onto the symmetric subspace via SWAP,
which does NOT respect [U, H_total] = 0. Under dephasing noise, this
limits purification because the test mixes energy sectors.

This module implements a COVARIANT swap test gadget using EC gates
that respects energy conservation, preserving the mode structure
needed for CQEC catalyst preparation.

Key idea: replace SWAP with energy-conserving partial swap
  U_EC(π/2) = iSWAP restricted to degenerate subspaces,
  then project onto the symmetric sector within each energy block.

Architecture:
  1. Standard recursive swap test (baseline)
  2. Covariant recursive purification (EC-gate based)
  3. Optimised covariant purification (variational EC angles)
  4. Full CQEC pipeline benchmark
"""

import numpy as np
from scipy.optimize import minimize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def fidelity(rho, sigma):
    sqrt_rho = _msqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(eigvals, 0))) ** 2)

def _msqrt(A):
    e, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.maximum(e, 0))) @ v.conj().T

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def l1_coherence(rho):
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))

def dephasing_channel(rho, gamma):
    d = rho.shape[0]
    out = rho.copy()
    for i in range(d):
        for j in range(d):
            if i != j:
                out[i, j] *= np.exp(-gamma)
    return out

def depolarizing_channel(rho, delta):
    d = rho.shape[0]
    return (1 - delta) * rho + delta * np.eye(d) / d


# ============================================================
# 1. Standard Recursive Swap Test (baseline)
# ============================================================

def build_swap(d):
    """Build SWAP operator on C^d ⊗ C^d."""
    d2 = d * d
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S

def swap_test_purify(rho, sigma, d):
    """Standard swap test: project ρ⊗σ onto Sym², trace out."""
    d2 = d * d
    rho_sigma = np.kron(rho, sigma)
    SWAP = build_swap(d)
    Pi = (np.eye(d2) + SWAP) / 2.0
    proj = Pi @ rho_sigma @ Pi
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

def recursive_swap(rho_noisy, d, n_rounds):
    """Standard recursive: SWAP(ρ_k, ρ_k) → ρ_{k+1}, copies = 2^n_rounds."""
    rho = rho_noisy.copy()
    p_total = 1.0
    for _ in range(n_rounds):
        rho, p = swap_test_purify(rho, rho, d)
        p_total *= p
    return rho, 2**n_rounds, p_total


# ============================================================
# 2. Covariant Swap Test (EC-gate based)
# ============================================================

def ec_gate_2level(d, i, j, theta):
    """
    EC gate on C^d: rotation in the (i,j) subspace.
    U_EC(θ)|i⟩ = cos(θ)|i⟩ - i sin(θ)|j⟩
    U_EC(θ)|j⟩ = -i sin(θ)|i⟩ + cos(θ)|j⟩
    Satisfies [U_EC, H] = 0 when E_i + E_j = const (degenerate pair).
    """
    U = np.eye(d, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -1j * s
    U[j, i] = -1j * s
    return U


def build_covariant_swap(d, angles=None):
    """
    Build a covariant partial-swap on C^d ⊗ C^d using EC gates.

    For a d-dimensional system with H = diag(0,1,...,d-1),
    the degenerate subspaces of H⊗I + I⊗H on C^d⊗C^d are:
      E_total = k: spanned by {|i,j⟩ : i+j = k}

    Within each degenerate subspace, we can freely rotate.
    The covariant swap applies EC rotations within each sector.

    If angles is None, use π/4 (half-swap) for all pairs.
    """
    d2 = d * d
    U = np.eye(d2, dtype=complex)

    # Energy sectors: E_total = i + j
    pair_idx = 0
    for E_total in range(2 * (d - 1) + 1):
        # Basis states with i + j = E_total
        sector_states = []
        for i in range(d):
            j = E_total - i
            if 0 <= j < d:
                sector_states.append((i, j))

        if len(sector_states) < 2:
            continue

        # Apply EC rotations between consecutive pairs in this sector
        for k in range(len(sector_states) - 1):
            i1, j1 = sector_states[k]
            i2, j2 = sector_states[k + 1]
            idx1 = i1 * d + j1
            idx2 = i2 * d + j2

            if angles is not None and pair_idx < len(angles):
                theta = angles[pair_idx]
            else:
                theta = np.pi / 4  # half-swap by default

            c, s = np.cos(theta), np.sin(theta)
            U[idx1, idx1] = c
            U[idx2, idx2] = c
            U[idx1, idx2] = -1j * s
            U[idx2, idx1] = -1j * s
            pair_idx += 1

    return U, pair_idx  # return num_params for optimisation


def covariant_swap_purify(rho, sigma, d, angles=None):
    """
    Covariant swap test:
    1. Apply covariant partial-swap U_cov on ρ⊗σ
    2. Measure second register in computational basis
    3. Post-select on symmetric outcome (diagonal of σ matches)
    4. Partial trace over second register

    The key difference from standard swap: U_cov respects [U, H_tot] = 0,
    so the purified state preserves the coherent mode structure.
    """
    d2 = d * d
    rho_sigma = np.kron(rho, sigma)

    U_cov, _ = build_covariant_swap(d, angles)

    # Apply covariant unitary
    state = U_cov @ rho_sigma @ U_cov.conj().T

    # Project: measure second register, post-select on
    # outcome matching the diagonal structure
    # Simple version: trace out second register with projection
    # onto the "symmetric" sector defined by the EC rotation

    # CORRECTED: (I + SWAP_E)/2 within each energy sector
    # SWAP_E maps |i,j⟩ → |j,i⟩ within the sector (swaps subsystems)
    Pi = np.zeros((d2, d2), dtype=complex)
    for E_total in range(2 * (d - 1) + 1):
        sector_pairs = []
        for i in range(d):
            j = E_total - i
            if 0 <= j < d:
                sector_pairs.append((i, j))

        indices = [i * d + j for i, j in sector_pairs]
        n_s = len(sector_pairs)

        if n_s == 1:
            Pi[indices[0], indices[0]] = 1.0
        else:
            # Build SWAP_E: maps (i,j) → (j,i) within sector
            SWAP_E = np.zeros((n_s, n_s), dtype=complex)
            for a, (ia, ja) in enumerate(sector_pairs):
                for b, (ib, jb) in enumerate(sector_pairs):
                    if ib == ja and jb == ia:
                        SWAP_E[a, b] = 1.0
            # Symmetric projector: Π_sym = (I + SWAP_E) / 2
            Pi_E = (np.eye(n_s) + SWAP_E) / 2.0
            for a in range(n_s):
                for b in range(n_s):
                    Pi[indices[a], indices[b]] = Pi_E[a, b]

    # Project
    proj = Pi @ state @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15:
        return rho.copy(), 0.0
    proj /= p

    # Partial trace over second register
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i * d + k, j * d + k]

    return out, p


def recursive_covariant(rho_noisy, d, n_rounds, angles=None):
    """Recursive covariant purification."""
    rho = rho_noisy.copy()
    p_total = 1.0
    for _ in range(n_rounds):
        rho, p = covariant_swap_purify(rho, rho, d, angles)
        p_total *= p
    return rho, 2**n_rounds, p_total


# ============================================================
# 3. Optimised Covariant Purification
# ============================================================

def optimise_covariant_angles(rho_noisy, rho_target, d, n_rounds=3,
                              n_restarts=5, maxiter=300):
    """
    Optimise the EC gate angles in the covariant swap test
    to maximise catalyst fidelity after n_rounds of purification.
    """
    _, n_params = build_covariant_swap(d)

    best_result = None
    best_fid = -1
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        x0 = rng.uniform(0, np.pi / 2, n_params)

        def objective(angles):
            rho_out, _, _ = recursive_covariant(rho_noisy, d, n_rounds, angles)
            return -fidelity(rho_target, rho_out)

        result = minimize(objective, x0, method='L-BFGS-B',
                          bounds=[(0, np.pi)] * n_params,
                          options={'maxiter': maxiter, 'ftol': 1e-12})

        if -result.fun > best_fid:
            best_fid = -result.fun
            best_result = result

    return best_result.x, best_fid


# ============================================================
# 4. CQEC Recovery
# ============================================================

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    """Simplified CQEC recovery."""
    d = rho_target.shape[0]
    tol = 1e-10
    rho_rec = rho_noisy.copy()

    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_target[i, j]) > tol and np.abs(rho_cat[i, j]) > tol:
                cat_coh = np.abs(rho_cat[i, j])
                noisy_coh = np.abs(rho_noisy[i, j])
                if noisy_coh > 1e-15:
                    phase = np.angle(rho_target[i, j])
                    mag_t = np.abs(rho_target[i, j])
                    eff = 1.0 - np.exp(-cat_coh * d * purity(rho_cat))
                    mag_r = noisy_coh + eff * (mag_t - noisy_coh)
                    rho_rec[i, j] = mag_r * np.exp(1j * phase)
                    rho_rec[j, i] = rho_rec[i, j].conj()

    e, v = np.linalg.eigh(rho_rec)
    e = np.maximum(e, 0)
    rho_rec = v @ np.diag(e) @ v.conj().T
    rho_rec /= np.trace(rho_rec)
    return rho_rec


# ============================================================
# 5. Benchmark
# ============================================================

def run_benchmark():
    rng = np.random.default_rng(42)
    dims = [4, 8]
    copy_counts = [2, 4, 8, 16, 32, 64]

    noise_configs = [
        ('Depolarizing', lambda rho: depolarizing_channel(rho, 0.3)),
        ('Dephasing', lambda rho: dephasing_channel(rho, 2.0)),
        ('Combined', lambda rho: dephasing_channel(
            depolarizing_channel(rho, 0.15), 1.0)),
    ]

    methods = ['standard', 'covariant', 'optimised']
    method_labels = {
        'standard': 'Standard recursive',
        'covariant': 'Covariant recursive',
        'optimised': 'Optimised covariant',
    }
    method_colors = {
        'standard': '#e74c3c',
        'covariant': '#3498db',
        'optimised': '#2ecc71',
    }

    all_results = {}

    print("=" * 90)
    print("Covariant Recursive Purification Benchmark")
    print("=" * 90)

    for d in dims:
        print(f"\n{'#' * 70}")
        print(f"# d = {d} ({int(np.log2(d))} qubits)")
        print(f"{'#' * 70}")

        # Target: maximally coherent catalyst
        psi_cat = np.ones(d, dtype=complex) / np.sqrt(d)
        rho_cat_ideal = np.outer(psi_cat, psi_cat.conj())

        # CQEC target state
        psi_t = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi_t /= np.linalg.norm(psi_t)
        rho_target = np.outer(psi_t, psi_t.conj())
        rho_noisy_cqec = dephasing_channel(rho_target, 2.0)
        fid_base = fidelity(rho_target, rho_noisy_cqec)
        print(f"CQEC target: F_noisy = {fid_base:.4f}")

        dim_results = {}

        for noise_name, noise_fn in noise_configs:
            print(f"\n--- {noise_name} noise on catalyst ---")
            rho_cat_noisy = noise_fn(rho_cat_ideal)
            fid0 = fidelity(rho_cat_ideal, rho_cat_noisy)
            print(f"  F_cat(noisy) = {fid0:.4f}, C_l1 = {l1_coherence(rho_cat_noisy):.4f}")

            # Pre-compute optimised angles for this noise
            print("  Optimising covariant angles...", end=' ', flush=True)
            t0 = time.time()
            opt_angles, opt_fid = optimise_covariant_angles(
                rho_cat_noisy, rho_cat_ideal, d,
                n_rounds=3, n_restarts=3, maxiter=200)
            print(f"done ({time.time()-t0:.1f}s, F={opt_fid:.4f})")

            noise_results = {}
            for method in methods:
                method_res = []
                for n_copies in copy_counts:
                    n_rounds = max(1, int(np.log2(n_copies)))
                    t0 = time.time()

                    if method == 'standard':
                        rho_out, used, p = recursive_swap(
                            rho_cat_noisy, d, n_rounds)
                    elif method == 'covariant':
                        rho_out, used, p = recursive_covariant(
                            rho_cat_noisy, d, n_rounds)
                    elif method == 'optimised':
                        rho_out, used, p = recursive_covariant(
                            rho_cat_noisy, d, n_rounds, opt_angles)

                    fid_cat = fidelity(rho_cat_ideal, rho_out)
                    coh = l1_coherence(rho_out)

                    # CQEC recovery test
                    rho_rec = cqec_recovery(rho_target, rho_noisy_cqec, rho_out)
                    fid_rec = fidelity(rho_target, rho_rec)
                    dt = time.time() - t0

                    method_res.append({
                        'n': n_copies, 'fid_cat': fid_cat,
                        'coh': coh, 'p': p,
                        'fid_rec': fid_rec, 'time': dt,
                    })

                noise_results[method] = method_res

            # Print table
            print(f"\n  {'Method':<24} {'n':>4} {'F_cat':>8} {'C_l1':>8} "
                  f"{'p_succ':>8} {'F_rec':>8}")
            print(f"  {'-'*64}")
            for method in methods:
                for r in noise_results[method]:
                    print(f"  {method_labels[method]:<24} "
                          f"{r['n']:>4} {r['fid_cat']:>8.5f} "
                          f"{r['coh']:>8.4f} {r['p']:>8.4f} "
                          f"{r['fid_rec']:>8.5f}")

            dim_results[noise_name] = noise_results

        all_results[d] = dim_results

    return all_results, dims, copy_counts, methods, method_labels, method_colors


def plot_results(all_results, dims, copy_counts, methods,
                 method_labels, method_colors):
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 8, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    noise_names = ['Depolarizing', 'Dephasing', 'Combined']
    n_noise = len(noise_names)
    n_dims = len(dims)

    # --- Catalyst fidelity: dims × noise ---
    fig, axes = plt.subplots(n_dims, n_noise, figsize=(6*n_noise, 5*n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :]

    for di, d in enumerate(dims):
        for ni, noise in enumerate(noise_names):
            ax = axes[di, ni]
            for method in methods:
                res = all_results[d][noise][method]
                ns = [r['n'] for r in res]
                fs = [r['fid_cat'] for r in res]
                ax.semilogx(ns, fs, 'o-', color=method_colors[method],
                            label=method_labels[method], linewidth=2,
                            markersize=5)
            ax.axhline(y=1.0, color='gray', ls=':', alpha=0.3)
            ax.set_xlabel(r'Copies $n$')
            ax.set_ylabel(r'$F(\rho_{\mathrm{cat}},\rho_{\mathrm{ideal}})$')
            ax.set_title(f'$d={d}$, {noise}')
            ax.legend(loc='lower right', fontsize=7)
            ax.set_ylim(auto=True)

    plt.suptitle('Covariant vs. Standard Recursive Purification',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_covariant_purification.png')
    plt.savefig('fig_covariant_purification.pdf')
    print("\nSaved fig_covariant_purification.png/pdf")
    plt.close()

    # --- CQEC recovery comparison ---
    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for di, d in enumerate(dims):
        ax = axes[di]
        # Dephasing results (hardest case)
        noise = 'Dephasing'
        for method in methods:
            res = all_results[d][noise][method]
            ns = [r['n'] for r in res]
            fs = [r['fid_rec'] for r in res]
            ax.semilogx(ns, fs, 's-', color=method_colors[method],
                        label=method_labels[method], linewidth=2,
                        markersize=6)
        ax.axhline(y=1.0, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel(r'Catalyst copies $n$')
        ax.set_ylabel(r'CQEC $F(\rho_{\mathrm{rec}},\rho_0)$')
        ax.set_title(f'$d={d}$, dephasing $\\gamma=2$')
        ax.legend(fontsize=8)

    plt.suptitle('CQEC Recovery: Covariant Purified Catalysts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_covariant_cqec.png')
    plt.savefig('fig_covariant_cqec.pdf')
    print("Saved fig_covariant_cqec.png/pdf")
    plt.close()

    # --- Coherence preservation ---
    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for di, d in enumerate(dims):
        ax = axes[di]
        noise = 'Dephasing'
        for method in methods:
            res = all_results[d][noise][method]
            ns = [r['n'] for r in res]
            cs = [r['coh'] for r in res]
            ax.semilogx(ns, cs, 'D-', color=method_colors[method],
                        label=method_labels[method], linewidth=2,
                        markersize=5)
        # Max coherence reference
        d_max_coh = d * (d - 1) / d  # C_l1 of maximally coherent state
        ax.axhline(y=d_max_coh, color='gray', ls='--', alpha=0.5,
                   label=r'$C_{\ell_1}^{\max}$')
        ax.set_xlabel(r'Copies $n$')
        ax.set_ylabel(r'$\mathcal{C}_{\ell_1}(\rho_{\mathrm{cat}})$')
        ax.set_title(f'$d={d}$, coherence under dephasing')
        ax.legend(fontsize=8)

    plt.suptitle(r'Coherence Recovery: $\ell_1$-Norm after Purification',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_covariant_coherence.png')
    plt.savefig('fig_covariant_coherence.pdf')
    print("Saved fig_covariant_coherence.png/pdf")
    plt.close()


if __name__ == '__main__':
    results, dims, copies, methods, labels, colors = run_benchmark()
    plot_results(results, dims, copies, methods, labels, colors)

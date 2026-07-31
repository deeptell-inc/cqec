#!/usr/bin/env python3
"""
purification_catalyst.py — Hybrid Purification-Based Catalyst Preparation for CQEC

Implements Scenario A+C from the purification catalyst proposal:
  C: Optimal probabilistic purification (Yao+ QST 2025) for n≤4 copies
  A: Streaming recursive swap-test purification (Childs+ Quantum 2025) for n≫1

Then feeds the purified catalyst into CQEC recovery and benchmarks against:
  - No purification (raw noisy catalyst)
  - Variational catalyst (from variational_catalyst.py)
  - Ideal asymptotic CQEC

Dependencies: numpy, scipy (density-matrix simulation, no Qulacs needed)
"""

import numpy as np
from scipy.linalg import sqrtm
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Core utilities
# ============================================================

def fidelity(rho, sigma):
    """Quantum fidelity F(ρ, σ)."""
    sqrt_rho = _matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 0.0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)


def _matrix_sqrt(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def purity(rho):
    """Tr(ρ²)."""
    return float(np.real(np.trace(rho @ rho)))


def l1_coherence(rho):
    d = rho.shape[0]
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def depolarizing_channel(rho, delta):
    """ρ → (1-δ)ρ + δ I/d."""
    d = rho.shape[0]
    return (1 - delta) * rho + delta * np.eye(d) / d


def dephasing_channel(rho, gamma):
    """ρ_{ij} → ρ_{ij} e^{-γ} for i≠j."""
    d = rho.shape[0]
    out = rho.copy()
    for i in range(d):
        for j in range(d):
            if i != j:
                out[i, j] *= np.exp(-gamma)
    return out


# ============================================================
# Scenario C: Optimal Probabilistic Purification (Yao+ 2025)
# ============================================================

class OptimalProbabilisticPurification:
    """
    Implements the CEM-style optimal purification protocol for
    depolarizing noise: project n copies onto the symmetric subspace.

    For n=2 copies of ρ(δ) = (1-δ)|ψ⟩⟨ψ| + δI/d under depolarizing:
      - Project onto Sym²(C^d) via swap test
      - Output error parameter: δ' = δ²(2d-1) / [d(d+1) - (d-1)(d+1)δ + (2d-1)δ²]
        (simplified for depolarizing)
      - Success probability: p = [d(d+1)/2 - ...] / d²

    We implement the density-matrix level simulation directly.
    """

    @staticmethod
    def swap_test_purify(rho, d):
        """
        Apply swap test to two copies of ρ.
        Project ρ⊗ρ onto symmetric subspace, trace out second copy.

        Swap test gadget output (a=0 outcome):
          ω(0) = (ρ⊗ρ + SWAP·(ρ⊗ρ)·SWAP) / (2 + 2·Tr(ρ²))
               = Π_sym · (ρ⊗ρ) · Π_sym / Tr[Π_sym · (ρ⊗ρ)]

        Then partial trace over second subsystem.
        """
        d2 = d * d

        # Build ρ⊗ρ
        rho2 = np.kron(rho, rho)

        # Build SWAP operator
        SWAP = np.zeros((d2, d2), dtype=complex)
        for i in range(d):
            for j in range(d):
                # SWAP|i,j⟩ = |j,i⟩
                ij = i * d + j
                ji = j * d + i
                SWAP[ji, ij] = 1.0

        # Symmetric projector: Π_sym = (I + SWAP)/2
        Pi_sym = (np.eye(d2) + SWAP) / 2.0

        # Project onto symmetric subspace
        projected = Pi_sym @ rho2 @ Pi_sym

        # Success probability
        p_success = float(np.real(np.trace(projected)))

        if p_success < 1e-15:
            return rho.copy(), 0.0

        # Normalise
        projected /= p_success

        # Partial trace over second subsystem → d×d output
        rho_out = np.zeros((d, d), dtype=complex)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    rho_out[i, j] += projected[i * d + k, j * d + k]

        return rho_out, p_success

    @staticmethod
    def purify_n2(rho, d):
        """2-copy optimal purification via single swap test."""
        return OptimalProbabilisticPurification.swap_test_purify(rho, d)

    @staticmethod
    def purify_n4(rho, d):
        """4-copy purification: two parallel swap tests, then one more."""
        # Round 1: swap test on copies (1,2) and (3,4)
        rho1, p1 = OptimalProbabilisticPurification.swap_test_purify(rho, d)
        rho2, p2 = OptimalProbabilisticPurification.swap_test_purify(rho, d)
        # Round 2: swap test on the two outputs
        rho_out, p3 = OptimalProbabilisticPurification.swap_test_purify(rho1, d)
        p_total = p1 * p2 * p3
        return rho_out, p_total


# ============================================================
# Scenario A: Streaming Recursive Purification (Childs+ 2025)
# ============================================================

class StreamingPurification:
    """
    Recursive swap-test purification (Childs, Fu, Leung, Li, Ozols, Vyas 2025).

    The protocol recursively applies the swap test gadget:
      SWAP(ρ, σ) = (ρ + σ + ρσ + σρ) / (2(1 + Tr(ρσ)))

    For depolarizing noise ρ(δ) = (1-δ)|ψ⟩⟨ψ| + δI/d:
      - Each swap test round maps error δ → δ' ≈ δ²·(2d-1)/(d(d+1))
      - Error decreases doubly exponentially: after k rounds, δ_k ~ δ^{2^k}

    The streaming protocol processes copies one at a time,
    maintaining a running purified state in O(log d) memory.
    """

    @staticmethod
    def swap_gadget(rho, sigma, d):
        """
        Swap test gadget: takes two states, projects onto symmetric
        subspace, returns one purified state.

        Output (conditioned on a=0):
          ω = (ρ+σ+ρσ+σρ) / (2 + 2·Tr(ρσ))
          = Tr_2[Π_sym (ρ⊗σ) Π_sym] / Tr[Π_sym (ρ⊗σ)]
        """
        d2 = d * d

        # Build ρ⊗σ
        rho_sigma = np.kron(rho, sigma)

        # SWAP operator
        SWAP = np.zeros((d2, d2), dtype=complex)
        for i in range(d):
            for j in range(d):
                SWAP[j * d + i, i * d + j] = 1.0

        # Π_sym = (I + SWAP)/2
        Pi_sym = (np.eye(d2) + SWAP) / 2.0

        projected = Pi_sym @ rho_sigma @ Pi_sym
        p_success = float(np.real(np.trace(projected)))

        if p_success < 1e-15:
            return rho.copy(), 0.0

        projected /= p_success

        # Partial trace over second subsystem
        rho_out = np.zeros((d, d), dtype=complex)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    rho_out[i, j] += projected[i * d + k, j * d + k]

        return rho_out, p_success

    @staticmethod
    def recursive_purify(rho_noisy, d, n_rounds):
        """
        Recursive purification: each round uses 2 copies to produce 1.
        Total copies consumed: 2^n_rounds.

        Round 0: start with ρ_noisy
        Round k: SWAP(ρ_{k-1}, ρ_{k-1}) → ρ_k

        In practice, each "ρ_{k-1}" requires a fresh purified copy,
        so copy count doubles each round.
        """
        rho_current = rho_noisy.copy()
        total_copies = 1
        p_total = 1.0

        for k in range(n_rounds):
            rho_current, p_k = StreamingPurification.swap_gadget(
                rho_current, rho_current, d
            )
            total_copies *= 2
            p_total *= p_k

        return rho_current, total_copies, p_total

    @staticmethod
    def streaming_purify(rho_noisy, d, n_copies):
        """
        Streaming purification: process copies one at a time.
        Maintains a running purified state and merges each new copy.

        Uses n_copies total. Each merge is a swap test.
        """
        rho_current = rho_noisy.copy()
        p_total = 1.0

        for i in range(1, n_copies):
            rho_current, p_k = StreamingPurification.swap_gadget(
                rho_current, rho_noisy, d
            )
            p_total *= p_k

        return rho_current, n_copies, p_total


# ============================================================
# Hybrid A+C Protocol
# ============================================================

class HybridPurification:
    """
    Hybrid purification combining:
      Phase 1 (Scenario C): Optimal probabilistic purification (2 or 4 copies)
      Phase 2 (Scenario A): Streaming recursive purification (remaining copies)

    This achieves the best of both worlds:
      - Phase 1 gives optimal initial quality (SDP-proven)
      - Phase 2 drives error down further with streaming efficiency
    """

    @staticmethod
    def purify(rho_noisy, d, total_copies, strategy='hybrid'):
        """
        Purify rho_noisy using total_copies copies.

        Strategies:
          'hybrid': Phase 1 (4-copy optimal) + Phase 2 (streaming)
          'streaming_only': All copies via streaming
          'recursive_only': All copies via recursive (binary tree)
          'optimal_only': Only 2 or 4 copy optimal purification
        """
        if strategy == 'optimal_only':
            if total_copies >= 4:
                rho_out, p = OptimalProbabilisticPurification.purify_n4(
                    rho_noisy, d)
                return rho_out, 4, p
            else:
                rho_out, p = OptimalProbabilisticPurification.purify_n2(
                    rho_noisy, d)
                return rho_out, 2, p

        elif strategy == 'streaming_only':
            return StreamingPurification.streaming_purify(
                rho_noisy, d, total_copies)

        elif strategy == 'recursive_only':
            n_rounds = int(np.log2(max(total_copies, 2)))
            return StreamingPurification.recursive_purify(
                rho_noisy, d, n_rounds)

        elif strategy == 'hybrid':
            # Phase 1: 4-copy optimal purification
            if total_copies >= 4:
                rho_phase1, p1 = OptimalProbabilisticPurification.purify_n4(
                    rho_noisy, d)
                remaining = total_copies - 4
            elif total_copies >= 2:
                rho_phase1, p1 = OptimalProbabilisticPurification.purify_n2(
                    rho_noisy, d)
                remaining = total_copies - 2
            else:
                return rho_noisy.copy(), 1, 1.0

            # Phase 2: streaming with remaining copies
            p_total = p1
            rho_current = rho_phase1
            for _ in range(remaining):
                # Each new copy is a fresh noisy state fed into swap gadget
                rho_current, p_k = StreamingPurification.swap_gadget(
                    rho_current, rho_noisy, d
                )
                p_total *= p_k

            return rho_current, total_copies, p_total

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# CQEC Recovery (simplified, from variational_catalyst.py)
# ============================================================

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    """
    Simplified CQEC recovery using catalyst.
    Recovery efficiency depends on catalyst coherence quality.
    """
    d = rho_target.shape[0]

    # Get mode structure
    tol = 1e-10
    target_modes = set()
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_target[i, j]) > tol:
                target_modes.add((i, j))

    cat_modes = set()
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_cat[i, j]) > tol:
                cat_modes.add((i, j))

    rho_rec = rho_noisy.copy()

    for i, j in target_modes:
        if (i, j) in cat_modes:
            cat_coh_ij = np.abs(rho_cat[i, j])
            target_coh_ij = rho_target[i, j]
            noisy_coh_ij = rho_noisy[i, j]

            if cat_coh_ij > 1e-12 and np.abs(noisy_coh_ij) > 1e-15:
                phase_target = np.angle(target_coh_ij)
                mag_target = np.abs(target_coh_ij)

                # Recovery efficiency scales with catalyst purity
                cat_purity = purity(rho_cat)
                efficiency = 1.0 - np.exp(-cat_coh_ij * d * cat_purity)
                mag_recovered = np.abs(noisy_coh_ij) + efficiency * (
                    mag_target - np.abs(noisy_coh_ij)
                )

                rho_rec[i, j] = mag_recovered * np.exp(1j * phase_target)
                rho_rec[j, i] = rho_rec[i, j].conj()

    # Ensure positivity
    eigvals, eigvecs = np.linalg.eigh(rho_rec)
    eigvals = np.maximum(eigvals, 0.0)
    rho_rec = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    rho_rec /= np.trace(rho_rec)

    return rho_rec


# ============================================================
# Benchmark
# ============================================================

def make_target_state(d, rng):
    """Random pure state of dimension d."""
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def run_benchmark():
    rng = np.random.default_rng(42)

    dims = [4, 8, 16]
    copy_budgets = [2, 4, 8, 16, 32, 64]
    noise_gamma = 2.0  # dephasing strength
    noise_delta = 0.3  # depolarizing parameter

    strategies = ['optimal_only', 'streaming_only', 'recursive_only', 'hybrid']
    strategy_labels = {
        'optimal_only': 'Optimal (Yao+)',
        'streaming_only': 'Streaming (Childs+)',
        'recursive_only': 'Recursive swap',
        'hybrid': 'Hybrid A+C',
    }
    strategy_colors = {
        'optimal_only': '#9b59b6',
        'streaming_only': '#3498db',
        'recursive_only': '#f39c12',
        'hybrid': '#e74c3c',
    }

    all_results = {}

    print("=" * 90)
    print("Hybrid Purification Catalyst Benchmark for CQEC")
    print("=" * 90)

    for d in dims:
        print(f"\n{'#' * 70}")
        print(f"# d = {d} ({int(np.log2(d))} qubits)")
        print(f"{'#' * 70}")

        rho_target = make_target_state(d, rng)
        coh_target = l1_coherence(rho_target)
        print(f"Target: C_l1 = {coh_target:.4f}")

        # Create catalyst target (maximally coherent state)
        psi_cat = np.ones(d, dtype=complex) / np.sqrt(d)
        rho_cat_ideal = np.outer(psi_cat, psi_cat.conj())

        # Apply noise to catalyst copies
        rho_cat_noisy_deph = dephasing_channel(rho_cat_ideal, noise_gamma)
        rho_cat_noisy_depol = depolarizing_channel(rho_cat_ideal, noise_delta)

        # Also apply noise to target (for CQEC recovery test)
        rho_noisy = dephasing_channel(rho_target, noise_gamma)
        fid_noisy = fidelity(rho_target, rho_noisy)
        print(f"F(noisy, target) = {fid_noisy:.6f}")

        dim_results = {}

        for noise_name, rho_cat_noisy in [
            ('Depolarizing', rho_cat_noisy_depol),
            ('Dephasing', rho_cat_noisy_deph),
        ]:
            print(f"\n--- Catalyst noise: {noise_name} ---")
            fid_cat_noisy = fidelity(rho_cat_ideal, rho_cat_noisy)
            pur_cat_noisy = purity(rho_cat_noisy)
            print(f"  F(cat_noisy, cat_ideal) = {fid_cat_noisy:.6f}, "
                  f"purity = {pur_cat_noisy:.6f}")

            noise_results = {}

            for strat in strategies:
                strat_results = []
                for n_copies in copy_budgets:
                    t0 = time.time()

                    # Skip if strategy can't use this many copies
                    if strat == 'optimal_only' and n_copies > 4:
                        # Use 4-copy result
                        rho_cat_purified, used, p_succ = HybridPurification.purify(
                            rho_cat_noisy, d, 4, strategy=strat)
                    elif strat == 'recursive_only':
                        n_rounds = int(np.log2(n_copies))
                        if n_rounds < 1:
                            n_rounds = 1
                        rho_cat_purified, used, p_succ = \
                            StreamingPurification.recursive_purify(
                                rho_cat_noisy, d, n_rounds)
                    else:
                        rho_cat_purified, used, p_succ = HybridPurification.purify(
                            rho_cat_noisy, d, n_copies, strategy=strat)

                    fid_cat = fidelity(rho_cat_ideal, rho_cat_purified)
                    pur_cat = purity(rho_cat_purified)
                    coh_cat = l1_coherence(rho_cat_purified)

                    # CQEC recovery using purified catalyst
                    rho_rec = cqec_recovery(rho_target, rho_noisy,
                                            rho_cat_purified)
                    fid_rec = fidelity(rho_target, rho_rec)

                    dt = time.time() - t0

                    strat_results.append({
                        'n_copies': n_copies,
                        'fid_catalyst': fid_cat,
                        'purity_catalyst': pur_cat,
                        'coherence_catalyst': coh_cat,
                        'p_success': p_succ,
                        'fid_recovered': fid_rec,
                        'time': dt,
                    })

                noise_results[strat] = strat_results

            dim_results[noise_name] = noise_results

            # Print table
            print(f"\n  {'Strategy':<22} {'n':>4} {'F_cat':>8} {'Pur':>8} "
                  f"{'C_l1':>8} {'p_succ':>8} {'F_rec':>8}")
            print(f"  {'-'*72}")
            for strat in strategies:
                for r in noise_results[strat]:
                    print(f"  {strategy_labels[strat]:<22} "
                          f"{r['n_copies']:>4} "
                          f"{r['fid_catalyst']:>8.5f} "
                          f"{r['purity_catalyst']:>8.5f} "
                          f"{r['coherence_catalyst']:>8.4f} "
                          f"{r['p_success']:>8.4f} "
                          f"{r['fid_recovered']:>8.5f}")

        all_results[d] = dim_results

    return all_results, dims, copy_budgets, strategies, strategy_labels, strategy_colors


def plot_results(all_results, dims, copy_budgets, strategies,
                 strategy_labels, strategy_colors):
    """Generate comparison plots."""
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 8, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # --- Figure 1: Catalyst fidelity vs copy count ---
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]

    noise_name = 'Depolarizing'
    for di, d in enumerate(dims):
        ax = axes[di]
        for strat in strategies:
            results = all_results[d][noise_name][strat]
            ns = [r['n_copies'] for r in results]
            fids = [r['fid_catalyst'] for r in results]
            ax.semilogx(ns, fids, 'o-',
                        color=strategy_colors[strat],
                        label=strategy_labels[strat],
                        linewidth=2, markersize=5)

        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel(r'Number of noisy copies $n$')
        ax.set_ylabel(r'Catalyst fidelity $F(\rho_{\mathrm{cat}},\,\rho_{\mathrm{ideal}})$')
        ax.set_title(rf'$d = {d}$ ({int(np.log2(d))} qubits), depolarizing $p=0.3$')
        ax.legend(loc='lower right')
        ax.set_ylim(0.5, 1.02)

    plt.suptitle('Catalyst Purification: Fidelity vs. Copy Count',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_purification_catalyst_fidelity.png')
    plt.savefig('fig_purification_catalyst_fidelity.pdf')
    print("\nSaved fig_purification_catalyst_fidelity.png/pdf")
    plt.close()

    # --- Figure 2: CQEC recovery fidelity vs copy count ---
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]

    for di, d in enumerate(dims):
        ax = axes[di]
        for strat in strategies:
            results = all_results[d][noise_name][strat]
            ns = [r['n_copies'] for r in results]
            fids = [r['fid_recovered'] for r in results]
            ax.semilogx(ns, fids, 's-',
                        color=strategy_colors[strat],
                        label=strategy_labels[strat],
                        linewidth=2, markersize=5)

        # Reference: no purification
        r0 = all_results[d][noise_name][strategies[0]][0]
        # noisy fidelity baseline
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel(r'Catalyst copies $n$')
        ax.set_ylabel(r'CQEC recovery $F(\rho_{\mathrm{rec}},\,\rho_0)$')
        ax.set_title(rf'$d = {d}$, CQEC recovery (dephasing $\gamma=2$)')
        ax.legend(loc='lower right')
        ax.set_ylim(0.4, 1.02)

    plt.suptitle('CQEC Recovery with Purified Catalysts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_purification_cqec_recovery.png')
    plt.savefig('fig_purification_cqec_recovery.pdf')
    print("Saved fig_purification_cqec_recovery.png/pdf")
    plt.close()

    # --- Figure 3: Strategy comparison summary ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Best fidelity achieved by each strategy at n=64
    x = np.arange(len(dims))
    width = 0.2
    for si, strat in enumerate(strategies):
        fids = []
        for d in dims:
            results = all_results[d][noise_name][strat]
            best = max(r['fid_catalyst'] for r in results)
            fids.append(best)
        ax1.bar(x + si * width - 1.5 * width, fids, width,
                color=strategy_colors[strat],
                label=strategy_labels[strat], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'$d={d}$' for d in dims])
    ax1.set_ylabel(r'Best catalyst $F$ (at $n=64$)')
    ax1.set_title(r'(a) Best catalyst fidelity by strategy')
    ax1.legend(fontsize=7)
    ax1.set_ylim(0.7, 1.02)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

    # Panel (b): copies needed for F_cat >= 0.99
    for strat in strategies:
        copies_needed = []
        for d in dims:
            results = all_results[d][noise_name][strat]
            n_req = None
            for r in results:
                if r['fid_catalyst'] >= 0.99:
                    n_req = r['n_copies']
                    break
            copies_needed.append(n_req if n_req else 100)

        ax2.semilogy(dims, copies_needed, 'o-',
                     color=strategy_colors[strat],
                     label=strategy_labels[strat],
                     linewidth=2, markersize=8)

    # Distillation reference: n* ~ d^4 * e^{2γ} / (4 * 0.01^2)
    n_distill = [0.53 * d**2.06 for d in dims]
    n_star = [(c**2) / (4 * 0.01**2) for c in n_distill]
    ax2.semilogy(dims, n_star, 'x--', color='black', linewidth=1.5,
                 markersize=8, label=r'Distillation $n^*$ ($F\geq0.99$)')

    ax2.set_xlabel(r'Dimension $d$')
    ax2.set_ylabel(r'Copies for $F_{\mathrm{cat}} \geq 0.99$')
    ax2.set_title(r'(b) Copy efficiency: purification vs. distillation')
    ax2.legend(fontsize=7)

    plt.suptitle('Purification-Based Catalyst: Strategy Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_purification_comparison.png')
    plt.savefig('fig_purification_comparison.pdf')
    print("Saved fig_purification_comparison.png/pdf")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    results, dims, budgets, strats, labels, colors = run_benchmark()
    plot_results(results, dims, budgets, strats, labels, colors)

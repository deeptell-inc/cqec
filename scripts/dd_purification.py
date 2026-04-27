#!/usr/bin/env python3
"""
dd_purification.py — Dynamical Decoupling + Twirling + Swap Test pipeline.

Pipeline:
  1. Dynamic decoupling (DD) reduces effective γ → γ_eff = γ/N_dd
  2. Clifford twirling converts residual dephasing → depolarizing
  3. Recursive swap test purifies the depolarized copies

DD models:
  - Spin echo (N_dd = 2): γ_eff ≈ γ/2
  - CPMG-N: γ_eff ≈ γ/N  (N refocusing pulses)
  - UDD-N: γ_eff ≈ γ · (π/N)^(N+1) for 1/f noise (much better)

We model DD as reducing the effective dephasing parameter.
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_algorithm_states import make_qkan, make_qdrift, make_cfqpe, make_regev

# ============================================================
# Core
# ============================================================

def fidelity(rho, sigma):
    e, v = np.linalg.eigh(rho)
    sr = v @ np.diag(np.sqrt(np.maximum(e, 0))) @ v.conj().T
    M = sr @ sigma @ sr
    e2 = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e2, 0))) ** 2)

def l1_coherence(rho):
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def dephasing_channel(rho, gamma):
    d = rho.shape[0]
    mask = np.exp(-gamma * np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]))
    return rho * mask

def depolarizing_channel(rho, p):
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d

def swap_test(rho, sigma, d):
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

def recursive_swap(rho_noisy, d, n_rounds):
    rho = rho_noisy.copy()
    for _ in range(n_rounds):
        rho, _ = swap_test(rho, rho, d)
    return rho

def twirl_analytical(rho_ideal, gamma_eff, d):
    """Analytical Clifford twirl: dephasing γ_eff → depolarizing p_eff."""
    e_g = np.exp(-gamma_eff)
    F_avg = e_g + (1 - e_g) * 2 / (d * (d + 1))
    p_eff = max(0, 1 - (d * F_avg - 1) / (d - 1))
    return depolarizing_channel(rho_ideal, p_eff), p_eff

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    d = rho_target.shape[0]
    rho_rec = rho_noisy.copy()
    pur = purity(rho_cat)
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_target[i, j]) > 1e-10 and np.abs(rho_cat[i, j]) > 1e-10:
                cc = np.abs(rho_cat[i, j])
                nc = np.abs(rho_noisy[i, j])
                if nc > 1e-15:
                    ph = np.angle(rho_target[i, j])
                    mt = np.abs(rho_target[i, j])
                    eff = 1.0 - np.exp(-cc * d * pur)
                    mr = nc + eff * (mt - nc)
                    rho_rec[i, j] = mr * np.exp(1j * ph)
                    rho_rec[j, i] = rho_rec[i, j].conj()
    e, v = np.linalg.eigh(rho_rec)
    e = np.maximum(e, 0)
    rho_rec = v @ np.diag(e) @ v.conj().T
    rho_rec /= np.trace(rho_rec)
    return rho_rec


# ============================================================
# DD models
# ============================================================

def dd_effective_gamma(gamma, n_dd, dd_type='cpmg'):
    """
    Effective dephasing after dynamical decoupling.

    CPMG-N: γ_eff ≈ γ / N (white noise, Gaussian decay)
    UDD-N:  γ_eff ≈ γ · (πe/(2N))^(2N) for 1/f noise (superpolynomial)
    Spin echo: CPMG with N=1 → γ_eff ≈ γ/2

    Conservative model (white noise): γ_eff = γ / (N+1)
    """
    if dd_type == 'cpmg':
        return gamma / (n_dd + 1)
    elif dd_type == 'udd':
        # UDD gives much better suppression for structured noise
        # γ_eff ~ γ · (π/(2N))^(2N) for ideal case
        # Use realistic model: γ_eff = γ · exp(-n_dd)
        return gamma * np.exp(-n_dd * 0.5)
    elif dd_type == 'none':
        return gamma
    else:
        raise ValueError(f"Unknown DD type: {dd_type}")


def make_cat(d):
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


# ============================================================
# Benchmark: Full pipeline
# ============================================================

def run():
    print("=" * 100)
    print("DD + Twirling + Swap Test Pipeline")
    print("=" * 100)

    algorithms = {
        'QKAN': make_qkan(),
        'qDRIFT': make_qdrift(seed=42),
        'CF-QPE': make_cfqpe(),
        'Regev': make_regev(),
    }

    gamma_base = 2.0
    max_rounds = {4: 6, 8: 6, 16: 5, 64: 3}

    # DD configurations: (label, n_dd, dd_type)
    dd_configs = [
        ('No DD', 0, 'none'),
        ('CPMG-2', 2, 'cpmg'),
        ('CPMG-4', 4, 'cpmg'),
        ('CPMG-8', 8, 'cpmg'),
        ('CPMG-16', 16, 'cpmg'),
    ]

    # Pipeline configs: (label, use_dd, use_twirl)
    pipelines = [
        ('Raw', False, False),
        ('DD only', True, False),
        ('Twirl only', False, True),
        ('DD + Twirl', True, True),
    ]

    all_results = {}

    for alg, (rho_target, d) in algorithms.items():
        print(f"\n{'#' * 80}")
        print(f"# {alg} (d={d})")
        print(f"{'#' * 80}")

        rho_cat_ideal = make_cat(d)
        rho_noisy_target = dephasing_channel(rho_target, gamma_base)
        mr = max_rounds[d]

        alg_res = {}

        # === Part 1: DD sweep (fixed n=8 copies, DD + Twirl pipeline) ===
        print(f"\n  --- DD sweep (n=8 copies, DD+Twirl pipeline) ---")
        print(f"  {'DD config':<12} {'γ_eff':>6} {'p_eff':>6} {'F_cat':>8} {'C_l1':>8} {'F_rec':>8}")
        print(f"  {'-' * 54}")

        dd_sweep = []
        for dd_label, n_dd, dd_type in dd_configs:
            gamma_eff = dd_effective_gamma(gamma_base, n_dd, dd_type)
            rho_cat_dd, p_eff = twirl_analytical(rho_cat_ideal, gamma_eff, d)
            rho_cat_purified = recursive_swap(rho_cat_dd, d, 3)  # n=8
            fc = fidelity(rho_cat_ideal, rho_cat_purified)
            cl = l1_coherence(rho_cat_purified)
            rho_rec = cqec_recovery(rho_target, rho_noisy_target, rho_cat_purified)
            fr = fidelity(rho_target, rho_rec)
            dd_sweep.append({
                'dd': dd_label, 'n_dd': n_dd, 'gamma_eff': gamma_eff,
                'p_eff': p_eff, 'fc': fc, 'cl': cl, 'fr': fr
            })
            print(f"  {dd_label:<12} {gamma_eff:>6.3f} {p_eff:>6.3f} {fc:>8.4f} {cl:>8.3f} {fr:>8.4f}")

        alg_res['dd_sweep'] = dd_sweep

        # === Part 2: Full pipeline comparison (CPMG-8, all copy counts) ===
        n_dd_chosen = 8
        gamma_eff = dd_effective_gamma(gamma_base, n_dd_chosen, 'cpmg')

        print(f"\n  --- Pipeline comparison (CPMG-{n_dd_chosen}, γ_eff={gamma_eff:.3f}) ---")
        print(f"  {'Pipeline':<16} {'n':>4} {'F_cat':>8} {'C_l1':>8} {'F_rec':>8}")
        print(f"  {'-' * 44}")

        pipeline_res = {}
        for pl_label, use_dd, use_twirl in pipelines:
            pres = []
            for n_r in range(1, mr + 1):
                nc = 2**n_r
                g = gamma_eff if use_dd else gamma_base

                if use_twirl:
                    rho_cat_input, p_e = twirl_analytical(rho_cat_ideal, g, d)
                else:
                    rho_cat_input = dephasing_channel(rho_cat_ideal, g)

                rho_cat_p = recursive_swap(rho_cat_input, d, n_r)
                fc = fidelity(rho_cat_ideal, rho_cat_p)
                cl = l1_coherence(rho_cat_p)
                rho_rec = cqec_recovery(rho_target, rho_noisy_target, rho_cat_p)
                fr = fidelity(rho_target, rho_rec)
                pres.append({'n': nc, 'fc': fc, 'cl': cl, 'fr': fr})
                print(f"  {pl_label:<16} {nc:>4} {fc:>8.4f} {cl:>8.3f} {fr:>8.4f}")
            print()
            pipeline_res[pl_label] = pres

        alg_res['pipeline'] = pipeline_res
        all_results[alg] = (d, alg_res)

    return all_results


def plot(all_results):
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'Arial',
        'mathtext.fontset': 'stix',
        'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    alg_names = list(all_results.keys())
    n_alg = len(alg_names)
    pipeline_colors = {
        'Raw': '#e74c3c',
        'DD only': '#f39c12',
        'Twirl only': '#9b59b6',
        'DD + Twirl': '#2ecc71',
    }

    # --- Fig 1: Pipeline comparison (F_cat vs copies) ---
    fig, axes = plt.subplots(2, n_alg, figsize=(5 * n_alg, 9))
    for ai, alg in enumerate(alg_names):
        d, ar = all_results[alg]
        for pl, pres in ar['pipeline'].items():
            ns = [r['n'] for r in pres]
            fc = [r['fc'] for r in pres]
            fr = [r['fr'] for r in pres]
            axes[0, ai].semilogx(ns, fc, 'o-', color=pipeline_colors[pl],
                                 lw=2, ms=5, label=pl)
            axes[1, ai].semilogx(ns, fr, 's-', color=pipeline_colors[pl],
                                 lw=2, ms=5, label=pl)
        axes[0, ai].set_title(f'{alg} ($d={d}$)')
        axes[0, ai].set_ylabel(r'$F_{\mathrm{cat}}$')
        axes[0, ai].set_xlabel(r'Copies $n$')
        axes[0, ai].legend(fontsize=6)
        axes[0, ai].axhline(y=0.95, color='gray', ls=':', alpha=0.4)
        axes[0, ai].axhline(y=0.99, color='gray', ls='--', alpha=0.3)
        axes[1, ai].set_ylabel(r'$F_{\mathrm{rec}}$')
        axes[1, ai].set_xlabel(r'Copies $n$')
        axes[1, ai].legend(fontsize=6)

    plt.suptitle(r'DD + Twirl + Swap Test Pipeline (CPMG-8, $\gamma=2$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_dd_pipeline.png')
    plt.savefig('fig_dd_pipeline.pdf')
    print("\nSaved fig_dd_pipeline.png/pdf")
    plt.close()

    # --- Fig 2: DD sweep (F_cat vs N_dd, fixed n=8) ---
    fig, axes = plt.subplots(1, n_alg, figsize=(5 * n_alg, 5))
    for ai, alg in enumerate(alg_names):
        d, ar = all_results[alg]
        dd = ar['dd_sweep']
        ndds = [r['n_dd'] for r in dd]
        fcs = [r['fc'] for r in dd]
        frs = [r['fr'] for r in dd]
        gammas = [r['gamma_eff'] for r in dd]

        ax = axes[ai]
        ax2 = ax.twinx()
        l1, = ax.plot(ndds, fcs, 'o-', color='#2ecc71', lw=2, ms=7, label=r'$F_{\mathrm{cat}}$')
        l2, = ax.plot(ndds, frs, 's-', color='#3498db', lw=2, ms=7, label=r'$F_{\mathrm{rec}}$')
        l3, = ax2.plot(ndds, gammas, 'D--', color='#e74c3c', lw=1.5, ms=5,
                       alpha=0.6, label=r'$\gamma_{\mathrm{eff}}$')

        ax.set_xlabel(r'DD pulses $N_{\mathrm{DD}}$')
        ax.set_ylabel(r'Fidelity')
        ax2.set_ylabel(r'$\gamma_{\mathrm{eff}}$', color='#e74c3c')
        ax.set_title(f'{alg} ($d={d}$, $n=8$)')
        ax.legend(handles=[l1, l2, l3], fontsize=7, loc='center right')
        ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
        ax.set_ylim(0, 1.05)

    plt.suptitle(r'Effect of Dynamical Decoupling (DD+Twirl+SwapTest, $n=8$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_dd_sweep.png')
    plt.savefig('fig_dd_sweep.pdf')
    print("Saved fig_dd_sweep.png/pdf")
    plt.close()

    # --- Fig 3: Summary bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_alg)
    width = 0.18
    for pi, (pl, color) in enumerate(pipeline_colors.items()):
        fcs = []
        for alg in alg_names:
            _, ar = all_results[alg]
            best = ar['pipeline'][pl][-1]  # max copies
            fcs.append(best['fc'])
        ax.bar(x + pi * width - 1.5 * width, fcs, width,
               color=color, label=pl, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}\n($d={all_results[a][0]}$)' for a in alg_names])
    ax.set_ylabel(r'$F_{\mathrm{cat}}$ (max copies)')
    ax.set_title(r'Catalyst Fidelity: Pipeline Comparison ($\gamma=2$, CPMG-8)')
    ax.legend(fontsize=9)
    ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('fig_dd_summary.png')
    plt.savefig('fig_dd_summary.pdf')
    print("Saved fig_dd_summary.png/pdf")
    plt.close()


if __name__ == '__main__':
    results = run()
    plot(results)

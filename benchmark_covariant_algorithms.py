#!/usr/bin/env python3
"""
benchmark_covariant_algorithms.py — Covariant purification on paper benchmarks.

Tests covariant recursive swap test catalyst preparation + CQEC recovery
on: QKAN (d=4), qDRIFT (d=8), CF-QPE (d=16), Regev (d=64).
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Utilities (inlined for self-containedness)
# ============================================================

def fidelity(rho, sigma):
    sr = _msqrt(rho)
    M = sr @ sigma @ sr
    e = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e, 0))) ** 2)

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

def combined_noise(rho):
    out = dephasing_channel(rho, 1.0)
    out = depolarizing_channel(out, 0.15)
    # Amplitude damping (per-qubit, simplified)
    d = rho.shape[0]
    n_q = int(np.log2(d))
    gamma_ad = 0.1
    for q in range(n_q):
        I_pre = np.eye(2**q, dtype=complex)
        I_post = np.eye(2**(n_q - q - 1), dtype=complex)
        E0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma_ad)]], dtype=complex)
        E1_1q = np.array([[0, np.sqrt(gamma_ad)], [0, 0]], dtype=complex)
        E0 = np.kron(np.kron(I_pre, E0_1q), I_post)
        E1 = np.kron(np.kron(I_pre, E1_1q), I_post)
        out = E0 @ out @ E0.conj().T + E1 @ out @ E1.conj().T
    return out


# ============================================================
# Swap test implementations
# ============================================================

def standard_swap_purify(rho, sigma, d):
    """Standard swap test on d×d states."""
    d2 = d * d
    rs = np.kron(rho, sigma)
    # Build SWAP
    SWAP = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            SWAP[j * d + i, i * d + j] = 1.0
    Pi = (np.eye(d2) + SWAP) / 2.0
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


def covariant_swap_purify(rho, sigma, d):
    """
    Covariant swap test: project within each energy sector's
    symmetric subspace.
    """
    d2 = d * d
    rs = np.kron(rho, sigma)

    # Build covariant unitary (EC rotations within energy sectors)
    U = np.eye(d2, dtype=complex)
    theta = np.pi / 4  # half-swap angle

    for E_total in range(2 * (d - 1) + 1):
        sector = []
        for i in range(d):
            j = E_total - i
            if 0 <= j < d:
                sector.append((i, j))
        if len(sector) < 2:
            continue
        for k in range(len(sector) - 1):
            i1, j1 = sector[k]
            i2, j2 = sector[k + 1]
            idx1 = i1 * d + j1
            idx2 = i2 * d + j2
            c, s = np.cos(theta), np.sin(theta)
            U[idx1, idx1] = c
            U[idx2, idx2] = c
            U[idx1, idx2] = -1j * s
            U[idx2, idx1] = -1j * s

    state = U @ rs @ U.conj().T

    # Project onto symmetric sector within each energy block
    Pi = np.zeros((d2, d2), dtype=complex)
    for E_total in range(2 * (d - 1) + 1):
        sector = []
        for i in range(d):
            j = E_total - i
            if 0 <= j < d:
                sector.append(i * d + j)
        if len(sector) == 1:
            Pi[sector[0], sector[0]] = 1.0
        else:
            n_s = len(sector)
            for a in range(n_s):
                for b in range(n_s):
                    Pi[sector[a], sector[b]] += 1.0 / n_s

    proj = Pi @ state @ Pi
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


def recursive_purify(rho_noisy, d, n_rounds, method='standard'):
    """Recursive purification for given number of rounds."""
    rho = rho_noisy.copy()
    p_total = 1.0
    fn = standard_swap_purify if method == 'standard' else covariant_swap_purify
    for _ in range(n_rounds):
        rho, p = fn(rho, rho, d)
        p_total *= p
    return rho, 2**n_rounds, p_total


# ============================================================
# CQEC recovery
# ============================================================

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    d = rho_target.shape[0]
    tol = 1e-10
    rho_rec = rho_noisy.copy()
    pur = purity(rho_cat)
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_target[i, j]) > tol and np.abs(rho_cat[i, j]) > tol:
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
# Algorithm target states — physically correct implementations
# ============================================================

from make_algorithm_states import make_qkan, make_qdrift, make_cfqpe, make_regev


# ============================================================
# Main benchmark
# ============================================================

def run():
    rng = np.random.default_rng(42)

    algorithms = {
        'QKAN': make_qkan(),
        'qDRIFT': make_qdrift(seed=42),
        'CF-QPE': make_cfqpe(),
    }

    # Regev d=64: swap test needs 4096×4096 matrices (~256MB)
    try:
        algorithms['Regev'] = make_regev()
    except MemoryError:
        print("Regev d=64 skipped (memory)")

    noise_configs = {
        'Dephasing': lambda rho: dephasing_channel(rho, 2.0),
        'Depolarizing': lambda rho: depolarizing_channel(rho, 0.3),
        'Combined': combined_noise,
    }

    # Rounds config per dimension
    max_rounds = {4: 6, 8: 6, 16: 5, 64: 3}

    methods = ['standard', 'covariant']
    method_labels = {'standard': 'Standard', 'covariant': 'Covariant'}
    method_colors = {'standard': '#e74c3c', 'covariant': '#3498db'}

    all_results = {}

    print("=" * 100)
    print("Covariant Purification: Paper Algorithm Benchmarks")
    print("=" * 100)

    for alg_name, (rho_target, d) in algorithms.items():
        print(f"\n{'#' * 80}")
        print(f"# {alg_name} (d={d}, {int(np.log2(d))} qubits)")
        print(f"# C_l1(target) = {l1_coherence(rho_target):.4f}")
        print(f"{'#' * 80}")

        # Catalyst: maximally coherent state
        psi_cat = np.ones(d, dtype=complex) / np.sqrt(d)
        rho_cat_ideal = np.outer(psi_cat, psi_cat.conj())

        mr = max_rounds.get(d, 4)
        round_list = list(range(1, mr + 1))
        copy_list = [2**r for r in round_list]

        alg_results = {}

        for noise_name, noise_fn in noise_configs.items():
            rho_noisy = noise_fn(rho_target)
            fid_noisy = fidelity(rho_target, rho_noisy)
            rho_cat_noisy = noise_fn(rho_cat_ideal)
            fid_cat0 = fidelity(rho_cat_ideal, rho_cat_noisy)

            print(f"\n  {noise_name}: F_noisy={fid_noisy:.4f}, "
                  f"F_cat(noisy)={fid_cat0:.4f}")

            noise_res = {}
            for method in methods:
                mres = []
                for n_rounds in round_list:
                    t0 = time.time()
                    rho_cat_p, n_copies, p_s = recursive_purify(
                        rho_cat_noisy, d, n_rounds, method)
                    fid_cat = fidelity(rho_cat_ideal, rho_cat_p)
                    coh_cat = l1_coherence(rho_cat_p)

                    # CQEC recovery
                    rho_rec = cqec_recovery(rho_target, rho_noisy, rho_cat_p)
                    fid_rec = fidelity(rho_target, rho_rec)
                    dt = time.time() - t0

                    mres.append({
                        'n': n_copies, 'rounds': n_rounds,
                        'fid_cat': fid_cat, 'coh': coh_cat,
                        'p': p_s, 'fid_rec': fid_rec, 'time': dt,
                    })

                    print(f"    {method_labels[method]:>12} n={n_copies:>4}: "
                          f"F_cat={fid_cat:.4f} C_l1={coh_cat:.3f} "
                          f"F_rec={fid_rec:.4f} ({dt:.1f}s)")

                noise_res[method] = mres
            alg_results[noise_name] = noise_res

        all_results[alg_name] = (d, alg_results)

    return all_results


def plot(all_results):
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 12,
        'legend.fontsize': 8, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    alg_names = list(all_results.keys())
    n_alg = len(alg_names)

    method_colors = {'standard': '#e74c3c', 'covariant': '#3498db'}
    method_labels = {'standard': 'Standard recursive',
                     'covariant': 'Covariant recursive'}

    # --- Figure: 2 rows (cat fidelity, CQEC recovery) × n_alg columns ---
    # Dephasing only (the interesting case)
    fig, axes = plt.subplots(2, n_alg, figsize=(5 * n_alg, 9))
    if n_alg == 1:
        axes = axes[:, np.newaxis]

    noise = 'Dephasing'
    for ai, alg in enumerate(alg_names):
        d, ar = all_results[alg]

        for method in ['standard', 'covariant']:
            res = ar[noise][method]
            ns = [r['n'] for r in res]
            fc = [r['fid_cat'] for r in res]
            fr = [r['fid_rec'] for r in res]

            axes[0, ai].semilogx(ns, fc, 'o-', color=method_colors[method],
                                 label=method_labels[method], linewidth=2)
            axes[1, ai].semilogx(ns, fr, 's-', color=method_colors[method],
                                 label=method_labels[method], linewidth=2)

        axes[0, ai].set_title(f'{alg} ($d={d}$)')
        axes[0, ai].set_ylabel(r'$F_{\mathrm{cat}}$')
        axes[0, ai].set_xlabel(r'Copies $n$')
        axes[0, ai].legend(fontsize=7)
        axes[0, ai].axhline(y=1.0, color='gray', ls=':', alpha=0.3)

        axes[1, ai].set_ylabel(r'$F_{\mathrm{rec}}$ (CQEC)')
        axes[1, ai].set_xlabel(r'Copies $n$')
        axes[1, ai].legend(fontsize=7)
        axes[1, ai].axhline(y=1.0, color='gray', ls=':', alpha=0.3)

    plt.suptitle(r'Covariant vs. Standard Purification (dephasing $\gamma=2$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_covariant_algorithms.png')
    plt.savefig('fig_covariant_algorithms.pdf')
    print("\nSaved fig_covariant_algorithms.png/pdf")
    plt.close()

    # --- Summary table figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    # Build table data
    headers = ['Algorithm', '$d$', 'Noise', 'Method', '$n$',
               '$F_{cat}$', '$C_{\\ell_1}$', '$F_{rec}$']
    rows = []
    for alg in alg_names:
        d, ar = all_results[alg]
        for noise in ['Dephasing', 'Depolarizing']:
            for method in ['standard', 'covariant']:
                res = ar[noise][method]
                best = res[-1]  # highest copy count
                rows.append([
                    alg, str(d), noise,
                    'Std' if method == 'standard' else 'Cov',
                    str(best['n']),
                    f"{best['fid_cat']:.4f}",
                    f"{best['coh']:.2f}",
                    f"{best['fid_rec']:.4f}",
                ])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Color covariant rows
    for i, row in enumerate(rows):
        if row[3] == 'Cov':
            for j in range(len(headers)):
                table[i + 1, j].set_facecolor('#d4e6f1')

    plt.title('Summary: Covariant vs Standard at Maximum Copies',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('fig_covariant_summary_table.png')
    plt.savefig('fig_covariant_summary_table.pdf')
    print("Saved fig_covariant_summary_table.png/pdf")
    plt.close()


if __name__ == '__main__':
    results = run()
    plot(results)

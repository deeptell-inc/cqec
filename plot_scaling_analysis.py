#!/usr/bin/env python3
"""
plot_scaling_analysis.py — Generate fidelity vs error rate and fidelity vs qubit count
plots for paper_catalyst.tex.

Two figures:
  fig_fidelity_vs_error.pdf: F_cat vs γ (dephasing) and p (depolarizing) for
    standard and covariant at fixed n=8 copies, multiple dimensions
  fig_fidelity_vs_qubits.pdf: F_cat vs d (dimension / qubit count) for
    standard and covariant at fixed noise, multiple copy counts
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- Utilities ---
def fidelity(rho, sigma):
    sr = _msqrt(rho)
    M = sr @ sigma @ sr
    e = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e, 0))) ** 2)

def _msqrt(A):
    e, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.maximum(e, 0))) @ v.conj().T

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


# --- Swap tests ---
def standard_swap(rho, sigma, d):
    d2 = d * d
    rs = np.kron(rho, sigma)
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

def covariant_swap(rho, sigma, d):
    d2 = d * d
    rs = np.kron(rho, sigma)
    theta = np.pi / 4
    U = np.eye(d2, dtype=complex)
    for E in range(2 * (d - 1) + 1):
        sector = []
        for i in range(d):
            j = E - i
            if 0 <= j < d:
                sector.append((i, j))
        if len(sector) < 2:
            continue
        for k in range(len(sector) - 1):
            i1, j1 = sector[k]
            i2, j2 = sector[k + 1]
            idx1, idx2 = i1 * d + j1, i2 * d + j2
            c, s = np.cos(theta), np.sin(theta)
            U[idx1, idx1] = c; U[idx2, idx2] = c
            U[idx1, idx2] = -1j * s; U[idx2, idx1] = -1j * s

    state = U @ rs @ U.conj().T
    Pi = np.zeros((d2, d2), dtype=complex)
    for E in range(2 * (d - 1) + 1):
        sector = []
        for i in range(d):
            j = E - i
            if 0 <= j < d:
                sector.append(i * d + j)
        n_s = len(sector)
        if n_s == 1:
            Pi[sector[0], sector[0]] = 1.0
        else:
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
    rho = rho_noisy.copy()
    fn = standard_swap if method == 'standard' else covariant_swap
    for _ in range(n_rounds):
        rho, _ = fn(rho, rho, d)
    return rho


# --- Main ---
plt.rcParams.update({
    'font.size': 11, 'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ============================================================
# Figure 1: F_cat vs error strength (γ and p)
# ============================================================
print("=== Figure 1: Fidelity vs error rate ===")

dims_err = [4, 8, 16]
n_rounds = 3  # 8 copies

gammas = np.linspace(0.2, 4.0, 15)
ps = np.linspace(0.05, 0.6, 12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): dephasing
ax = axes[0]
for d in dims_err:
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    rho_ideal = np.outer(psi, psi.conj())
    fids_std, fids_cov, fids_noisy = [], [], []
    for gamma in gammas:
        rho_n = dephasing_channel(rho_ideal, gamma)
        fids_noisy.append(fidelity(rho_ideal, rho_n))
        rho_s = recursive_purify(rho_n, d, n_rounds, 'standard')
        fids_std.append(fidelity(rho_ideal, rho_s))
        rho_c = recursive_purify(rho_n, d, n_rounds, 'covariant')
        fids_cov.append(fidelity(rho_ideal, rho_c))
        print(f"  d={d}, γ={gamma:.1f}: std={fids_std[-1]:.3f}, cov={fids_cov[-1]:.3f}")

    color = {'4': '#2ecc71', '8': '#3498db', '16': '#e74c3c'}[str(d)]
    ax.plot(gammas, fids_cov, 'o-', color=color, linewidth=2, markersize=4,
            label=rf'Covariant $d={d}$')
    ax.plot(gammas, fids_std, 's--', color=color, linewidth=1.2, markersize=3,
            alpha=0.6, label=rf'Standard $d={d}$')

ax.set_xlabel(r'Dephasing strength $\gamma$')
ax.set_ylabel(r'Catalyst fidelity $F_{\mathrm{cat}}$')
ax.set_title(r'(a) $F_{\mathrm{cat}}$ vs. dephasing $\gamma$ ($n = 8$ copies)')
ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.set_ylim(0.1, 1.02)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
ax.text(0.3, 0.96, r'$F = 0.95$', fontsize=8, color='gray')

# Panel (b): depolarizing
ax = axes[1]
for d in dims_err:
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    rho_ideal = np.outer(psi, psi.conj())
    fids_std, fids_cov = [], []
    for p in ps:
        rho_n = depolarizing_channel(rho_ideal, p)
        rho_s = recursive_purify(rho_n, d, n_rounds, 'standard')
        fids_std.append(fidelity(rho_ideal, rho_s))
        rho_c = recursive_purify(rho_n, d, n_rounds, 'covariant')
        fids_cov.append(fidelity(rho_ideal, rho_c))

    color = {'4': '#2ecc71', '8': '#3498db', '16': '#e74c3c'}[str(d)]
    ax.plot(ps, fids_cov, 'o-', color=color, linewidth=2, markersize=4,
            label=rf'Covariant $d={d}$')
    ax.plot(ps, fids_std, 's--', color=color, linewidth=1.2, markersize=3,
            alpha=0.6, label=rf'Standard $d={d}$')

ax.set_xlabel(r'Depolarizing probability $p$')
ax.set_ylabel(r'Catalyst fidelity $F_{\mathrm{cat}}$')
ax.set_title(r'(b) $F_{\mathrm{cat}}$ vs. depolarizing $p$ ($n = 8$ copies)')
ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.set_ylim(0.5, 1.02)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

plt.suptitle(r'Catalyst Fidelity vs. Error Strength', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fidelity_vs_error.png')
plt.savefig('fig_fidelity_vs_error.pdf')
print("Saved fig_fidelity_vs_error.png/pdf")
plt.close()


# ============================================================
# Figure 2: F_cat vs dimension (qubit count)
# ============================================================
print("\n=== Figure 2: Fidelity vs qubit count ===")

dims_q = [2, 4, 8, 16, 32]  # d=32 may be slow
n_rounds_list = [1, 2, 3, 4]  # 2, 4, 8, 16 copies
gamma_fixed = 2.0
p_fixed = 0.3

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): dephasing
ax = axes[0]
for n_r in n_rounds_list:
    fids_std, fids_cov = [], []
    valid_dims = []
    for d in dims_q:
        print(f"  d={d}, rounds={n_r}...", end=' ', flush=True)
        t0 = time.time()
        psi = np.ones(d, dtype=complex) / np.sqrt(d)
        rho_ideal = np.outer(psi, psi.conj())
        rho_n = dephasing_channel(rho_ideal, gamma_fixed)
        rho_s = recursive_purify(rho_n, d, n_r, 'standard')
        fs = fidelity(rho_ideal, rho_s)
        rho_c = recursive_purify(rho_n, d, n_r, 'covariant')
        fc = fidelity(rho_ideal, rho_c)
        fids_std.append(fs)
        fids_cov.append(fc)
        valid_dims.append(d)
        print(f"std={fs:.3f}, cov={fc:.3f} ({time.time()-t0:.1f}s)")

    n_copies = 2**n_r
    qubits = [int(np.log2(d)) for d in valid_dims]
    ax.plot(qubits, fids_cov, 'o-', linewidth=2, markersize=6,
            label=rf'Covariant $n={n_copies}$')
    ax.plot(qubits, fids_std, 's--', linewidth=1, markersize=4, alpha=0.5,
            label=rf'Standard $n={n_copies}$')

ax.set_xlabel(r'Number of qubits $n_q = \log_2 d$')
ax.set_ylabel(r'Catalyst fidelity $F_{\mathrm{cat}}$')
ax.set_title(r'(a) Dephasing $\gamma = 2.0$')
ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.set_ylim(0.0, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
ax.set_xticks([1, 2, 3, 4, 5])

# Panel (b): depolarizing
ax = axes[1]
for n_r in n_rounds_list:
    fids_std, fids_cov = [], []
    for d in dims_q:
        psi = np.ones(d, dtype=complex) / np.sqrt(d)
        rho_ideal = np.outer(psi, psi.conj())
        rho_n = depolarizing_channel(rho_ideal, p_fixed)
        rho_s = recursive_purify(rho_n, d, n_r, 'standard')
        fs = fidelity(rho_ideal, rho_s)
        rho_c = recursive_purify(rho_n, d, n_r, 'covariant')
        fc = fidelity(rho_ideal, rho_c)
        fids_std.append(fs)
        fids_cov.append(fc)

    n_copies = 2**n_r
    qubits = [int(np.log2(d)) for d in dims_q]
    ax.plot(qubits, fids_cov, 'o-', linewidth=2, markersize=6,
            label=rf'Covariant $n={n_copies}$')
    ax.plot(qubits, fids_std, 's--', linewidth=1, markersize=4, alpha=0.5,
            label=rf'Standard $n={n_copies}$')

ax.set_xlabel(r'Number of qubits $n_q = \log_2 d$')
ax.set_ylabel(r'Catalyst fidelity $F_{\mathrm{cat}}$')
ax.set_title(r'(b) Depolarizing $p = 0.3$')
ax.legend(loc='lower left', fontsize=7, ncol=2)
ax.set_ylim(0.5, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
ax.set_xticks([1, 2, 3, 4, 5])

plt.suptitle(r'Catalyst Fidelity vs. System Size', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fidelity_vs_qubits.png')
plt.savefig('fig_fidelity_vs_qubits.pdf')
print("Saved fig_fidelity_vs_qubits.png/pdf")
plt.close()

print("\nDone.")

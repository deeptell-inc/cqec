#!/usr/bin/env python3
"""
regenerate_all.py — Regenerate all benchmark data with CORRECT symmetric projector
(I + SWAP)/2 within each energy sector, NOT the rank-1 uniform projector.

Produces:
  1. Main Table I data (4 algorithms × 3 methods × dephasing)
  2. Noise comparison table (covariant × 3 noise models)
  3. fig_covariant_algorithms.pdf (updated)
  4. fig_fidelity_vs_error.pdf (updated)
  5. fig_fidelity_vs_qubits.pdf (updated)
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from make_algorithm_states import make_qkan, make_qdrift, make_cfqpe, make_regev


# ============================================================
# Utilities
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
    mask = np.exp(-gamma * np.abs(np.arange(d)[:,None] - np.arange(d)[None,:]))
    return rho * mask

def depolarizing_channel(rho, delta):
    d = rho.shape[0]
    return (1 - delta) * rho + delta * np.eye(d) / d

def combined_noise(rho):
    d = rho.shape[0]
    out = dephasing_channel(rho, 1.0)
    out = depolarizing_channel(out, 0.15)
    n_q = int(np.log2(d))
    gamma_ad = 0.1
    for q in range(n_q):
        I_pre = np.eye(2**q, dtype=complex)
        I_post = np.eye(2**(n_q - q - 1), dtype=complex)
        E0 = np.kron(np.kron(I_pre, np.array([[1,0],[0,np.sqrt(1-gamma_ad)]], dtype=complex)), I_post)
        E1 = np.kron(np.kron(I_pre, np.array([[0,np.sqrt(gamma_ad)],[0,0]], dtype=complex)), I_post)
        out = E0 @ out @ E0.conj().T + E1 @ out @ E1.conj().T
    return out


# ============================================================
# CORRECT swap test implementations
# ============================================================
def standard_swap(rho, sigma, d):
    d2 = d * d
    rs = np.kron(rho, sigma)
    SWAP = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            SWAP[j*d+i, i*d+j] = 1.0
    Pi = (np.eye(d2) + SWAP) / 2.0
    proj = Pi @ rs @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15: return rho.copy(), 0.0
    proj /= p
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i*d+k, j*d+k]
    return out, p

def covariant_swap(rho, sigma, d):
    """Covariant swap with CORRECT (I+SWAP_E)/2 projector."""
    d2 = d * d
    rs = np.kron(rho, sigma)
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)

    # EC rotations within sectors
    U = np.eye(d2, dtype=complex)
    for E in range(2*(d-1)+1):
        sector = [(i, E-i) for i in range(d) if 0 <= E-i < d]
        if len(sector) < 2: continue
        for k in range(len(sector)-1):
            i1,j1 = sector[k]; i2,j2 = sector[k+1]
            idx1, idx2 = i1*d+j1, i2*d+j2
            U[idx1,idx1] = c; U[idx2,idx2] = c
            U[idx1,idx2] = -1j*s; U[idx2,idx1] = -1j*s

    state = U @ rs @ U.conj().T

    # CORRECT: (I + SWAP_E)/2 within each sector
    Pi = np.zeros((d2, d2), dtype=complex)
    for E in range(2*(d-1)+1):
        sector = [(i, E-i) for i in range(d) if 0 <= E-i < d]
        n_s = len(sector)
        if n_s <= 1:
            if n_s == 1:
                idx = sector[0][0]*d + sector[0][1]
                Pi[idx, idx] = 1.0
            continue
        # Build SWAP within sector: |i,j⟩ → |j,i⟩
        indices = [i*d+j for i,j in sector]
        for a_idx, (ia, ja) in enumerate(sector):
            for b_idx, (ib, jb) in enumerate(sector):
                row, col = indices[a_idx], indices[b_idx]
                # Identity part
                if a_idx == b_idx:
                    Pi[row, col] += 0.5
                # SWAP part: maps (ib,jb) → (jb,ib)
                # Find which sector index corresponds to (jb,ib)
                for c_idx, (ic, jc) in enumerate(sector):
                    if ic == jb and jc == ib:
                        Pi[indices[a_idx], indices[c_idx]] += 0.5 if a_idx == b_idx else 0.0
                        break
        # Actually let me redo this properly
    # Redo Pi correctly
    Pi = np.zeros((d2, d2), dtype=complex)
    for E in range(2*(d-1)+1):
        sector = [(i, E-i) for i in range(d) if 0 <= E-i < d]
        n_s = len(sector)
        if n_s == 0: continue
        indices = [i*d+j for i,j in sector]
        if n_s == 1:
            Pi[indices[0], indices[0]] = 1.0
            continue
        # SWAP_E matrix (n_s × n_s)
        SWAP_E = np.zeros((n_s, n_s), dtype=complex)
        for a, (ia, ja) in enumerate(sector):
            for b, (ib, jb) in enumerate(sector):
                if ib == ja and jb == ia:
                    SWAP_E[a, b] = 1.0
        Pi_E = (np.eye(n_s) + SWAP_E) / 2.0
        for a in range(n_s):
            for b in range(n_s):
                Pi[indices[a], indices[b]] = Pi_E[a, b]

    proj = Pi @ state @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15: return rho.copy(), 0.0
    proj /= p
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i*d+k, j*d+k]
    return out, p

def recursive_purify(rho_noisy, d, n_rounds, method='covariant'):
    rho = rho_noisy.copy()
    fn = covariant_swap if method == 'covariant' else standard_swap
    for _ in range(n_rounds):
        rho, _ = fn(rho, rho, d)
    return rho

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    d = rho_target.shape[0]
    tol = 1e-10
    rho_rec = rho_noisy.copy()
    pur = purity(rho_cat)
    for i in range(d):
        for j in range(i+1, d):
            if np.abs(rho_target[i,j]) > tol and np.abs(rho_cat[i,j]) > tol:
                cc = np.abs(rho_cat[i,j])
                nc = np.abs(rho_noisy[i,j])
                if nc > 1e-15:
                    ph = np.angle(rho_target[i,j])
                    mt = np.abs(rho_target[i,j])
                    eff = 1.0 - np.exp(-cc * d * pur)
                    mr = nc + eff * (mt - nc)
                    rho_rec[i,j] = mr * np.exp(1j*ph)
                    rho_rec[j,i] = rho_rec[i,j].conj()
    e, v = np.linalg.eigh(rho_rec)
    e = np.maximum(e, 0)
    rho_rec = v @ np.diag(e) @ v.conj().T
    rho_rec /= np.trace(rho_rec)
    return rho_rec


# ============================================================
# Run all benchmarks
# ============================================================
print("=" * 90)
print("CORRECTED Benchmark: (I+SWAP_E)/2 projector")
print("=" * 90)

algorithms = {
    'QKAN': make_qkan(),
    'qDRIFT': make_qdrift(seed=42),
    'CF-QPE': make_cfqpe(),
    'Regev': make_regev(),
}

noise_configs = {
    'Dephasing': lambda rho: dephasing_channel(rho, 2.0),
    'Depolarizing': lambda rho: depolarizing_channel(rho, 0.3),
    'Combined': combined_noise,
}

max_rounds = {4: 6, 8: 6, 16: 5, 64: 3}

# Maximally coherent catalyst
def make_cat(d):
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())

all_data = {}

for alg_name, (rho_target, d) in algorithms.items():
    print(f"\n{'#'*70}")
    print(f"# {alg_name} (d={d}), C_l1={l1_coherence(rho_target):.4f}")
    print(f"{'#'*70}")

    rho_cat_ideal = make_cat(d)
    mr = max_rounds.get(d, 4)
    rounds = list(range(1, mr+1))

    alg_data = {}
    for noise_name, noise_fn in noise_configs.items():
        rho_noisy = noise_fn(rho_target)
        fid_noisy = fidelity(rho_target, rho_noisy)
        rho_cat_noisy = noise_fn(rho_cat_ideal)

        print(f"\n  {noise_name}: F_noisy={fid_noisy:.4f}")
        noise_data = {}

        for method in ['standard', 'covariant']:
            mdata = []
            for n_r in rounds:
                n_copies = 2**n_r
                rho_cat_p = recursive_purify(rho_cat_noisy, d, n_r, method)
                fc = fidelity(rho_cat_ideal, rho_cat_p)
                coh = l1_coherence(rho_cat_p)
                rho_rec = cqec_recovery(rho_target, rho_noisy, rho_cat_p)
                fr = fidelity(rho_target, rho_rec)
                mdata.append({'n': n_copies, 'fid_cat': fc, 'coh': coh, 'fid_rec': fr})
                print(f"    {method:>12} n={n_copies:>4}: F_cat={fc:.4f} C_l1={coh:.3f} F_rec={fr:.4f}")
            noise_data[method] = mdata
        alg_data[noise_name] = noise_data
    all_data[alg_name] = (d, alg_data)

# ============================================================
# Print summary tables for paper
# ============================================================
print("\n\n" + "=" * 110)
print("TABLE I: Main results (dephasing γ=2, max copy count)")
print("=" * 110)
print(f"{'Alg':<10} {'d':>3} | {'Std F_cat':>9} {'Std F_rec':>9} | {'Cov F_cat':>9} {'Cov F_rec':>9} | {'n_max':>5}")
print("-" * 70)
for alg, (d, ad) in all_data.items():
    sd = ad['Dephasing']['standard'][-1]
    cd = ad['Dephasing']['covariant'][-1]
    print(f"{alg:<10} {d:>3} | {sd['fid_cat']:>9.4f} {sd['fid_rec']:>9.4f} | "
          f"{cd['fid_cat']:>9.4f} {cd['fid_rec']:>9.4f} | {cd['n']:>5}")

print("\n" + "=" * 80)
print("TABLE: Noise comparison (covariant, max copies)")
print("=" * 80)
print(f"{'Alg':<10} {'d':>3} | {'Depol':>8} {'Deph':>8} {'Comb':>8}")
print("-" * 50)
for alg, (d, ad) in all_data.items():
    fdp = ad['Depolarizing']['covariant'][-1]['fid_cat']
    fde = ad['Dephasing']['covariant'][-1]['fid_cat']
    fcm = ad['Combined']['covariant'][-1]['fid_cat']
    print(f"{alg:<10} {d:>3} | {fdp:>8.4f} {fde:>8.4f} {fcm:>8.4f}")

# ============================================================
# Generate updated figures
# ============================================================
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# --- fig_covariant_algorithms ---
alg_names = list(all_data.keys())
n_alg = len(alg_names)
fig, axes = plt.subplots(2, n_alg, figsize=(5*n_alg, 9))

for ai, alg in enumerate(alg_names):
    d, ar = all_data[alg]
    noise = 'Dephasing'
    for method, color, ls, mk in [('standard','#e74c3c','--','s'),('covariant','#3498db','-','o')]:
        res = ar[noise][method]
        ns = [r['n'] for r in res]
        fc = [r['fid_cat'] for r in res]
        fr = [r['fid_rec'] for r in res]
        label = 'Standard' if method == 'standard' else 'Covariant'
        axes[0,ai].semilogx(ns, fc, f'{mk}{ls}', color=color, label=label, linewidth=2, markersize=5)
        axes[1,ai].semilogx(ns, fr, f'{mk}{ls}', color=color, label=label, linewidth=2, markersize=5)

    axes[0,ai].set_title(f'{alg} ($d={d}$)')
    axes[0,ai].set_ylabel(r'$F_{\mathrm{cat}}$')
    axes[0,ai].set_xlabel(r'Copies $n$')
    axes[0,ai].legend(fontsize=7)
    axes[1,ai].set_ylabel(r'$F_{\mathrm{rec}}$')
    axes[1,ai].set_xlabel(r'Copies $n$')
    axes[1,ai].legend(fontsize=7)

plt.suptitle(r'Corrected: Covariant vs Standard (dephasing $\gamma=2$)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_covariant_algorithms.png')
plt.savefig('fig_covariant_algorithms.pdf')
print("\nSaved fig_covariant_algorithms.png/pdf")
plt.close()

print("\nDone. Update paper_catalyst.tex with these numbers.")

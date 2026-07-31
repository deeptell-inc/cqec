#!/usr/bin/env python3
"""
plot_dd_scaling.py — Three scaling figures comparing Raw vs DD+Twirl pipeline.

  fig_dd_vs_error.pdf:  F_cat vs γ (fixed n=8, ±DD+Twirl)
  fig_dd_vs_qubits.pdf: F_cat vs n_q (fixed γ=2, ±DD+Twirl)
  fig_dd_vs_copies.pdf: F_cat vs n  (fixed γ=2, ±DD+Twirl)
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fidelity(rho, sigma):
    e,v = np.linalg.eigh(rho)
    sr = v @ np.diag(np.sqrt(np.maximum(e,0))) @ v.conj().T
    M = sr @ sigma @ sr
    e2 = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e2,0)))**2)

def l1_coherence(rho):
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))

def dephasing(rho, gamma):
    d = rho.shape[0]
    mask = np.exp(-gamma * np.abs(np.arange(d)[:,None] - np.arange(d)[None,:]))
    return rho * mask

def depolarizing(rho, p):
    d = rho.shape[0]
    return (1-p)*rho + p*np.eye(d)/d

def swap_test(rho, sigma, d):
    d2 = d*d; rs = np.kron(rho,sigma)
    S = np.zeros((d2,d2),dtype=complex)
    for i in range(d):
        for j in range(d): S[j*d+i,i*d+j]=1.0
    Pi = (np.eye(d2)+S)/2.0
    proj = Pi@rs@Pi; p = float(np.real(np.trace(proj)))
    if p<1e-15: return rho.copy()
    proj /= p
    out = np.zeros((d,d),dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d): out[i,j] += proj[i*d+k,j*d+k]
    return out

def recursive(rho, d, n_rounds):
    r = rho.copy()
    for _ in range(n_rounds): r = swap_test(r, r, d)
    return r

def twirl_analytical(rho_ideal, gamma_eff, d):
    e_g = np.exp(-gamma_eff)
    F_avg = e_g + (1-e_g)*2/(d*(d+1))
    p_eff = max(0, 1-(d*F_avg-1)/(d-1))
    return depolarizing(rho_ideal, p_eff)

def make_cat(d):
    psi = np.ones(d,dtype=complex)/np.sqrt(d)
    return np.outer(psi,psi.conj())

N_DD = 8  # CPMG-8

def pipeline(rho_ideal, d, gamma, n_rounds, use_dd):
    """Raw or DD+Twirl pipeline."""
    if use_dd:
        gamma_eff = gamma / (N_DD + 1)
        rho_noisy = twirl_analytical(rho_ideal, gamma_eff, d)
    else:
        rho_noisy = dephasing(rho_ideal, gamma)
    return recursive(rho_noisy, d, n_rounds)

plt.rcParams.update({
    'font.size':11,'font.family':'Arial',
    'mathtext.fontset':'stix',
    'axes.labelsize':13,'axes.titlesize':13,
    'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':9,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight',
})

C_RAW = '#e74c3c'
C_DD  = '#2ecc71'

# ============================================================
# Fig 1: F_cat vs γ (fixed n=8)
# ============================================================
print("=== Fig 1: F_cat vs γ ===")
dims = [4, 8, 16]
gammas = np.linspace(0.1, 4.0, 20)
n_r = 3  # 8 copies
dim_colors = {4:'#3498db', 8:'#e67e22', 16:'#8e44ad'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Raw dephasing
ax = axes[0]
for d in dims:
    ideal = make_cat(d)
    fs = [fidelity(ideal, pipeline(ideal, d, g, n_r, False)) for g in gammas]
    ax.plot(gammas, fs, 'o-', color=dim_colors[d], lw=2, ms=4,
            label=rf'$d={d}$')
ax.set_xlabel(r'Dephasing strength $\gamma$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) Raw dephasing ($n=8$)')
ax.legend(); ax.set_ylim(0, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

# Panel (b): DD+Twirl
ax = axes[1]
for d in dims:
    ideal = make_cat(d)
    fs = [fidelity(ideal, pipeline(ideal, d, g, n_r, True)) for g in gammas]
    ax.plot(gammas, fs, 's-', color=dim_colors[d], lw=2, ms=4,
            label=rf'$d={d}$')
ax.set_xlabel(r'Dephasing strength $\gamma$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) DD+Twirl (CPMG-8, $n=8$)')
ax.legend(); ax.set_ylim(0, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

plt.suptitle(r'$F_{\mathrm{cat}}$ vs. Error Strength: Raw vs DD+Twirl',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_dd_vs_error.png'); plt.savefig('fig_dd_vs_error.pdf')
print("Saved fig_dd_vs_error"); plt.close()

# ============================================================
# Fig 2: F_cat vs n_q (fixed γ=2)
# ============================================================
print("\n=== Fig 2: F_cat vs n_q ===")
dims_q = [2, 4, 8, 16, 32]
gamma_fix = 2.0
rounds_list = [1, 2, 3, 4]  # n=2,4,8,16
rcolors = ['#3498db','#e74c3c','#2ecc71','#9b59b6']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for ri, nr in enumerate(rounds_list):
    fs = []
    for d in dims_q:
        ideal = make_cat(d)
        rho_p = pipeline(ideal, d, gamma_fix, nr, False)
        fs.append(fidelity(ideal, rho_p))
    qb = [int(np.log2(d)) for d in dims_q]
    ax.plot(qb, fs, 'o-', color=rcolors[ri], lw=2, ms=6,
            label=rf'$n={2**nr}$')
ax.set_xlabel(r'Qubits $n_q$'); ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) Raw dephasing ($\gamma=2$)'); ax.legend(fontsize=8)
ax.set_ylim(0, 1.05); ax.set_xticks([1,2,3,4,5])
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

ax = axes[1]
for ri, nr in enumerate(rounds_list):
    fs = []
    for d in dims_q:
        ideal = make_cat(d)
        rho_p = pipeline(ideal, d, gamma_fix, nr, True)
        fs.append(fidelity(ideal, rho_p))
        print(f"  DD+Tw d={d} n={2**nr}: F={fs[-1]:.4f}")
    qb = [int(np.log2(d)) for d in dims_q]
    ax.plot(qb, fs, 's-', color=rcolors[ri], lw=2, ms=6,
            label=rf'$n={2**nr}$')
ax.set_xlabel(r'Qubits $n_q$'); ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) DD+Twirl CPMG-8 ($\gamma=2$)'); ax.legend(fontsize=8)
ax.set_ylim(0, 1.05); ax.set_xticks([1,2,3,4,5])
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

plt.suptitle(r'$F_{\mathrm{cat}}$ vs. System Size: Raw vs DD+Twirl',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_dd_vs_qubits.png'); plt.savefig('fig_dd_vs_qubits.pdf')
print("Saved fig_dd_vs_qubits"); plt.close()

# ============================================================
# Fig 3: F_cat vs copies (fixed γ=2)
# ============================================================
print("\n=== Fig 3: F_cat vs copies ===")
dims_c = [4, 8, 16]
max_r = {4:6, 8:6, 16:5}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for d in dims_c:
    ideal = make_cat(d)
    ns, fs = [], []
    for nr in range(1, max_r[d]+1):
        rho_p = pipeline(ideal, d, 2.0, nr, False)
        ns.append(2**nr); fs.append(fidelity(ideal, rho_p))
    ax.semilogx(ns, fs, 'o-', color=dim_colors[d], lw=2, ms=5,
                label=rf'$d={d}$')
ax.set_xlabel(r'Copies $n$'); ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) Raw dephasing ($\gamma=2$)'); ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)

ax = axes[1]
for d in dims_c:
    ideal = make_cat(d)
    ns, fs = [], []
    for nr in range(1, max_r[d]+1):
        rho_p = pipeline(ideal, d, 2.0, nr, True)
        ns.append(2**nr); fs.append(fidelity(ideal, rho_p))
        print(f"  DD+Tw d={d} n={ns[-1]}: F={fs[-1]:.4f}")
    ax.semilogx(ns, fs, 's-', color=dim_colors[d], lw=2, ms=5,
                label=rf'$d={d}$')
ax.set_xlabel(r'Copies $n$'); ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) DD+Twirl CPMG-8 ($\gamma=2$)'); ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(y=0.95, color='gray', ls=':', alpha=0.4)
ax.axhline(y=0.99, color='gray', ls='--', alpha=0.3)

plt.suptitle(r'$F_{\mathrm{cat}}$ vs. Copy Count: Raw vs DD+Twirl',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_dd_vs_copies.png'); plt.savefig('fig_dd_vs_copies.pdf')
print("Saved fig_dd_vs_copies"); plt.close()

print("\nAll done.")

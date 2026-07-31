#!/usr/bin/env python3
"""
plot_three_scaling.py — Three scaling figures with CORRECT (I+SWAP_E)/2 projector.

  fig_fidelity_vs_error.pdf:  F_cat vs γ and p  (fixed n=8)
  fig_fidelity_vs_qubits.pdf: F_cat vs n_q      (fixed noise)
  fig_fidelity_vs_copies.pdf: F_cat vs n         (fixed d, fixed noise)
"""

import numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- core ---
def fidelity(rho, sigma):
    e,v = np.linalg.eigh(rho)
    sr = v @ np.diag(np.sqrt(np.maximum(e,0))) @ v.conj().T
    M = sr @ sigma @ sr
    e2 = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e2,0)))**2)

def dephasing(rho, gamma):
    d = rho.shape[0]
    mask = np.exp(-gamma * np.abs(np.arange(d)[:,None] - np.arange(d)[None,:]))
    return rho * mask

def depolarizing(rho, p):
    d = rho.shape[0]
    return (1-p)*rho + p*np.eye(d)/d

# --- CORRECT swap tests (full d^2 x d^2) ---
def _build_swap(d):
    d2 = d*d
    S = np.zeros((d2,d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j*d+i, i*d+j] = 1.0
    return S

def standard_swap(rho, d):
    d2 = d*d
    rs = np.kron(rho, rho)
    Pi = (np.eye(d2) + _build_swap(d)) / 2.0
    proj = Pi @ rs @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15: return rho.copy()
    proj /= p
    out = np.zeros((d,d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i,j] += proj[i*d+k, j*d+k]
    return out

def covariant_swap(rho, d):
    d2 = d*d
    rs = np.kron(rho, rho)
    theta = np.pi/4; c,s = np.cos(theta), np.sin(theta)
    U = np.eye(d2, dtype=complex)
    for E in range(2*(d-1)+1):
        sec = [(i,E-i) for i in range(d) if 0<=E-i<d]
        if len(sec)<2: continue
        for k in range(len(sec)-1):
            i1,j1=sec[k]; i2,j2=sec[k+1]
            a,b = i1*d+j1, i2*d+j2
            U[a,a]=c; U[b,b]=c; U[a,b]=-1j*s; U[b,a]=-1j*s
    state = U @ rs @ U.conj().T
    # CORRECT (I+SWAP_E)/2
    Pi = np.zeros((d2,d2), dtype=complex)
    for E in range(2*(d-1)+1):
        sec = [(i,E-i) for i in range(d) if 0<=E-i<d]
        ns = len(sec)
        if ns==0: continue
        idx = [i*d+j for i,j in sec]
        if ns==1:
            Pi[idx[0],idx[0]]=1.0; continue
        SW = np.zeros((ns,ns),dtype=complex)
        for a,(ia,ja) in enumerate(sec):
            for b,(ib,jb) in enumerate(sec):
                if ib==ja and jb==ia: SW[a,b]=1.0
        PE = (np.eye(ns)+SW)/2.0
        for a in range(ns):
            for b in range(ns):
                Pi[idx[a],idx[b]] = PE[a,b]
    proj = Pi @ state @ Pi
    p = float(np.real(np.trace(proj)))
    if p<1e-15: return rho.copy()
    proj /= p
    out = np.zeros((d,d),dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i,j] += proj[i*d+k, j*d+k]
    return out

def rec(rho, d, rounds, method):
    fn = covariant_swap if method=='cov' else standard_swap
    r = rho.copy()
    for _ in range(rounds): r = fn(r, d)
    return r

def make_cat(d):
    psi = np.ones(d,dtype=complex)/np.sqrt(d)
    return np.outer(psi,psi.conj())

# --- style ---
plt.rcParams.update({
    'font.size':11,'font.family':'serif',
    'axes.labelsize':13,'axes.titlesize':13,
    'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':8,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight',
})
mc = {'std':'#e74c3c','cov':'#3498db'}
dc = {4:'#2ecc71',8:'#3498db',16:'#e74c3c',32:'#9b59b6'}

# ============================================================
# Fig 1: F_cat vs error strength
# ============================================================
print("=== Fig 1: F_cat vs error ===")
dims_e = [4,8,16]
rounds_fix = 3  # n=8
gammas = np.linspace(0.2, 4.0, 15)
ps = np.linspace(0.05, 0.6, 12)

fig, axes = plt.subplots(1,2,figsize=(14,5.5))

ax = axes[0]
for d in dims_e:
    ideal = make_cat(d)
    fs,fc = [],[]
    for g in gammas:
        rn = dephasing(ideal, g)
        rs = rec(rn,d,rounds_fix,'std'); rc = rec(rn,d,rounds_fix,'cov')
        fs.append(fidelity(ideal,rs)); fc.append(fidelity(ideal,rc))
    ax.plot(gammas,fc,'o-',color=dc[d],lw=2,ms=4,label=rf'Covariant $d={d}$')
    ax.plot(gammas,fs,'s--',color=dc[d],lw=1,ms=3,alpha=0.5,label=rf'Standard $d={d}$')
    print(f"  d={d} deph done")
ax.set_xlabel(r'Dephasing strength $\gamma$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) $F_{\mathrm{cat}}$ vs. $\gamma$ ($n=8$ copies)')
ax.legend(loc='upper right',fontsize=7,ncol=2)
ax.set_ylim(0,1.02)

ax = axes[1]
for d in dims_e:
    ideal = make_cat(d)
    fs,fc = [],[]
    for p in ps:
        rn = depolarizing(ideal,p)
        rs = rec(rn,d,rounds_fix,'std'); rc = rec(rn,d,rounds_fix,'cov')
        fs.append(fidelity(ideal,rs)); fc.append(fidelity(ideal,rc))
    ax.plot(ps,fc,'o-',color=dc[d],lw=2,ms=4,label=rf'Covariant $d={d}$')
    ax.plot(ps,fs,'s--',color=dc[d],lw=1,ms=3,alpha=0.5,label=rf'Standard $d={d}$')
    print(f"  d={d} depol done")
ax.set_xlabel(r'Depolarizing probability $p$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) $F_{\mathrm{cat}}$ vs. $p$ ($n=8$ copies)')
ax.legend(loc='lower left',fontsize=7,ncol=2)
ax.set_ylim(0.3,1.02)

plt.suptitle('Catalyst Fidelity vs. Error Strength',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fidelity_vs_error.png'); plt.savefig('fig_fidelity_vs_error.pdf')
print("Saved fig_fidelity_vs_error")
plt.close()

# ============================================================
# Fig 2: F_cat vs qubit count
# ============================================================
print("\n=== Fig 2: F_cat vs qubits ===")
dims_q = [2,4,8,16,32]
rlist = [1,2,3,4]  # n=2,4,8,16

fig, axes = plt.subplots(1,2,figsize=(14,5.5))
colors_r = ['#3498db','#e74c3c','#2ecc71','#9b59b6']

ax = axes[0]
for ri,nr in enumerate(rlist):
    fs_all,fc_all = [],[]
    for d in dims_q:
        ideal = make_cat(d)
        rn = dephasing(ideal,2.0)
        rs = rec(rn,d,nr,'std'); rc = rec(rn,d,nr,'cov')
        fs_all.append(fidelity(ideal,rs)); fc_all.append(fidelity(ideal,rc))
        print(f"  deph d={d} n={2**nr}: std={fs_all[-1]:.4f} cov={fc_all[-1]:.4f}")
    qb = [int(np.log2(d)) for d in dims_q]
    nc = 2**nr
    ax.plot(qb,fc_all,'o-',color=colors_r[ri],lw=2,ms=6,label=rf'Cov $n={nc}$')
    ax.plot(qb,fs_all,'s--',color=colors_r[ri],lw=1,ms=4,alpha=0.4,label=rf'Std $n={nc}$')
ax.set_xlabel(r'Qubits $n_q = \log_2 d$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) Dephasing $\gamma=2.0$')
ax.legend(loc='upper right',fontsize=6,ncol=2)
ax.set_ylim(0,0.85)
ax.set_xticks([1,2,3,4,5])

ax = axes[1]
for ri,nr in enumerate(rlist):
    fs_all,fc_all = [],[]
    for d in dims_q:
        ideal = make_cat(d)
        rn = depolarizing(ideal,0.3)
        rs = rec(rn,d,nr,'std'); rc = rec(rn,d,nr,'cov')
        fs_all.append(fidelity(ideal,rs)); fc_all.append(fidelity(ideal,rc))
    qb = [int(np.log2(d)) for d in dims_q]
    nc = 2**nr
    ax.plot(qb,fc_all,'o-',color=colors_r[ri],lw=2,ms=6,label=rf'Cov $n={nc}$')
    ax.plot(qb,fs_all,'s--',color=colors_r[ri],lw=1,ms=4,alpha=0.4,label=rf'Std $n={nc}$')
ax.set_xlabel(r'Qubits $n_q = \log_2 d$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) Depolarizing $p=0.3$')
ax.legend(loc='lower left',fontsize=6,ncol=2)
ax.set_ylim(0.55,1.02)
ax.set_xticks([1,2,3,4,5])

plt.suptitle('Catalyst Fidelity vs. System Size',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fidelity_vs_qubits.png'); plt.savefig('fig_fidelity_vs_qubits.pdf')
print("Saved fig_fidelity_vs_qubits")
plt.close()

# ============================================================
# Fig 3: F_cat vs copy count
# ============================================================
print("\n=== Fig 3: F_cat vs copies ===")
dims_c = [4,8,16]
max_r = {4:6, 8:6, 16:5}

fig, axes = plt.subplots(1,2,figsize=(14,5.5))

ax = axes[0]
for d in dims_c:
    ideal = make_cat(d)
    rn = dephasing(ideal,2.0)
    rs_all,rc_all,ns_all = [],[],[]
    for nr in range(1,max_r[d]+1):
        rs = rec(rn,d,nr,'std'); rc = rec(rn,d,nr,'cov')
        ns_all.append(2**nr)
        rs_all.append(fidelity(ideal,rs)); rc_all.append(fidelity(ideal,rc))
    ax.semilogx(ns_all,rc_all,'o-',color=dc[d],lw=2,ms=5,label=rf'Cov $d={d}$')
    ax.semilogx(ns_all,rs_all,'s--',color=dc[d],lw=1,ms=3,alpha=0.5,label=rf'Std $d={d}$')
    print(f"  deph d={d}: std={rs_all[-1]:.4f} cov={rc_all[-1]:.4f} at n={ns_all[-1]}")
ax.set_xlabel(r'Number of copies $n$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(a) Dephasing $\gamma=2.0$')
ax.legend(loc='lower right',fontsize=7,ncol=2)
ax.set_ylim(0,0.7)

ax = axes[1]
for d in dims_c:
    ideal = make_cat(d)
    rn = depolarizing(ideal,0.3)
    rs_all,rc_all,ns_all = [],[],[]
    for nr in range(1,max_r[d]+1):
        rs = rec(rn,d,nr,'std'); rc = rec(rn,d,nr,'cov')
        ns_all.append(2**nr)
        rs_all.append(fidelity(ideal,rs)); rc_all.append(fidelity(ideal,rc))
    ax.semilogx(ns_all,rc_all,'o-',color=dc[d],lw=2,ms=5,label=rf'Cov $d={d}$')
    ax.semilogx(ns_all,rs_all,'s--',color=dc[d],lw=1,ms=3,alpha=0.5,label=rf'Std $d={d}$')
    print(f"  depol d={d}: std={rs_all[-1]:.4f} cov={rc_all[-1]:.4f} at n={ns_all[-1]}")
ax.set_xlabel(r'Number of copies $n$')
ax.set_ylabel(r'$F_{\mathrm{cat}}$')
ax.set_title(r'(b) Depolarizing $p=0.3$')
ax.legend(loc='lower right',fontsize=7,ncol=2)
ax.set_ylim(0.65,1.02)

plt.suptitle('Catalyst Fidelity vs. Copy Count',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fidelity_vs_copies.png'); plt.savefig('fig_fidelity_vs_copies.pdf')
print("Saved fig_fidelity_vs_copies")
plt.close()

print("\nAll 3 figures done.")

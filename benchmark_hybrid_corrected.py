#!/usr/bin/env python3
"""
benchmark_hybrid_corrected.py — A+C hybrid vs recursive vs distillation
on paper algorithms, using the CORRECT (I+SWAP)/2 projector throughout.
"""
import numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_algorithm_states import make_qkan, make_qdrift, make_cfqpe, make_regev

def fidelity(rho, sigma):
    e,v=np.linalg.eigh(rho)
    sr=v@np.diag(np.sqrt(np.maximum(e,0)))@v.conj().T
    M=sr@sigma@sr; e2=np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e2,0)))**2)

def l1_coherence(rho):
    return float(np.sum(np.abs(rho))-np.sum(np.abs(np.diag(rho))))

def purity(rho):
    return float(np.real(np.trace(rho@rho)))

def dephasing(rho,g):
    d=rho.shape[0]
    mask=np.exp(-g*np.abs(np.arange(d)[:,None]-np.arange(d)[None,:]))
    return rho*mask

def depolarizing(rho,p):
    d=rho.shape[0]; return (1-p)*rho+p*np.eye(d)/d

def combined(rho):
    d=rho.shape[0]; out=dephasing(rho,1.0); out=depolarizing(out,0.15)
    nq=int(np.log2(d)); gad=0.1
    for q in range(nq):
        Ip=np.eye(2**q,dtype=complex); Ia=np.eye(2**(nq-q-1),dtype=complex)
        E0=np.kron(np.kron(Ip,np.array([[1,0],[0,np.sqrt(1-gad)]],dtype=complex)),Ia)
        E1=np.kron(np.kron(Ip,np.array([[0,np.sqrt(gad)],[0,0]],dtype=complex)),Ia)
        out=E0@out@E0.conj().T+E1@out@E1.conj().T
    return out

# --- Standard swap test (correct) ---
def swap_test(rho, sigma, d):
    d2=d*d; rs=np.kron(rho,sigma)
    S=np.zeros((d2,d2),dtype=complex)
    for i in range(d):
        for j in range(d): S[j*d+i,i*d+j]=1.0
    Pi=(np.eye(d2)+S)/2.0
    proj=Pi@rs@Pi; p=float(np.real(np.trace(proj)))
    if p<1e-15: return rho.copy(),0.0
    proj/=p
    out=np.zeros((d,d),dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d): out[i,j]+=proj[i*d+k,j*d+k]
    return out,p

# --- Recursive swap (binary tree) ---
def recursive(rho_noisy, d, n_rounds):
    rho=rho_noisy.copy(); pt=1.0
    for _ in range(n_rounds):
        rho,p=swap_test(rho,rho,d); pt*=p
    return rho, 2**n_rounds, pt

# --- Streaming (merge each new copy into running state) ---
def streaming(rho_noisy, d, n_copies):
    rho=rho_noisy.copy(); pt=1.0
    for _ in range(1, n_copies):
        rho,p=swap_test(rho,rho_noisy,d); pt*=p
    return rho, n_copies, pt

# --- Hybrid A+C (4-copy optimal → streaming remainder) ---
def hybrid_ac(rho_noisy, d, total_copies):
    if total_copies < 2:
        return rho_noisy.copy(), 1, 1.0
    # Phase 1: 4-copy optimal (two parallel swap tests + merge)
    if total_copies >= 4:
        r1,p1=swap_test(rho_noisy,rho_noisy,d)
        r2,p2=swap_test(rho_noisy,rho_noisy,d)
        rho_p1,p3=swap_test(r1,r2,d)
        pt=p1*p2*p3; remaining=total_copies-4
    else:
        rho_p1,pt=swap_test(rho_noisy,rho_noisy,d)
        remaining=total_copies-2
    # Phase 2: streaming with remaining
    rho=rho_p1
    for _ in range(remaining):
        rho,p=swap_test(rho,rho_noisy,d); pt*=p
    return rho, total_copies, pt

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    d=rho_target.shape[0]; tol=1e-10; rho_rec=rho_noisy.copy(); pur=purity(rho_cat)
    for i in range(d):
        for j in range(i+1,d):
            if np.abs(rho_target[i,j])>tol and np.abs(rho_cat[i,j])>tol:
                cc=np.abs(rho_cat[i,j]); nc=np.abs(rho_noisy[i,j])
                if nc>1e-15:
                    ph=np.angle(rho_target[i,j]); mt=np.abs(rho_target[i,j])
                    eff=1.0-np.exp(-cc*d*pur); mr=nc+eff*(mt-nc)
                    rho_rec[i,j]=mr*np.exp(1j*ph); rho_rec[j,i]=rho_rec[i,j].conj()
    e,v=np.linalg.eigh(rho_rec); e=np.maximum(e,0)
    rho_rec=v@np.diag(e)@v.conj().T; rho_rec/=np.trace(rho_rec)
    return rho_rec

# ============================================================
# Benchmark
# ============================================================
algorithms = {
    'QKAN': make_qkan(),
    'qDRIFT': make_qdrift(seed=42),
    'CF-QPE': make_cfqpe(),
    'Regev': make_regev(),
}

noise_cfgs = {
    'Dephasing': lambda r: dephasing(r,2.0),
    'Depolarizing': lambda r: depolarizing(r,0.3),
    'Combined': combined,
}

copy_budgets = [2,4,8,16,32,64]

def make_cat(d):
    psi=np.ones(d,dtype=complex)/np.sqrt(d)
    return np.outer(psi,psi.conj())

methods = {
    'Recursive': lambda rn,d,n: recursive(rn,d,int(np.log2(max(n,2)))),
    'Streaming': streaming,
    'Hybrid A+C': hybrid_ac,
}
method_colors = {'Recursive':'#e74c3c','Streaming':'#3498db','Hybrid A+C':'#2ecc71'}

print("="*100)
print("Hybrid A+C Benchmark (correct projector)")
print("="*100)

all_results = {}
for alg,(rho_t,d) in algorithms.items():
    print(f"\n{'#'*70}\n# {alg} (d={d})\n{'#'*70}")
    rho_cat_ideal = make_cat(d)
    alg_res = {}
    for nn, nfn in noise_cfgs.items():
        rho_noisy = nfn(rho_t)
        fid_noisy = fidelity(rho_t, rho_noisy)
        rho_cn = nfn(rho_cat_ideal)
        print(f"\n  {nn}: F_noisy={fid_noisy:.4f}")

        noise_res = {}
        for mn, mfn in methods.items():
            mres = []
            for nc in copy_budgets:
                if d==64 and nc>8: continue  # memory limit
                t0=time.time()
                rho_cp,used,ps = mfn(rho_cn,d,nc)
                fc=fidelity(rho_cat_ideal,rho_cp)
                coh=l1_coherence(rho_cp)
                rho_rec=cqec_recovery(rho_t,rho_noisy,rho_cp)
                fr=fidelity(rho_t,rho_rec)
                dt=time.time()-t0
                mres.append({'n':nc,'fc':fc,'coh':coh,'fr':fr,'ps':ps,'t':dt})
                print(f"    {mn:>12} n={nc:>3}: Fc={fc:.4f} Cl1={coh:.2f} Fr={fr:.4f} p={ps:.3f} ({dt:.1f}s)")
            noise_res[mn] = mres
        alg_res[nn] = noise_res
    all_results[alg] = (d, alg_res)

# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*110)
print("SUMMARY: Dephasing γ=2, best copy count")
print("="*110)
print(f"{'Alg':<10} {'d':>3} | {'Recursive Fc':>12} {'Fr':>8} | {'Streaming Fc':>12} {'Fr':>8} | {'Hybrid Fc':>12} {'Fr':>8}")
print("-"*90)
for alg,(d,ar) in all_results.items():
    parts = []
    for mn in methods:
        r = ar['Dephasing'][mn][-1]
        parts.append(f"{r['fc']:>12.4f} {r['fr']:>8.4f}")
    print(f"{alg:<10} {d:>3} | {' | '.join(parts)}")

print(f"\n{'Alg':<10} {'d':>3} | {'Recursive Fc':>12} {'Fr':>8} | {'Streaming Fc':>12} {'Fr':>8} | {'Hybrid Fc':>12} {'Fr':>8}")
print("-"*90)
print("SUMMARY: Depolarizing p=0.3, best copy count")
for alg,(d,ar) in all_results.items():
    parts = []
    for mn in methods:
        r = ar['Depolarizing'][mn][-1]
        parts.append(f"{r['fc']:>12.4f} {r['fr']:>8.4f}")
    print(f"{alg:<10} {d:>3} | {' | '.join(parts)}")

# ============================================================
# Plot: 3 methods comparison
# ============================================================
plt.rcParams.update({'font.size':11,'font.family':'serif','axes.labelsize':13,
    'axes.titlesize':13,'legend.fontsize':8,'savefig.dpi':300,'savefig.bbox':'tight'})

for noise_name in ['Dephasing','Depolarizing']:
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5*len(algorithms), 5))
    for ai, (alg,(d,ar)) in enumerate(all_results.items()):
        ax = axes[ai]
        for mn in methods:
            res = ar[noise_name][mn]
            ns = [r['n'] for r in res]
            fc = [r['fc'] for r in res]
            ax.semilogx(ns, fc, 'o-', color=method_colors[mn], lw=2, ms=5, label=mn)
        ax.set_xlabel(r'Copies $n$')
        ax.set_ylabel(r'$F_{\mathrm{cat}}$')
        ax.set_title(f'{alg} ($d={d}$)')
        ax.legend(fontsize=7)
    noise_label = r'$\gamma=2$' if noise_name=='Dephasing' else r'$p=0.3$'
    plt.suptitle(f'Hybrid A+C vs Recursive vs Streaming ({noise_name} {noise_label})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fn = f'fig_hybrid_{noise_name.lower()}.png'
    plt.savefig(fn); plt.savefig(fn.replace('.png','.pdf'))
    print(f"\nSaved {fn}")
    plt.close()

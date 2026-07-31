"""Supplementary data addressing the Nature-review gaps:
  A  dimension-free collapse: 1-F vs N coincides across n=2..6 (validates Thm 1).
  B  instance-averaged Table I: mean +- std over random noise instances.
  C  mixed-target rank scaling: 1-F ~ sqrt(R) (validates Remark 1(iv)).
  D  error-rate overhead: (1-F)*sqrt(N) ~ e^{Gamma} (validates lambda_min~e^-Gamma).
Outputs figures/fig_dimfree.pdf and prints tables.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from algorithms_faithful import (make_qkan_faithful, make_qdrift_faithful,
                                 make_cfqpe_faithful, make_regev_faithful)
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (learn_pauli_channel, apply_pauli_channel,
                                        pauli_matrix, pauli_eigenvalue, psd_normalize)

FIGDIR = os.path.join(HERE, "..", "Lnl", "figures")
plt.rcParams.update({"font.size": 10})


def pstack(n):
    return np.stack([pauli_matrix(b, n) for b in range(4 ** n)])


def rand_channel(n, rng, scale=1.0):
    ch = {}
    for i in range(n):
        ch[3 * 4 ** i] = scale * rng.uniform(0.02, 0.06)
        ch[1 * 4 ** i] = scale * rng.uniform(0.005, 0.025)
    s = sum(ch.values())
    if s > 0.85:                                   # keep c_0 > 0.15 (valid channel)
        for k in ch:
            ch[k] *= 0.85 / s
    ch[0] = 1.0 - sum(ch.values())
    return ch


def recover(rho, ch, n, Ps, N, rng, want_tn=False):
    noisy = apply_pauli_channel(rho, ch, n)
    info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=rng)
    lr = dict(info["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.einsum("bij,ji->b", Ps, noisy)
        lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
        lam = np.where(np.abs(lam) < 1e-3, np.sign(lam) * 1e-3 + (lam == 0) * 1e-3, lam)
        est = np.tensordot(r / lam / 2 ** n, Ps, axes=([0], [0]))
    est = psd_normalize(est)
    F = fidelity(est, rho)
    if want_tn:
        tn = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(est - rho))))
        return F, tn
    return F


# ---------- A: dimension-free fidelity (controlled: fixed support & lambda_min) ----------
def part_A():
    """Controlled test of Eq. (fiddim): hold the support (qubit-0 Z,X) and hence
    |S| and lambda_min fixed across n, vary only the dimension d, average over
    random pure targets. The fidelity bound predicts 1-F independent of d."""
    print("A  dimension-free fidelity (fixed support {Z0,X0}, random pure targets)")
    N = 16000
    ds, inf_m, inf_e = [], [], []
    for n in [2, 3, 4, 5, 6]:
        d = 2 ** n; Ps = pstack(n)
        ch = {3: 0.05, 1: 0.02}; ch[0] = 1 - sum(ch.values())   # same channel, all n
        n_inst = 24 if n <= 4 else 12
        fs = []
        for k in range(n_inst):
            rng = np.random.default_rng(100 * k + n)
            v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
            rho = np.outer(v, v.conj())
            fs.append(1 - recover(rho, ch, n, Ps, N, rng))
        ds.append(d); inf_m.append(np.mean(fs)); inf_e.append(np.std(fs) / np.sqrt(n_inst))
        print(f"   n={n} d={d:>2d}: 1-F = {np.mean(fs):.5f} +- {inf_e[-1]:.5f}")
    ds = np.array(ds); inf_m = np.array(inf_m)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.errorbar(ds, inf_m, yerr=inf_e, marker="o", ms=7, capsize=3, color="C0",
                label=r"measured $1-F$ (flat in $d$)")
    ax.plot(ds, inf_m[0] * np.sqrt(ds / ds[0]), "C3--",
            label=r"worst-case $\sqrt{d}$ trace-norm bound")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"Hilbert dimension $d=2^n$")
    ax.set_ylabel(r"recovery infidelity $1-F$")
    ax.set_ylim(1e-4, 3e-3)
    ax.set_title(r"Fidelity is dimension-free (Eq.~7); $N=16000$", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); out = os.path.join(FIGDIR, "fig_dimfree.pdf")
    fig.savefig(out); print("   wrote", out)


# ---------- B: instance-averaged Table I ----------
def part_B():
    print("\nB  instance-averaged recovery (mean +- std over 30 noise instances, N=64000)")
    algos = [("QKAN", make_qkan_faithful), ("qDRIFT", make_qdrift_faithful),
             ("CF-QPE", make_cfqpe_faithful), ("Regev", make_regev_faithful)]
    N = 64000
    print(f"   {'algo':<7s} {'n':>2s}  {'F_noisy':>14s}  {'F_LearnCI':>16s}")
    for name, fn in algos:
        rho, d = fn(); n = int(round(np.log2(d))); Ps = pstack(n)
        n_inst = 30 if n <= 4 else 12
        fno, flc = [], []
        for k in range(n_inst):
            rng = np.random.default_rng(7 * k + d)
            ch = rand_channel(n, rng)
            fno.append(fidelity(apply_pauli_channel(rho, ch, n), rho))
            flc.append(recover(rho, ch, n, Ps, N, rng))
        print(f"   {name:<7s} {n:>2d}  {np.mean(fno):.4f}+-{np.std(fno):.4f}  "
              f"{np.mean(flc):.4f}+-{np.std(flc):.4f}")


# ---------- C: mixed-target rank scaling ----------
def part_C():
    print("\nC  mixed-target rank scaling (1-F vs rank R; predict ~ sqrt(R))")
    n = 4; d = 16; Ps = pstack(n); N = 16000
    print(f"   {'R':>3s}  {'1-F':>8s}  {'(1-F)/sqrt(R)':>14s}")
    for R in [1, 2, 4, 8, 16]:
        vals = []
        for k in range(15):
            rng = np.random.default_rng(3 * k + R)
            # rank-R target: mixture of R random pure states
            rho = np.zeros((d, d), dtype=complex)
            for _ in range(R):
                v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
                rho += np.outer(v, v.conj())
            rho /= R
            ch = rand_channel(n, rng)
            vals.append(1 - recover(rho, ch, n, Ps, N, rng))
        m = np.mean(vals)
        print(f"   {R:>3d}  {m:>8.4f}  {m/np.sqrt(R):>14.4f}")


# ---------- D: error-rate overhead ----------
def part_D():
    print("\nD  error-rate overhead ((1-F)*sqrt(N) vs total rate Gamma; predict ~ e^Gamma)")
    n = 4; Ps = pstack(n); N = 16000
    rho, _ = make_cfqpe_faithful()
    print(f"   {'Gamma':>6s}  {'lambda_min':>10s}  {'(1-F)*sqrt(N)':>14s}")
    for scale in [0.5, 1.0, 2.0, 3.0, 4.0]:
        vals, lmins = [], []
        for k in range(12):
            rng = np.random.default_rng(5 * k + int(scale * 10))
            ch = rand_channel(n, rng, scale=scale)
            Gamma = sum(v for a, v in ch.items() if a != 0)
            lam = [pauli_eigenvalue(ch, b, n) for b in range(1, 4 ** n)]
            lmins.append(min(abs(l) for l in lam))
            vals.append((1 - recover(rho, ch, n, Ps, N, rng)) * np.sqrt(N))
        Gamma = sum(v for a, v in ch.items() if a != 0)
        print(f"   {Gamma:>6.3f}  {np.mean(lmins):>10.4f}  {np.mean(vals):>14.3f}")


if __name__ == "__main__":
    part_A(); part_B(); part_C(); part_D()

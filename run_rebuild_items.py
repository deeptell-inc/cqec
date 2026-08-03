"""Rebuild items demanded by the adversarial panel (2026-07-31).

A  Finite state-copy scaling: with M copies spent on Pauli-component estimation
   of rho_noisy (exact channel known, to isolate state access), measure
   1-F vs M. Prediction (new Prop): 1-F = O(sqrt(d/M)/lambda_min)  — SQL in M
   with an explicitly dimension-dependent prefactor ~ sqrt(d).
B  Truncation bias under TRUE Markovian noise: product channel e^{tL} of a
   sparse generator (per-qubit X and Z rates) has FULL 4^n Pauli support.
   Run the actual Bell+WHT pipeline; measure the N-independent infidelity
   floor and compare it to the truncated tail weight sum_{a not in S_hat} c_a.
C  Purity-constrained zero-channel-shot inversion (panel finding, reproduced):
   with exact classical access to rho_noisy and a pure-target promise, minimise
   the rank-1 residual of E_c^{-1}(rho_noisy) over sparse rates c — zero
   channel-learning shots.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    learn_pauli_channel, apply_pauli_channel, pauli_matrix, pauli_eigenvalue,
    psd_normalize, pauli_digits,
)


def pstack(n):
    return np.stack([pauli_matrix(b, n) for b in range(4 ** n)])


def rand_pure(d, rng):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


# ---------------------------------------------------------------- A ----------
def part_A():
    print("A  finite state-copy scaling (exact channel; M split over 4^n Pauli components)")
    print(f"   {'n':>2s} {'d':>3s}  " + "  ".join(f"M=10^{k}".rjust(9) for k in (5, 6, 7))
          + "   slope_M   prefactor")
    rows = {}
    for n in (2, 3, 4):
        d = 2 ** n
        Ps = pstack(n)
        ch = {3: 0.05, 1: 0.02}
        ch[0] = 1 - sum(ch.values())
        lam = np.array([pauli_eigenvalue(ch, b, n) for b in range(4 ** n)])
        means = []
        for M in (1e5, 1e6, 1e7):
            vals = []
            n_inst = 30 if n <= 3 else 15
            for k in range(n_inst):
                rng = np.random.default_rng(11 * k + n + int(np.log10(M)))
                rho = rand_pure(d, rng)
                noisy = apply_pauli_channel(rho, ch, n)
                with np.errstate(all="ignore"):
                    r = np.real(np.einsum("bij,ji->b", Ps, noisy))
                    Mb = M / 4 ** n                      # shots per Pauli component
                    sd = np.sqrt(np.maximum(1 - r ** 2, 0) / Mb)
                    r_hat = r + rng.normal(scale=sd)
                    est = np.tensordot(r_hat / lam / d, Ps, axes=([0], [0]))
                vals.append(1 - fidelity(psd_normalize(est), rho))
            means.append(np.mean(vals))
        slope = np.polyfit(np.log([1e5, 1e6, 1e7]), np.log(means), 1)[0]
        rows[n] = means[1]                                # at M=1e6, for prefactor ratios
        print(f"   {n:>2d} {d:>3d}  " + "  ".join(f"{m:9.5f}" for m in means)
              + f"   {slope:+.2f}     {means[1]:.5f}")
    r32 = rows[3] / rows[2]
    r43 = rows[4] / rows[3]
    print(f"   prefactor ratios at M=1e6:  (n=3)/(n=2) = {r32:.2f},  (n=4)/(n=3) = {r43:.2f}"
          f"   (sqrt(2) = 1.41 per qubit predicted by sqrt(d))")


# ---------------------------------------------------------------- B ----------
def product_markovian_channel(n, t, rng):
    """e^{tL} for a sparse generator: per-qubit X,Z rates -> FULL-support channel."""
    per_qubit = []
    for _ in range(n):
        gz = rng.uniform(0.02, 0.06)
        gx = rng.uniform(0.005, 0.025)
        lam = {0: 1.0,
               1: np.exp(-2 * t * gz),            # X survives Z-noise
               2: np.exp(-2 * t * (gx + gz)),     # Y hit by both
               3: np.exp(-2 * t * gx)}            # Z survives X-noise
        p = {a: 0.25 * sum(lam[b] * (1 if (a == 0 or b == 0 or a == b) else -1)
                           for b in range(4)) for a in range(4)}
        per_qubit.append(p)
    ch = {}
    for a in range(4 ** n):
        digs = pauli_digits(a, n)
        c = 1.0
        for i, dgt in enumerate(digs):
            c *= per_qubit[i][dgt]
        if c > 1e-15:
            ch[a] = c
    return ch


def part_B():
    print("\nB  truncation bias under TRUE Markovian product noise (full 4^n support)")
    print(f"   {'n':>2s}  {'|supp|':>6s}  " + "  ".join(f"N={N}".rjust(9) for N in (4000, 16000, 64000, 256000))
          + "   tail-weight bound")
    for n in (2, 3, 4):
        d = 2 ** n
        Ps = pstack(n)
        cols = {N: [] for N in (4000, 16000, 64000, 256000)}
        tails = []
        supps = []
        n_inst = 12 if n <= 3 else 8
        for k in range(n_inst):
            rng = np.random.default_rng(101 * k + n)
            ch = product_markovian_channel(n, 1.0, rng)
            supps.append(sum(1 for a, c in ch.items() if a != 0 and c > 1e-12))
            rho = rand_pure(d, rng)
            noisy = apply_pauli_channel(rho, ch, n)
            for N in cols:
                r = np.random.default_rng(7 * k + N + n)
                info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
                lr = dict(info["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
                tail = sum(c for a, c in ch.items() if a != 0 and a not in info["support"])
                if N == 256000:
                    tails.append(tail)
                with np.errstate(all="ignore"):
                    rr = np.einsum("bij,ji->b", Ps, noisy)
                    lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
                    lam = np.where(np.abs(lam) < 1e-3,
                                   np.sign(lam) * 1e-3 + (lam == 0) * 1e-3, lam)
                    est = np.tensordot(rr / lam / d, Ps, axes=([0], [0]))
                cols[N].append(1 - fidelity(psd_normalize(est), rho))
        print(f"   {n:>2d}  {np.mean(supps):6.1f}  "
              + "  ".join(f"{np.mean(cols[N]):9.5f}" for N in cols)
              + f"   {np.mean(tails):.5f}")


# ---------------------------------------------------------------- C ----------
def part_C():
    print("\nC  purity-constrained inversion: ZERO channel-learning shots (reproduction)")
    n = 2
    d = 4
    Ps = pstack(n)
    idxs = [3, 1, 3 * 4, 1 * 4]                    # Z0, X0, Z1, X1
    fails, infs = 0, []
    for k in range(12):
        rng = np.random.default_rng(31 * k)
        true = {a: g for a, g in zip(idxs, rng.uniform(0.01, 0.06, size=4))}
        ch = dict(true); ch[0] = 1 - sum(ch.values())
        rho = rand_pure(d, rng)
        noisy = apply_pauli_channel(rho, ch, n)
        with np.errstate(all="ignore"):
            r = np.real(np.einsum("bij,ji->b", Ps, noisy))

        def resid(x):
            lr = {a: max(v, 0.0) for a, v in zip(idxs, x)}
            lr[0] = max(1 - sum(lr.values()), 0.0)
            lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
            lam = np.where(np.abs(lam) < 1e-6, 1e-6, lam)
            est = np.tensordot(r / lam / d, Ps, axes=([0], [0]))
            est = (est + est.conj().T) / 2
            w = np.sort(np.linalg.eigvalsh(est))
            return float(np.sum(w[:-1] ** 2))       # rank-1 residual

        best = None
        for s in range(6):
            x0 = np.full(4, 0.03) if s == 0 else np.random.default_rng(s + 50 * k).uniform(0.005, 0.08, 4)
            res = minimize(resid, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-12, "fatol": 1e-16})
            if best is None or res.fun < best.fun:
                best = res
        lr = {a: max(v, 0.0) for a, v in zip(idxs, best.x)}
        lr[0] = max(1 - sum(lr.values()), 0.0)
        with np.errstate(all="ignore"):
            lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
            lam = np.where(np.abs(lam) < 1e-6, 1e-6, lam)
            est = psd_normalize(np.tensordot(r / lam / d, Ps, axes=([0], [0])))
        inf = 1 - fidelity(est, rho)
        infs.append(inf)
        if inf > 1e-6:
            fails += 1
    print(f"   n=2, 12 instances: median 1-F = {np.median(infs):.2e}, "
          f"max = {np.max(infs):.2e}, exact (<1e-6) in {12 - fails}/12")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()

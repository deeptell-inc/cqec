"""Regenerate Table I under a SINGLE protocol (panel finding: the previous
table composited two different experiments).

One protocol for every column: faithful algorithm states, random per-qubit
Pauli rates (Z in [0.02,0.06], X in [0.005,0.025]), learn via Bell + WHT with
per-probe budget N, recover with the HS-projection psd_normalize. Per instance
we record F_noisy, F at each N, cond(M); the slope is fit per instance over
the N grid and averaged. Mean +- std over instances.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from algorithms_faithful import (make_qkan_faithful, make_qdrift_faithful,
                                 make_cfqpe_faithful, make_regev_faithful)
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (learn_pauli_channel, apply_pauli_channel,
                                        pauli_matrix, pauli_eigenvalue, psd_normalize)
from run_supplementary_data import rand_channel

ALGOS = [("QKAN", make_qkan_faithful), ("qDRIFT", make_qdrift_faithful),
         ("CF-QPE", make_cfqpe_faithful), ("Regev", make_regev_faithful)]
NS = [1000, 4000, 16000, 64000]


def pstack(n):
    return np.stack([pauli_matrix(b, n) for b in range(4 ** n)])


def recover(noisy, lr, n, Ps):
    with np.errstate(all="ignore"):
        r = np.einsum("bij,ji->b", Ps, noisy)
        lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
        lam = np.where(np.abs(lam) < 1e-3, np.sign(lam) * 1e-3 + (lam == 0) * 1e-3, lam)
        est = np.tensordot(r / lam / 2 ** n, Ps, axes=([0], [0]))
    return psd_normalize(est)


def main():
    print("Single-protocol Table I regeneration (faithful states, random rates,")
    print(f"HS-projection recovery). N grid = {NS}; per-probe budget semantics.\n")
    print(f"{'algo':<7s} {'n':>2s}  {'F_noisy':>15s}  {'F_64k':>17s}  "
          f"{'slope':>13s}  {'cond':>11s}")
    for name, fn in ALGOS:
        rho, d = fn(); n = int(round(np.log2(d))); Ps = pstack(n)
        n_inst = 12 if n <= 4 else 6
        fno, f64, slopes, conds = [], [], [], []
        for k in range(n_inst):
            rng = np.random.default_rng(7 * k + d)
            ch = rand_channel(n, rng)
            noisy = apply_pauli_channel(rho, ch, n)
            fno.append(fidelity(noisy, rho))
            infs = []
            for N in NS:
                r = np.random.default_rng(1000 * k + N + d)
                info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
                lr = dict(info["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
                F = fidelity(recover(noisy, lr, n, Ps), rho)
                infs.append(max(1 - F, 1e-9))
                if N == 64000:
                    f64.append(F); conds.append(info["cond"])
            slopes.append(np.polyfit(np.log(NS), np.log(infs), 1)[0])
        print(f"{name:<7s} {n:>2d}  {np.mean(fno):.4f}+-{np.std(fno):.4f}  "
              f"{np.mean(f64):.5f}+-{np.std(f64):.5f}  "
              f"{np.mean(slopes):+.2f}+-{np.std(slopes):.2f}  "
              f"{np.mean(conds):.2f}+-{np.std(conds):.2f}")


if __name__ == "__main__":
    main()

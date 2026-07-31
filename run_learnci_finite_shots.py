"""Finite-shot LearnCI recovery of the four papers' algorithm states.

The density-matrix results (run_learnci_on_papers.py) assume noiseless
read-out.  Here we do the realistic finite-shot evaluation using the n-qubit
Pauli-basis method: each algorithm output (d = 2^n) is treated as an n-qubit
state, hit by an n-qubit Pauli decoherence channel, and recovered with the
finite-shot LearnCI of `nqubit_learning` --- Bell sampling for the support and a
condition-number-O(1) Walsh-Hadamard inversion for the rates, all from N shots.

Reports F_LearnCI vs shot budget N (expected SQL: 1 - F ~ N^{-1/2}), the
support recall, and cond(M) (O(1), independent of n) for n = 2..6.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from cqec.algorithms import make_qkan, make_qdrift, make_cfqpe, make_regev
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    learn_pauli_channel, apply_pauli_channel, invert_pauli_channel, psd_normalize,
)

# Per-qubit Pauli decoherence rates (dephasing Z + bit-flip X on each qubit).
GZ, GX = 0.04, 0.015

ALGOS = [
    ("QKAN   (2410.04435)", make_qkan),
    ("qDRIFT (PRXQ 2.040305)", make_qdrift),
    ("CF-QPE (7qcr-znl2)", make_cfqpe),
    ("Regev  (2308.06572)", make_regev),
]


def paper_noise_channel(n):
    """Sparse single-qubit Pauli channel: Z_i (rate GZ), X_i (rate GX)."""
    ch = {}
    for i in range(n):
        ch[3 * 4 ** i] = GZ   # Z on qubit i
        ch[1 * 4 ** i] = GX   # X on qubit i
    ch[0] = 1.0 - sum(ch.values())
    return ch


def main() -> None:
    Ns = [250, 1000, 4000, 16000, 64000]
    n_trials = 6
    print(f"n-qubit Pauli decoherence: per-qubit Z@{GZ}, X@{GX}; "
          f"finite-shot LearnCI (Bell + WHT).\n")
    print(f"{'algorithm':<24s} {'n':>2s} {'F_noisy':>8s}  " +
          "  ".join(f"N={N}".rjust(9) for N in Ns) + f"  {'cond':>5s}")
    print("-" * 96)
    for name, fn in ALGOS:
        rho, d = fn()
        n = int(np.log2(d))
        ch = paper_noise_channel(n)
        true_supp = set(a for a in ch if a != 0)
        noisy = apply_pauli_channel(rho, ch, n)
        f_noisy = fidelity(noisy, rho)
        cells, conds, recalls = [], [], []
        for N in Ns:
            fr, cd, rc = [], [], []
            for tr in range(n_trials):
                r = np.random.default_rng(97 * tr + N + d)
                info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
                learned = dict(info["rates"])
                learned[0] = max(1.0 - sum(learned.values()), 0.0)
                est = psd_normalize(invert_pauli_channel(noisy, learned, n))
                fr.append(fidelity(est, rho))
                cd.append(info["cond"])
                rc.append(len(true_supp & set(info["support"])) / len(true_supp))
            cells.append(np.mean(fr)); conds.append(np.mean(cd)); recalls.append(np.mean(rc))
        print(f"{name:<24s} {n:>2d} {f_noisy:>8.4f}  " +
              "  ".join(f"{c:>9.4f}" for c in cells) + f"  {np.mean(conds):>5.2f}")
        # SQL slope of infidelity vs N
        inf = np.maximum(1.0 - np.array(cells), 1e-6)
        slope = np.polyfit(np.log(Ns), np.log(inf), 1)[0]
        print(f"{'':<24s}    supp_recall(min)={min(recalls):.2f}   "
              f"1-F slope = {slope:.2f}  (SQL: -0.5)")


if __name__ == "__main__":
    main()

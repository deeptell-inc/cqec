"""Topological surface code (baseline) vs LearnCI on the four algorithm states.

Fair, matched-physical-noise comparison.  Same physical per-qubit depolarizing
rate p drives both:

  * Surface code [[d^2,1,d]] (baseline): each of the n logical qubits is encoded
    in a distance-d patch; residual per-logical-qubit depolarizing at rate
    p_L(d,p) ~ A (p/p_th)^{(d+1)/2}, p_th ~ 0.01 (sub-threshold scaling model;
    no stim/pymatching available).  Overhead: n*(2 d^2 - 1) physical qubits.
  * LearnCI: the bare n-qubit output suffers per-qubit depolarizing at rate p
    and is recovered by learned Pauli-eigenvalue inversion from N shots.
    Overhead: single copy + N measurement shots, no extra qubits.

CAVEAT (different paradigms): the surface code protects an UNKNOWN logical state
through arbitrary computation (fault tolerant); LearnCI only recovers a FINAL
output state from invertible, learnable noise (not fault tolerant).  This
compares them solely as final-state protectors at matched physical noise.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from algorithms_faithful import (
    make_qkan_faithful, make_qdrift_faithful, make_cfqpe_faithful, make_regev_faithful,
)
from min_qubits_perfect_fidelity import surface_code_logical_error_rate
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import pauli_matrix, psd_normalize

ALGOS = [
    ("QKAN  ", make_qkan_faithful), ("qDRIFT", make_qdrift_faithful),
    ("CF-QPE", make_cfqpe_faithful), ("Regev ", make_regev_faithful),
]


def pauli_stack(n):
    return np.stack([pauli_matrix(b, n) for b in range(4 ** n)])


def pauli_weight(b, n):
    w = 0
    for _ in range(n):
        if b & 0b11:
            w += 1
        b >>= 2
    return w


def per_qubit_depolarizing(rho, n, p):
    """Apply single-qubit depolarizing at rate p to each qubit independently."""
    out = rho
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i in range(n):
            X = pauli_matrix(1 * 4 ** i, n)
            Y = pauli_matrix(2 * 4 ** i, n)
            Z = pauli_matrix(3 * 4 ** i, n)
            out = (1 - p) * out + (p / 3) * (X @ out @ X + Y @ out @ Y + Z @ out @ Z)
    return out


def surface_fidelity(rho, n, d, p):
    p_L = surface_code_logical_error_rate(d, p)
    prot = per_qubit_depolarizing(rho, n, p_L)        # residual logical noise
    return fidelity(prot, rho), 2 * d * d - 1


def learnci_fidelity(rho, n, p, N, Ps, rng):
    """Recover per-qubit depolarizing(p) by finite-shot Pauli-eigenvalue inversion.

    Product-depolarizing eigenvalues lambda_b = (1-4p/3)^{wt(b)} (full support);
    estimate every lambda_b from N shots and rescale.  Diagonal inversion ->
    condition number 1; the only amplification is the contractivity 1/lambda_b.
    """
    noisy = per_qubit_depolarizing(rho, n, p)
    f = 1.0 - 4 * p / 3
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.einsum("bij,ji->b", Ps, noisy)
        lam_hat = np.empty(4 ** n)
        for b in range(4 ** n):
            lam = f ** pauli_weight(b, n)
            lam_hat[b] = lam + rng.normal(scale=np.sqrt(max(1 - lam * lam, 0) / N))
        lam_hat = np.where(np.abs(lam_hat) < 1e-3,
                           np.sign(lam_hat) * 1e-3 + (lam_hat == 0) * 1e-3, lam_hat)
        est = np.tensordot(r / lam_hat / (2 ** n), Ps, axes=([0], [0]))
    return fidelity(psd_normalize(est), rho)


def main():
    N = 16000
    print("Surface code [[d^2,1,d]] (sub-threshold model, p_th=0.01) vs LearnCI")
    print(f"matched physical per-qubit depolarizing rate p;  LearnCI shots N={N}\n")

    # ---- p-sweep on the hardest case (Regev, n=6) ----
    print("Regev (n=6):  fidelity vs physical error rate p")
    print(f"{'p':>7s}  {'F_noisy':>8s}  {'Surf d=3':>9s}  {'Surf d=5':>9s}  "
          f"{'Surf d=7':>9s}  {'LearnCI':>8s}")
    rho, d = make_regev_faithful(); n = 6; Ps = pauli_stack(n)
    for p in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        fno = fidelity(per_qubit_depolarizing(rho, n, p), rho)
        fs = {dd: surface_fidelity(rho, n, dd, p)[0] for dd in (3, 5, 7)}
        fl = np.mean([learnci_fidelity(rho, n, p, N, Ps, np.random.default_rng(k))
                      for k in range(5)])
        print(f"{p:>7.3f}  {fno:>8.4f}  {fs[3]:>9.4f}  {fs[5]:>9.4f}  "
              f"{fs[7]:>9.4f}  {fl:>8.4f}")

    # ---- cross-algorithm at p above threshold (p=0.05) ----
    p = 0.05
    print(f"\nAll algorithms at p={p} (above surface threshold 0.01):")
    print(f"{'algo':<7s} {'n':>2s} {'F_noisy':>8s}  {'Surf d=5':>9s} "
          f"{'(qubits)':>9s}  {'LearnCI':>8s} {'(qubits)':>10s}")
    for name, fn in ALGOS:
        rho, d = fn(); n = int(round(np.log2(d))); Ps = pauli_stack(n)
        fno = fidelity(per_qubit_depolarizing(rho, n, p), rho)
        fs5, nq = surface_fidelity(rho, n, 5, p)
        fl = np.mean([learnci_fidelity(rho, n, p, N, Ps, np.random.default_rng(k))
                      for k in range(5)])
        print(f"{name:<7s} {n:>2d} {fno:>8.4f}  {fs5:>9.4f} {n*nq:>9d}  "
              f"{fl:>8.4f} {'1 + shots':>10s}")


if __name__ == "__main__":
    main()

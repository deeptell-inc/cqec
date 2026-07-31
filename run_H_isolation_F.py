"""H (conditioning isolated from misspecification) and F (n=2 CI) refinements.

H: both methods invert their OWN noise with the EXACT (oracle) channel -- no
   learning, no model misspecification -- under identical density-matrix readout
   noise eps, matched per-coherence decoherence.  We then sweep Hilbert
   dimension d=2^n and show the conditioning of the inversion:
     d-level energy-gap dephasing : amplification exp(gamma*(d-1))  ~ exp(gamma*d)
     n-qubit Pauli dephasing      : amplification (1-2q)^{-n}       ~ poly(d)
   The d-level recovery collapses with d (conditioning ~ Hilbert dim); the
   n-qubit recovery is graceful (conditioning ~ qubit number = log d).

F: ideal-recovery SQL slope for the n=2 algorithm with many trials -> tight CI.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from algorithms_faithful import make_qkan_faithful, make_regev_faithful
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    pauli_matrix, pauli_eigenvalue, psd_normalize, learn_pauli_channel,
    apply_pauli_channel,
)

GAMMA = 0.30                      # d-level per-gap dephasing
Q = 0.5 * (1 - np.exp(-GAMMA))    # n-qubit per-qubit Z rate: (1-2q)=exp(-gamma) (matched)


def herm_noise(d, eps, rng):
    m = rng.normal(scale=eps, size=(d, d)) + 1j * rng.normal(scale=eps, size=(d, d))
    return (m + m.conj().T) / 2


def dlevel_recover(rho, d, eps, rng):
    idx = np.arange(d)
    mask = np.exp(-GAMMA * np.abs(idx[:, None] - idx[None, :]))
    noisy = rho * mask + herm_noise(d, eps, rng)
    inv = noisy * np.exp(GAMMA * np.abs(idx[:, None] - idx[None, :]))   # oracle inverse
    amp = float(np.exp(GAMMA * (d - 1)))
    return fidelity(psd_normalize(inv), rho), amp


def nqubit_recover(rho, n, eps, rng, Ps):
    d = 2 ** n
    # per-qubit Z dephasing (exact), then density-matrix readout noise
    noisy = rho
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i in range(n):
            Z = pauli_matrix(3 * 4 ** i, n)
            noisy = (1 - Q) * noisy + Q * (Z @ noisy @ Z)
    noisy = noisy + herm_noise(d, eps, rng)
    # oracle Pauli eigenvalues lambda_b = (1-2q)^{#qubits with X/Y in b}
    def wtXY(b):
        c = 0
        for _ in range(n):
            c += 1 if (b & 0b11) in (1, 2) else 0
            b >>= 2
        return c
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.einsum("bij,ji->b", Ps, noisy)
        lam = np.array([(1 - 2 * Q) ** wtXY(b) for b in range(4 ** n)])
        lam = np.where(lam < 1e-9, 1e-9, lam)
        est = np.tensordot(r / lam / d, Ps, axes=([0], [0]))
    amp = float((1 - 2 * Q) ** (-n))
    return fidelity(psd_normalize(est), rho), amp


def part_H():
    print("=" * 76)
    print("H  Conditioning isolated (oracle channels, identical readout noise eps)")
    print(f"   matched decoherence: d-level gamma={GAMMA}, n-qubit q={Q:.3f} "
          f"((1-2q)=exp(-gamma)={np.exp(-GAMMA):.3f}); eps=0.01")
    print("=" * 76)
    eps = 0.01
    print(f"{'n':>2s} {'d':>4s}   {'d-level F':>9s} {'amp~exp(gd)':>12s}   "
          f"{'n-qubit F':>9s} {'amp~poly':>9s}")
    for n in [2, 3, 4, 5, 6]:
        d = 2 ** n
        Ps = np.stack([pauli_matrix(b, n) for b in range(4 ** n)])
        # generic pure state in this dimension
        rng0 = np.random.default_rng(n)
        psi = rng0.normal(size=d) + 1j * rng0.normal(size=d); psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        fd = np.mean([dlevel_recover(rho, d, eps, np.random.default_rng(k))[0] for k in range(8)])
        ampd = dlevel_recover(rho, d, eps, np.random.default_rng(0))[1]
        fq = np.mean([nqubit_recover(rho, n, eps, np.random.default_rng(k), Ps)[0] for k in range(8)])
        ampq = nqubit_recover(rho, n, eps, np.random.default_rng(0), Ps)[1]
        print(f"{n:>2d} {d:>4d}   {fd:>9.4f} {ampd:>12.1f}   {fq:>9.4f} {ampq:>9.2f}")


def part_F():
    print("\n" + "=" * 76)
    print("F  n=2 SQL slope with many trials (tight CI), ideal shot-limited recovery")
    print("=" * 76)
    rho, d = make_qkan_faithful(); n = 2
    Ps = np.stack([pauli_matrix(b, n) for b in range(4 ** n)])
    ch = {3: 0.04, 1: 0.015, 3 * 4: 0.04, 1 * 4: 0.015}
    ch[0] = 1 - sum(ch.values())
    noisy = apply_pauli_channel(rho, ch, n)
    Ns = [250, 1000, 4000, 16000, 64000, 256000]
    n_trials = 80
    slopes = []
    for tr in range(n_trials):
        inf = []
        for N in Ns:
            r = np.random.default_rng(1234 * tr + N)
            info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
            lr = dict(info["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rr = np.einsum("bij,ji->b", Ps, noisy)
                lam = np.array([pauli_eigenvalue(lr, b, n) for b in range(4 ** n)])
                lam = np.where(np.abs(lam) < 1e-3, np.sign(lam) * 1e-3 + (lam == 0) * 1e-3, lam)
                est = np.tensordot(rr / lam / 2 ** n, Ps, axes=([0], [0]))
            inf.append(max(1 - fidelity(psd_normalize(est), rho), 1e-9))
        slopes.append(np.polyfit(np.log(Ns), np.log(inf), 1)[0])
    se = np.std(slopes) / np.sqrt(n_trials)
    print(f"n=2 QKAN: SQL slope = {np.mean(slopes):.3f} +/- {np.std(slopes):.3f} (std), "
          f"95% CI = [{np.mean(slopes)-1.96*se:.3f}, {np.mean(slopes)+1.96*se:.3f}]  "
          f"({n_trials} trials)")


if __name__ == "__main__":
    part_H()
    part_F()

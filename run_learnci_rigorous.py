"""Rigorous finite-shot LearnCI evaluation addressing the confidence gaps.

Improvements over run_learnci_finite_shots.py:
  C/D : faithful algorithm states (algorithms_faithful.py) -- the actual outputs.
  E/I : SPAM (eigenvalue attenuation) + gate-noise floor on recovery; a
        SPAM-robust (two-depth ratio) variant; real shots/SPAM via aer is in
        run_learnci_aer.py.
  F   : 6 shot budgets x many trials; per-trial SQL slope with mean +- std (CI).
  H   : the SAME n-qubit Pauli noise recovered by both the n-qubit Pauli LearnCI
        and the d-level energy-gap LearnCI (model-misspecified) -- head to head.
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
from blind_cqec import (
    fidelity, combined_noise, estimate_learned_inversion, icec_recover,
)
from blind_cqec.nqubit_learning import (
    learn_pauli_channel, apply_pauli_channel, pauli_matrix, pauli_eigenvalue,
    psd_normalize, pauli_generator_eigenvalue,
)

GZ, GX = 0.04, 0.015          # per-qubit Pauli decoherence (Z, X)
SPAM_Q = 0.04                 # SPAM attenuation factor A = 1 - 2q
GATE_NOISE = 0.01             # depolarizing floor modelling the recovery circuit
ALGOS = [
    ("QKAN  ", make_qkan_faithful),
    ("qDRIFT", make_qdrift_faithful),
    ("CF-QPE", make_cfqpe_faithful),
    ("Regev ", make_regev_faithful),
]


def paper_noise(n):
    ch = {}
    for i in range(n):
        ch[3 * 4 ** i] = GZ
        ch[1 * 4 ** i] = GX
    ch[0] = 1.0 - sum(ch.values())
    return ch


def pauli_stack(n):
    return np.stack([pauli_matrix(b, n) for b in range(4 ** n)])


def recover_fast(rho_noisy, rates, n, Pstack, gate_noise=0.0, lam_floor=1e-3):
    d = 2 ** n
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.einsum("bij,ji->b", Pstack, rho_noisy)
        lam = np.array([pauli_eigenvalue(rates, b, n) for b in range(4 ** n)])
        lam = np.where(np.abs(lam) < lam_floor, np.sign(lam) * lam_floor + (lam == 0) * lam_floor, lam)
        est = np.tensordot(r / lam / d, Pstack, axes=([0], [0]))
    est = psd_normalize(est)
    if gate_noise > 0:
        est = (1 - gate_noise) * est + gate_noise * np.eye(d) / d
    return est


def part1():
    print("=" * 88)
    print("PART 1  Faithful states, SPAM + gate noise, with SQL-slope CIs")
    print(f"        per-qubit Z@{GZ} X@{GX};  SPAM q={SPAM_Q} (A={1-2*SPAM_Q:.2f});  "
          f"gate-noise floor={GATE_NOISE}")
    print("=" * 88)
    Ns = [250, 1000, 4000, 16000, 64000, 256000]
    n_trials = 12
    A = 1.0 - 2 * SPAM_Q
    print("\n(i) ideal recovery (no floor): pure shot-limited -> SQL slope ~ -0.5")
    print("(ii) realistic: SPAM-robust + gate-noise floor -> fidelity saturates at the floor\n")
    print(f"{'algo':<7s} {'n':>2s} {'F_noisy':>7s}  {'F_ideal@256k':>12s}  "
          f"{'SQL slope (ideal)':>17s}   {'F_real@1k':>9s} {'F_real@256k':>11s}")
    print("-" * 90)
    for name, fn in ALGOS:
        rho, d = fn()
        n = int(round(np.log2(d)))
        Ps = pauli_stack(n)
        ch = paper_noise(n)
        noisy = apply_pauli_channel(rho, ch, n)
        f_noisy = fidelity(noisy, rho)
        ideal_by_N = {N: [] for N in Ns}
        real_by_N = {N: [] for N in Ns}
        ideal_slopes = []
        for tr in range(n_trials):
            inf_ideal = []
            for N in Ns:
                r = np.random.default_rng(977 * tr + N + d)
                # (i) ideal: no SPAM, no gate floor -> shot-limited
                ii = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
                li = dict(ii["rates"]); li[0] = max(1 - sum(li.values()), 0.0)
                fi = fidelity(recover_fast(noisy, li, n, Ps, gate_noise=0.0), rho)
                ideal_by_N[N].append(fi); inf_ideal.append(max(1 - fi, 1e-9))
                # (ii) realistic: SPAM-robust + gate-noise floor
                ir = learn_pauli_channel(ch, n, n_bell=N, n_shots=N,
                                         spam_atten=A, spam_robust=True, rng=r)
                lr = dict(ir["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
                fr = fidelity(recover_fast(noisy, lr, n, Ps, gate_noise=GATE_NOISE), rho)
                real_by_N[N].append(fr)
            ideal_slopes.append(np.polyfit(np.log(Ns), np.log(inf_ideal), 1)[0])
        print(f"{name:<7s} {n:>2d} {f_noisy:>7.4f}  {np.mean(ideal_by_N[256000]):>12.4f}  "
              f"{np.mean(ideal_slopes):>8.2f} +/- {np.std(ideal_slopes):.2f}   "
              f"{np.mean(real_by_N[1000]):>9.4f} {np.mean(real_by_N[256000]):>11.4f}")


def part2():
    print("\n" + "=" * 88)
    print("PART 2  Head-to-head under the SAME n-qubit Pauli noise (finite shots)")
    print("        n-qubit Pauli LearnCI (correct model) vs d-level energy-gap "
          "LearnCI (misspecified)")
    print("=" * 88)
    Ns = [1000, 16000, 256000]
    n_trials = 10
    print(f"\n{'algo':<7s} {'d':>3s}  {'method':<16s} " +
          "  ".join(f"N={N}".rjust(9) for N in Ns))
    print("-" * 80)
    for name, fn in ALGOS:
        rho, d = fn()
        n = int(round(np.log2(d)))
        Ps = pauli_stack(n)
        ch = paper_noise(n)
        rates = {a: c for a, c in ch.items() if a != 0}
        noisy = apply_pauli_channel(rho, ch, n)

        # d-level black box: the SAME Pauli noise as a semigroup e^{tL}
        Lam = np.array([pauli_generator_eigenvalue(rates, b, n) for b in range(4 ** n)])
        rvec = np.einsum("bij,ji->b", Ps, rho)

        def pauli_channel_at_time(_rho, t, _rvec=rvec):
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rr = np.einsum("bij,ji->b", Ps, _rho)
                out = np.tensordot(np.exp(t * Lam) * rr / (2 ** n), Ps, axes=([0], [0]))
            return out

        nq, dl = [], []
        for N in Ns:
            fq, fd = [], []
            for tr in range(n_trials):
                r1 = np.random.default_rng(11 * tr + N + d)
                info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r1)
                lr = dict(info["rates"]); lr[0] = max(1 - sum(lr.values()), 0.0)
                fq.append(fidelity(recover_fast(noisy, lr, n, Ps), rho))
                # d-level method on the same noise (its readout is noiseless here;
                # the failure is misspecification + exp(gamma*gap) conditioning)
                est = estimate_learned_inversion(noisy, pauli_channel_at_time, d=d)
                fd.append(fidelity(icec_recover(noisy, est), rho))
            nq.append(np.mean(fq)); dl.append(np.mean(fd))
        print(f"{name:<7s} {d:>3d}  {'n-qubit Pauli':<16s} " +
              "  ".join(f"{x:>9.4f}" for x in nq))
        print(f"{'':<7s} {'':>3s}  {'d-level (mis.)':<16s} " +
              "  ".join(f"{x:>9.4f}" for x in dl))


if __name__ == "__main__":
    part1()
    part2()

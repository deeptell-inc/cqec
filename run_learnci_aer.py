"""Circuit-level LearnCI: channel learning on qiskit-aer with real shots + SPAM.

Grounds the finite-shot / SPAM claims in an actual simulator rather than a
Gaussian model.  For the 2-qubit QKAN state we estimate every Pauli eigenvalue
of the decoherence channel from real aer shots under a NoiseModel with readout
(SPAM) error, do the (condition-number-1, full 4^n) Walsh-Hadamard inversion,
and recover the noisy state classically.

Pipeline = circuit-level estimation + classical inversion (the catalytic
recovery itself is not synthesised as a circuit -- E^{-1} is not CPTP).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, ReadoutError
from qiskit.quantum_info import DensityMatrix

from algorithms_faithful import make_qkan_faithful
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    pauli_digits, sign, pauli_eigenvalue, apply_pauli_channel,
    invert_pauli_channel, psd_normalize,
)

N = 2
GZ, GX = 0.04, 0.015
READOUT_Q = 0.03            # SPAM: symmetric readout error per qubit

LETTER = {0: "I", 1: "X", 2: "Y", 3: "Z"}


def my_index_to_qiskit_label(idx):
    """My base-4 index (qubit i at digit i) -> qiskit label (qubit 0 rightmost)."""
    digs = pauli_digits(idx, N)                       # (q0, q1, ...)
    return "".join(LETTER[digs[N - 1 - i]] for i in range(N))


def true_channel():
    ch = {}
    for i in range(N):
        ch[3 * 4 ** i] = GZ
        ch[1 * 4 ** i] = GX
    ch[0] = 1.0 - sum(ch.values())
    return ch


def channel_error(ch):
    return pauli_error([(my_index_to_qiskit_label(a), c) for a, c in ch.items()])


def prep_eigenstate(qc, b):
    digs = pauli_digits(b, N)
    for i in range(N):
        L = digs[i]
        if L == 1:      # X: |+>
            qc.h(i)
        elif L == 2:    # Y: |+i> = S H |0>
            qc.h(i); qc.s(i)
        # Z or I: |0>


def rotate_to_measure(qc, b):
    digs = pauli_digits(b, N)
    for i in range(N):
        L = digs[i]
        if L == 1:      # X -> H
            qc.h(i)
        elif L == 2:    # Y -> S^dagger then H
            qc.sdg(i); qc.h(i)


def estimate_eigenvalue_aer(sim, ch_err, b, shots, noise_model):
    digs = pauli_digits(b, N)
    active = [i for i in range(N) if digs[i] != 0]
    qc = QuantumCircuit(N, len(active))
    prep_eigenstate(qc, b)
    qc.append(ch_err.to_instruction(), range(N))     # apply decoherence channel
    rotate_to_measure(qc, b)
    for k, i in enumerate(active):
        qc.measure(i, k)
    res = sim.run(qc, shots=shots, noise_model=noise_model).result()
    counts = res.get_counts()
    exp = 0.0
    for bitstr, cnt in counts.items():
        parity = (-1) ** (bitstr.count("1"))
        exp += parity * cnt
    return exp / shots


def learn_via_aer(ch, shots, noise_model):
    sim = AerSimulator()
    ch_err = channel_error(ch)
    # full WHT: estimate all 15 non-identity eigenvalues, lambda_0 = 1
    lam = np.zeros(4 ** N)
    lam[0] = 1.0
    for b in range(1, 4 ** N):
        lam[b] = estimate_eigenvalue_aer(sim, ch_err, b, shots, noise_model)
    # inverse Walsh-Hadamard (orthogonal, cond 1): c_a = 4^-n sum_b lam_b (-1)^<a,b>
    c = {}
    for a in range(4 ** N):
        val = sum(lam[b] * sign(a, b, N) for b in range(4 ** N)) / 4 ** N
        if a != 0:
            c[a] = max(val, 0.0)
    return c


def consistency_check(ch):
    """Verify the aer pauli_error matches apply_pauli_channel (qubit convention)."""
    qc = QuantumCircuit(N)
    qc.h(0); qc.cx(0, 1)                       # a Bell state, generic
    rho0 = DensityMatrix(qc).data
    qc2 = qc.copy(); qc2.append(channel_error(ch).to_instruction(), range(N))
    rho_aer = DensityMatrix(qc2).data
    rho_ref = apply_pauli_channel(rho0, ch, N)
    return float(np.linalg.norm(rho_aer - rho_ref))


def main():
    ch = true_channel()
    diff = consistency_check(ch)
    print(f"channel consistency (aer vs density-matrix): ||.|| = {diff:.2e}  "
          f"({'OK' if diff < 1e-9 else 'MISMATCH'})\n")

    rho, _ = make_qkan_faithful()
    noisy = apply_pauli_channel(rho, ch, N)
    f_noisy = fidelity(noisy, rho)

    nm = NoiseModel()
    ro = ReadoutError([[1 - READOUT_Q, READOUT_Q], [READOUT_Q, 1 - READOUT_Q]])
    for q in range(N):
        nm.add_readout_error(ro, [q])

    print(f"QKAN n=2, decoherence Z@{GZ}/X@{GX}, aer readout error q={READOUT_Q}")
    print(f"F_noisy = {f_noisy:.4f}\n")
    print(f"{'shots':>8s}  {'F_LearnCI(aer)':>14s}   learned (Z0,X0,Z1,X1)")
    print("-" * 60)
    idxs = [3, 1, 3 * 4, 1 * 4]   # Z0, X0, Z1, X1
    for shots in [1024, 4096, 16384, 65536]:
        f_acc, rates_acc = [], []
        for trial in range(5):
            np.random.seed(trial)
            learned = learn_via_aer(ch, shots, nm)
            full = dict(learned); full[0] = max(1 - sum(learned.values()), 0.0)
            est = psd_normalize(invert_pauli_channel(noisy, full, N))
            f_acc.append(fidelity(est, rho))
            rates_acc.append([learned.get(i, 0.0) for i in idxs])
        rm = np.mean(rates_acc, axis=0)
        print(f"{shots:>8d}  {np.mean(f_acc):>14.4f}   "
              f"({rm[0]:.3f},{rm[1]:.3f},{rm[2]:.3f},{rm[3]:.3f})  true Z@{GZ} X@{GX}")


if __name__ == "__main__":
    main()

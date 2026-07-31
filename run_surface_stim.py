"""Real rotated surface code (stim + pymatching MWPM) vs LearnCI.

Replaces the analytic sub-threshold model with a genuine circuit-level surface
code memory experiment decoded by minimum-weight perfect matching, under
circuit-level depolarizing noise of strength p.  The decoded logical error rate
p_L(d,p) then drives the same per-logical-qubit residual-depolarizing fidelity
model used for LearnCI, at matched physical noise p.
"""
import os
import sys

import numpy as np
import stim
import pymatching

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from algorithms_faithful import (
    make_qkan_faithful, make_qdrift_faithful, make_cfqpe_faithful, make_regev_faithful,
)
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import pauli_matrix, psd_normalize

ALGOS = [
    ("QKAN  ", make_qkan_faithful), ("qDRIFT", make_qdrift_faithful),
    ("CF-QPE", make_cfqpe_faithful), ("Regev ", make_regev_faithful),
]


def _decode(circ, shots):
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
    pred = matching.decode_batch(dets)
    return float(np.mean(pred[:, 0] != obs[:, 0]))


def surface_pL_codecap(d, p, shots=100_000):
    """Code-capacity model: one round of data depolarizing, perfect syndromes.
    This is the FAIR match to LearnCI's single-shot recovery (threshold ~10%)."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=1, distance=d,
        before_round_data_depolarization=p,
    )
    return _decode(circ, shots)


def surface_pL_circuit(d, p, shots=100_000):
    """Full circuit-level noise over d syndrome rounds (realistic FT; threshold ~0.6%)."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=d, distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    return _decode(circ, shots)


def per_qubit_depolarizing(rho, n, p):
    out = rho
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i in range(n):
            X = pauli_matrix(1 * 4 ** i, n); Y = pauli_matrix(2 * 4 ** i, n); Z = pauli_matrix(3 * 4 ** i, n)
            out = (1 - p) * out + (p / 3) * (X @ out @ X + Y @ out @ Y + Z @ out @ Z)
    return out


def pauli_weight(b, n):
    w = 0
    for _ in range(n):
        w += 1 if (b & 0b11) else 0
        b >>= 2
    return w


def learnci_fidelity(rho, n, p, N, Ps, rng):
    noisy = per_qubit_depolarizing(rho, n, p)
    f = 1.0 - 4 * p / 3
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r = np.einsum("bij,ji->b", Ps, noisy)
        lam = np.array([f ** pauli_weight(b, n) for b in range(4 ** n)])
        lam_hat = lam + rng.normal(scale=np.sqrt(np.maximum(1 - lam ** 2, 0) / N))
        lam_hat = np.where(np.abs(lam_hat) < 1e-3, np.sign(lam_hat) * 1e-3 + (lam_hat == 0) * 1e-3, lam_hat)
        est = np.tensordot(r / lam_hat / 2 ** n, Ps, axes=([0], [0]))
    return fidelity(psd_normalize(est), rho)


def main():
    rho, _ = make_regev_faithful(); n = 6
    Ps = np.stack([pauli_matrix(b, n) for b in range(4 ** n)])

    for label, pL_fn, ps in [
        ("CODE-CAPACITY (fair single-shot match; data errors only, perfect syndrome)",
         surface_pL_codecap, [0.02, 0.05, 0.08, 0.11, 0.15]),
        ("CIRCUIT-LEVEL (realistic FT; d syndrome rounds, all-location noise)",
         surface_pL_circuit, [0.002, 0.005, 0.01, 0.02, 0.05]),
    ]:
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        pL = {(d, p): pL_fn(d, p) for p in ps for d in (3, 5, 7)}
        print(f"{'p':>7s}  " + "  ".join(f"pL(d={d})".rjust(9) for d in (3, 5, 7)))
        for p in ps:
            print(f"{p:>7.3f}  " + "  ".join(f"{pL[(d,p)]:>9.5f}" for d in (3, 5, 7)))
        print(f"\nRegev (n=6) logical fidelity vs LearnCI:")
        print(f"{'p':>7s}  {'F_noisy':>8s}  {'Surf3':>7s}  {'Surf5':>7s}  "
              f"{'Surf7':>7s}  {'LearnCI':>8s}")
        for p in ps:
            fno = fidelity(per_qubit_depolarizing(rho, n, p), rho)
            fs = {d: fidelity(per_qubit_depolarizing(rho, n, pL[(d, p)]), rho) for d in (3, 5, 7)}
            fl = np.mean([learnci_fidelity(rho, n, p, 16000, Ps, np.random.default_rng(k)) for k in range(5)])
            print(f"{p:>7.3f}  {fno:>8.4f}  {fs[3]:>7.4f}  {fs[5]:>7.4f}  "
                  f"{fs[7]:>7.4f}  {fl:>8.4f}")


if __name__ == "__main__":
    main()

"""Run the four attached papers' algorithm outputs under decoherence and
recover them with LearnCI (learned channel inversion, integration B).

Paper -> cqec benchmark algorithm state:
    2410.04435      QKAN                       -> make_qkan   (d=4)
    PRXQuantum.2.040305  qDRIFT (random PF)    -> make_qdrift (d=8)
    7qcr-znl2       control-free QPE           -> make_cfqpe  (d=16)
    2308.06572      Regev factoring            -> make_regev  (d=64)

Method = LearnCI: the noise channel is an unknown one-parameter semigroup
e^{tL} (dephasing -> depolarizing -> amplitude damping) given only as a black
box; LearnCI probes it at short Chebyshev times, learns the strengths, and
inverts.  Baselines: no correction, and model-free coherence maximization.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # cqec.algorithms
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))  # blind_cqec

from cqec.algorithms import make_qkan, make_qdrift, make_cfqpe, make_regev
from blind_cqec import (
    combined_noise, estimate_coherence_max, estimate_learned_inversion,
    icec_recover, fidelity,
)

# Decoherence semigroup rates: at t=1 -> gamma=1.0, p~0.15, gamma_ad~0.10.
GR, LA, KA = 1.0, 0.1625, 0.1054


def make_channel_at_time():
    def cat(rho, t):
        return combined_noise(
            rho, gamma=GR * t, p=1.0 - np.exp(-LA * t),
            gamma_ad=1.0 - np.exp(-KA * t),
        )
    return cat


ALGOS = [
    ("QKAN        (2410.04435)", make_qkan),
    ("qDRIFT      (PRXQ 2.040305)", make_qdrift),
    ("CF-QPE      (7qcr-znl2)", make_cfqpe),
    ("Regev       (2308.06572)", make_regev),
]


def main() -> None:
    cat = make_channel_at_time()
    print("Decoherence (t=1):  gamma=1.00  p=0.15  gamma_ad=0.10  "
          "(dephasing->depol->amp.damp.)\n")
    print(f"{'algorithm':<28s} {'d':>3s}  {'F_noisy':>8s}  {'F_CohMax':>9s}  "
          f"{'F_LearnCI':>10s}   learned (gamma,p,gamma_ad)")
    print("-" * 92)
    for name, fn in ALGOS:
        rho, d = fn()
        noisy = cat(rho, 1.0)
        f_noisy = fidelity(noisy, rho)
        f_cm = fidelity(icec_recover(noisy, estimate_coherence_max(noisy)), rho)
        est, info = estimate_learned_inversion(noisy, cat, d=d, return_info=True)
        f_learn = fidelity(icec_recover(noisy, est), rho)
        g, p, gad = info["strengths"]
        print(f"{name:<28s} {d:>3d}  {f_noisy:>8.4f}  {f_cm:>9.4f}  "
              f"{f_learn:>10.4f}   ({g:.3f}, {p:.3f}, {gad:.3f})")


def sweep() -> None:
    """LearnCI fidelity vs dephasing strength (threshold-free robustness)."""
    print("\nLearnCI fidelity vs dephasing strength gamma "
          "(p, gamma_ad fixed at t=1 levels):")
    gammas = [0.5, 1.0, 2.0, 4.0]
    print(f"{'algorithm':<28s} {'d':>3s}  " +
          "  ".join(f"g={g}".rjust(8) for g in gammas))
    print("-" * 72)
    for name, fn in ALGOS:
        rho, d = fn()
        cells = []
        for gval in gammas:
            def cat(r, t, gv=gval):
                return combined_noise(
                    r, gamma=gv * t, p=1.0 - np.exp(-LA * t),
                    gamma_ad=1.0 - np.exp(-KA * t))
            noisy = cat(rho, 1.0)
            est = estimate_learned_inversion(noisy, cat, d=d)
            cells.append(fidelity(icec_recover(noisy, est), rho))
        print(f"{name:<28s} {d:>3d}  " +
              "  ".join(f"{c:>8.4f}" for c in cells))


if __name__ == "__main__":
    main()
    sweep()

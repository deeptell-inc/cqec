#!/usr/bin/env python3
"""
benchmark_joint.py — CPTP joint-channel validation benchmarks.

Runs the genuine joint-space covariant recovery channel (cqec.joint.JointCQEC)
and reports, for each configuration:
  * F_noisy, F_rec (system marginal vs target)
  * F_cat (catalyst marginal vs input catalyst, by partial trace — NOT copied)
  * covariance residual ||[U, H_total]||
  * linearity residual
  * Choi minimum eigenvalue (complete positivity)
  * target-swap control: channel optimized for a WRONG target applied to the
    same input — reported fidelity vs the true target.

Output: results_joint.json + console table.
"""

import json
import time
import numpy as np

from cqec.core import (fidelity, dephasing_channel, depolarizing_channel,
                       l1_coherence)
from cqec.algorithms import make_qkan, make_qdrift
from cqec.catalyst import dd_twirl_pipeline
from cqec.joint import JointCQEC


def max_coherent(d):
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


def run_config(name, rho_target, d, noise_fn, noise_label,
               cat_label="ideal", maxiter=80, n_restarts=2, seed=42):
    rho_noisy = noise_fn(rho_target)
    if cat_label == "ideal":
        cat = max_coherent(d)
    else:  # DD+Twirl-prepared catalyst
        cat = dd_twirl_pipeline(max_coherent(d), d, gamma=2.0,
                                n_copies=8, n_dd=8)["rho_cat"]

    ch = JointCQEC(d)
    t0 = time.time()
    ch.optimize(rho_target, rho_noisy, cat,
                maxiter=maxiter, n_restarts=n_restarts, seed=seed)
    dt = time.time() - t0

    rs, rc = ch.apply(rho_noisy, cat)
    f_noisy = fidelity(rho_target, rho_noisy)
    f_rec = fidelity(rho_target, rs)
    f_cat = fidelity(cat, rc)

    lin = ch.linearity_residual(rho_target,
                                depolarizing_channel(rho_target, 0.8), cat)
    cov = ch.covariance_residual()
    choi = ch.choi_min_eigenvalue(cat)

    print(f"  {name:<22} {noise_label:<14} cat={cat_label:<8} "
          f"F_noisy={f_noisy:.4f} F_rec={f_rec:.4f} F_cat={f_cat:.4f} "
          f"cov={cov:.1e} lin={lin:.1e} choi={choi:+.1e} ({dt:.0f}s)")
    return {
        "name": name, "noise": noise_label, "catalyst": cat_label, "d": d,
        "F_noisy": f_noisy, "F_rec": f_rec, "F_cat_marginal": f_cat,
        "covariance_residual": cov, "linearity_residual": lin,
        "choi_min_eig": choi, "opt_seconds": dt,
    }


def target_swap_control(seed=42):
    """Channel optimized for the WRONG target: honest control experiment."""
    print("\n== Target-swap control (d = 4, dephasing gamma = 2) ==")
    rho_a, d = make_qkan()                      # true target A
    # wrong target B: max-coherent state with sign-flipped coherences
    amps = np.array([1.0, -0.5, 0.5, 0.0], dtype=complex)
    amps /= np.linalg.norm(amps)
    rho_b = np.outer(amps, amps.conj())

    rho_noisy = dephasing_channel(rho_a, 2.0)   # noise acted on A
    cat = max_coherent(d)

    ch_a = JointCQEC(d)
    ch_a.optimize(rho_a, rho_noisy, cat, maxiter=80, n_restarts=2, seed=seed)
    out_a, _ = ch_a.apply(rho_noisy, cat)

    ch_b = JointCQEC(d)
    ch_b.optimize(rho_b, rho_noisy, cat, maxiter=80, n_restarts=2, seed=seed)
    out_b, _ = ch_b.apply(rho_noisy, cat)

    res = {
        "F_noisy_vs_A": fidelity(rho_a, rho_noisy),
        "channelA_F_vs_A": fidelity(rho_a, out_a),
        "channelB_F_vs_A": fidelity(rho_a, out_b),
        "channelB_F_vs_B": fidelity(rho_b, out_b),
    }
    print(f"  channel optimized for A: F(out, A) = {res['channelA_F_vs_A']:.4f}")
    print(f"  channel optimized for B: F(out, A) = {res['channelB_F_vs_A']:.4f}"
          f"  (F(out, B) = {res['channelB_F_vs_B']:.4f})")
    print("  -> outputs differ with the optimization target; at application"
          " time each channel is a fixed linear map with no target input.")
    return res


if __name__ == "__main__":
    print("=" * 100)
    print("CPTP joint-channel validation (JointCQEC)")
    print("=" * 100)
    results = []

    rho_qkan, d4 = make_qkan()
    rho_qdrift, d8 = make_qdrift()

    print("\n== QKAN (d = 4, joint dim 64, 12 EC gates) ==")
    results.append(run_config("QKAN", rho_qkan, d4,
                              lambda r: dephasing_channel(r, 2.0),
                              "deph g=2"))
    results.append(run_config("QKAN", rho_qkan, d4,
                              lambda r: depolarizing_channel(r, 0.3),
                              "depol p=0.3"))
    results.append(run_config("QKAN", rho_qkan, d4,
                              lambda r: dephasing_channel(r, 2.0),
                              "deph g=2", cat_label="ddtwirl"))

    print("\n== qDRIFT (d = 8, joint dim 256, 21 EC gates) ==")
    results.append(run_config("qDRIFT", rho_qdrift, d8,
                              lambda r: dephasing_channel(r, 2.0),
                              "deph g=2", maxiter=60))
    results.append(run_config("qDRIFT", rho_qdrift, d8,
                              lambda r: depolarizing_channel(r, 0.3),
                              "depol p=0.3", maxiter=60))

    swap = target_swap_control()

    with open("results_joint.json", "w") as f:
        json.dump({"configs": results, "target_swap": swap}, f, indent=2)
    print("\nSaved results_joint.json")

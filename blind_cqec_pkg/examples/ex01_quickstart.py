"""
Quickstart: reproduce the numbers shown in the README.
Run from the package root:  python examples/ex01_quickstart.py
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_naive, estimate_coherence_max, estimate_channel_inversion,
    icec_recover, fidelity,
)


def main() -> None:
    rng = np.random.default_rng(42)
    d = 8
    target = haar_random_pure(d, rng=rng)
    noisy = combined_noise(target, gamma=1.0, p=0.15, gamma_ad=0.1)

    print(f"=== Blind CQEC quickstart (d = {d}) ===")
    print(f"  Noisy fidelity:           {fidelity(noisy, target):.4f}")

    for name, est in [
        ("Naive",                estimate_naive(noisy)),
        ("Coherence max",        estimate_coherence_max(noisy)),
        ("Channel inversion",    estimate_channel_inversion(
            noisy, gamma=1.0, p=0.15, gamma_ad=0.1)),
    ]:
        rec = icec_recover(noisy, est)
        print(f"  {name:22s}:  F = {fidelity(rec, target):.4f}")


if __name__ == "__main__":
    main()

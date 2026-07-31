"""
Reproduce the Haar-random dimension sweep (paper Fig. dimension_sweep).
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_naive, estimate_coherence_max, estimate_channel_inversion,
    icec_recover, fidelity,
)


def main() -> None:
    dims = [2, 4, 8, 16, 32, 64]
    n_samples = 20

    print(f"{'d':>4s}  {'NoCor':>8s}  {'Naive':>8s}  {'CohMax':>8s}  {'ChInv':>8s}")
    for d in dims:
        rng = np.random.default_rng(42 + d)
        buckets = {"nocor": [], "naive": [], "cohmax": [], "chinv": []}
        for _ in range(n_samples):
            target = haar_random_pure(d, rng=rng)
            noisy = combined_noise(target, gamma=1.0, p=0.15, gamma_ad=0.1)

            buckets["nocor"].append(fidelity(noisy, target))
            buckets["naive"].append(
                fidelity(icec_recover(noisy, estimate_naive(noisy)), target))
            buckets["cohmax"].append(
                fidelity(icec_recover(noisy, estimate_coherence_max(noisy)), target))
            buckets["chinv"].append(
                fidelity(icec_recover(noisy,
                    estimate_channel_inversion(noisy, gamma=1.0, p=0.15, gamma_ad=0.1)),
                    target))

        print(f"{d:>4d}  "
              f"{np.mean(buckets['nocor']):>8.4f}  "
              f"{np.mean(buckets['naive']):>8.4f}  "
              f"{np.mean(buckets['cohmax']):>8.4f}  "
              f"{np.mean(buckets['chinv']):>8.4f}")


if __name__ == "__main__":
    main()

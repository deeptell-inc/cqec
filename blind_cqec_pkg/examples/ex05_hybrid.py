"""
Reproduce the hybrid strategy crossover (paper Fig. hybrid).

Sweeps the mixing weight w in [0, 1] at each dimension and reports
the optimal weight w*.
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_hybrid, icec_recover, fidelity,
)


def main() -> None:
    dims = [4, 8, 16, 32, 64]
    weights = np.linspace(0.0, 1.0, 11)
    n_samples = 8

    print(f"{'d':>4s}  {'w*':>6s}  {'F*':>8s}  (weights 0.0..1.0 step 0.1)")
    for d in dims:
        means = []
        for w in weights:
            rng = np.random.default_rng(42 + d)
            fs = []
            for _ in range(n_samples):
                target = haar_random_pure(d, rng=rng)
                noisy = combined_noise(target, gamma=1.0, p=0.15, gamma_ad=0.1)
                est = estimate_hybrid(
                    noisy, weight=float(w),
                    gamma=1.0, p=0.15, gamma_ad=0.1,
                )
                fs.append(fidelity(icec_recover(noisy, est), target))
            means.append(float(np.mean(fs)))
        idx = int(np.argmax(means))
        print(f"{d:>4d}  {weights[idx]:>6.2f}  {means[idx]:>8.4f}   "
              f"{[f'{m:.3f}' for m in means]}")


if __name__ == "__main__":
    main()

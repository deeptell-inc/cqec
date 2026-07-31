"""
Reproduce the noise-parameter sensitivity analysis (paper Fig. sensitivity).

Sweeps relative error delta applied to all three noise parameters
simultaneously, at selected dimensions.
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_channel_inversion, icec_recover, fidelity,
)


def main() -> None:
    dims = [4, 8, 16, 64]
    deltas = [-0.30, -0.10, 0.00, 0.10, 0.30]
    n_samples = 10
    gamma_true, p_true, gad_true = 1.0, 0.15, 0.1

    # header
    print(f"{'d / delta':>10s}", end="")
    for delta in deltas:
        print(f"  {delta:>+6.0%}", end="")
    print()

    for d in dims:
        print(f"{d:>10d}", end="")
        for delta in deltas:
            rng = np.random.default_rng(42 + d)
            fs = []
            for _ in range(n_samples):
                target = haar_random_pure(d, rng=rng)
                noisy = combined_noise(target, gamma_true, p_true, gad_true)
                est = estimate_channel_inversion(
                    noisy,
                    gamma=gamma_true * (1 + delta),
                    p=p_true * (1 + delta),
                    gamma_ad=gad_true * (1 + delta),
                )
                rec = icec_recover(noisy, est)
                fs.append(fidelity(rec, target))
            print(f"  {np.mean(fs):>6.3f}", end="")
        print()


if __name__ == "__main__":
    main()

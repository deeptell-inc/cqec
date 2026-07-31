"""
Reproduce the QEM comparison (paper Fig. qem_compare).

Compares blind CQEC (coherence max, channel inversion) with:
- no correction
- naive (uses noisy state as target; equivalent to no correction)

Note: full ZNE / virtual distillation comparisons from the paper require
the circuit-level scripts in the repository (benchmark_blind_qem_compare.py);
here we focus on the blind-CQEC-specific strategies.
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_coherence_max, estimate_channel_inversion,
    icec_recover, fidelity,
)


def main() -> None:
    dims = [4, 8, 16, 32, 64]
    n_samples = 15

    print(f"{'d':>4s}  {'NoCor':>8s}  {'BlindCM':>8s}  {'BlindCI':>8s}")
    for d in dims:
        rng = np.random.default_rng(42 + d)
        nocor, cm, ci = [], [], []
        for _ in range(n_samples):
            target = haar_random_pure(d, rng=rng)
            noisy = combined_noise(target, gamma=1.0, p=0.15, gamma_ad=0.1)
            nocor.append(fidelity(noisy, target))
            cm.append(fidelity(
                icec_recover(noisy, estimate_coherence_max(noisy)), target))
            ci.append(fidelity(
                icec_recover(noisy,
                    estimate_channel_inversion(noisy, gamma=1.0, p=0.15, gamma_ad=0.1)),
                target))
        print(f"{d:>4d}  {np.mean(nocor):>8.4f}  {np.mean(cm):>8.4f}  {np.mean(ci):>8.4f}")


if __name__ == "__main__":
    main()

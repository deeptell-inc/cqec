"""Gap-adaptive learned hybrid: an honest negative result.

Motivation: the learned channel inversion (ex06) inverts dephasing by
multiplying each coherence at gap g by exp(gamma*g).  We hypothesized that
blending toward coherence maximization at large gaps (where that factor
amplifies error) would be more robust --- `estimate_learned_hybrid`.

Result: it is NOT.  This script documents why.

  (1) Noiseless / high-precision read-out:
      Learned inversion already dominates coherence maximization at ALL
      dimensions, so the model-free fallback has nothing to add; every finite
      amp_cap only discards good inverted information.  amp_cap = inf (pure
      inversion) is best.  Notably this *removes* the coherence-max <-> inversion
      crossover that motivated the original blind-CQEC hybrid: once the channel
      is learned accurately, inversion just wins.

  (2) The gap-adaptive blend is dominated by pure inversion.

The honest takeaway: for blind CQEC, learn the channel and invert
(estimate_learned_inversion).  The remaining hard regime (finite statistics at
large d) is the standard-quantum-limit / inversion-conditioning wall, which no
blend repairs.
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_coherence_max,
    estimate_learned_inversion, estimate_learned_hybrid,
    icec_recover, fidelity,
)

GAMMA_RATE, LAMBDA, KAPPA = 1.0, 0.1625, 0.1054


def channel_at_time(rho, t):
    return combined_noise(
        rho,
        gamma=GAMMA_RATE * t,
        p=1.0 - np.exp(-LAMBDA * t),
        gamma_ad=1.0 - np.exp(-KAPPA * t),
    )


def main() -> None:
    dims = [4, 8, 16, 32]
    n_samples = 15
    caps = [4.0, 8.0, np.inf]

    print("Noiseless read-out: learned inversion dominates; the gap-adaptive")
    print("blend (finite amp_cap) only hurts. amp_cap=inf == pure inversion.\n")
    header = (f"{'d':>4s}  {'BlindCM':>8s}  {'LearnCI':>8s}  "
              + "  ".join(f"Hyb(cap={c})".rjust(12)
                          for c in ('4', '8', 'inf')))
    print(header)
    for d in dims:
        rng = np.random.default_rng(42 + d)
        cm, ci = [], []
        hyb = {c: [] for c in caps}
        for _ in range(n_samples):
            target = haar_random_pure(d, rng=rng)
            noisy = channel_at_time(target, 1.0)
            cm.append(fidelity(
                icec_recover(noisy, estimate_coherence_max(noisy)), target))
            ci.append(fidelity(
                icec_recover(noisy, estimate_learned_inversion(
                    noisy, channel_at_time, d=d)), target))
            for c in caps:
                hyb[c].append(fidelity(
                    icec_recover(noisy, estimate_learned_hybrid(
                        noisy, channel_at_time, d=d, amp_cap=c)), target))
        row = (f"{d:>4d}  {np.mean(cm):>8.4f}  {np.mean(ci):>8.4f}  "
               + "  ".join(f"{np.mean(hyb[c]):>12.4f}" for c in caps))
        print(row)


if __name__ == "__main__":
    main()

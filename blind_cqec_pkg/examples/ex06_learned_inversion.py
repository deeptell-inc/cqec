"""Integration B: learned channel inversion for blind CQEC.

Closes the oracle gap in `estimate_channel_inversion` (which needs the noise
strengths handed to it) by *learning* the noise generator from black-box
short-time probes, using the Chebyshev-interpolation derivative primitive of
the recent Lindbladian-learning literature (arXiv:2606.20535, 2606.23652).

We compare, at several dimensions:
    NoCor   : no correction (fidelity of the noisy state)
    BlindCM : coherence maximization (no noise model)
    OracleCI: channel inversion with the TRUE strengths (upper baseline)
    LearnCI : channel inversion with LEARNED strengths (this work)

LearnCI should track OracleCI closely while using only black-box probes.
"""
import numpy as np

from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_coherence_max, estimate_channel_inversion,
    estimate_learned_inversion, learn_channel_rates,
    icec_recover, fidelity,
)

# True (unknown-to-the-learner) noise rates defining the semigroup.
GAMMA_RATE = 1.0    # dephasing strength at t=1
LAMBDA = 0.1625     # depolarizing: p(1) = 1 - e^{-LAMBDA} ~ 0.15
KAPPA = 0.1054      # amplitude damping: gamma_ad(1) = 1 - e^{-KAPPA} ~ 0.10


def make_channel_at_time(d):
    """Black-box noise semigroup e^{tL}; the learner sees only this."""
    def channel_at_time(rho, t):
        gamma = GAMMA_RATE * t
        p = 1.0 - np.exp(-LAMBDA * t)
        gamma_ad = 1.0 - np.exp(-KAPPA * t)
        return combined_noise(rho, gamma=gamma, p=p, gamma_ad=gamma_ad)
    return channel_at_time


def main() -> None:
    dims = [4, 8, 16, 32]
    n_samples = 15

    # true strengths at t = 1 (for the oracle baseline and for reporting)
    p_true = 1.0 - np.exp(-LAMBDA)
    gad_true = 1.0 - np.exp(-KAPPA)
    print(f"true strengths @t=1:  gamma={GAMMA_RATE:.4f}  p={p_true:.4f}  "
          f"gamma_ad={gad_true:.4f}\n")

    print(f"{'d':>4s}  {'NoCor':>8s}  {'BlindCM':>8s}  {'OracleCI':>9s}  "
          f"{'LearnCI':>8s}  {'cond(A)':>8s}  learned (gamma,p,gamma_ad)")
    for d in dims:
        rng = np.random.default_rng(42 + d)
        channel_at_time = make_channel_at_time(d)

        # learn the rates ONCE per dimension (channel is state-independent)
        info = learn_channel_rates(channel_at_time, d)
        g_l, p_l, gad_l = info["strengths"]

        nocor, cm, oci, lci = [], [], [], []
        for _ in range(n_samples):
            target = haar_random_pure(d, rng=rng)
            noisy = channel_at_time(target, 1.0)
            nocor.append(fidelity(noisy, target))
            cm.append(fidelity(
                icec_recover(noisy, estimate_coherence_max(noisy)), target))
            oci.append(fidelity(
                icec_recover(noisy, estimate_channel_inversion(
                    noisy, gamma=GAMMA_RATE, p=p_true, gamma_ad=gad_true)),
                target))
            lci.append(fidelity(
                icec_recover(noisy, estimate_learned_inversion(
                    noisy, channel_at_time, d=d)),
                target))
        print(f"{d:>4d}  {np.mean(nocor):>8.4f}  {np.mean(cm):>8.4f}  "
              f"{np.mean(oci):>9.4f}  {np.mean(lci):>8.4f}  "
              f"{info['cond']:>8.2f}  "
              f"({g_l:.3f}, {p_l:.3f}, {gad_l:.3f})")


if __name__ == "__main__":
    main()

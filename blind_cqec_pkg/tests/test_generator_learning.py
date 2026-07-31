"""Tests for generator learning (blind channel inversion, integration B)."""
import numpy as np
import pytest

from blind_cqec import (
    haar_random_pure, combined_noise,
    learn_channel_rates, estimate_learned_inversion, estimate_learned_hybrid,
    chebyshev_time_nodes, estimate_coherence_max, icec_recover, fidelity,
)

# semigroup rates -> strengths at t=1
GAMMA_RATE, LAMBDA, KAPPA = 1.0, 0.1625, 0.1054


def _channel_at_time(rho, t):
    return combined_noise(
        rho,
        gamma=GAMMA_RATE * t,
        p=1.0 - np.exp(-LAMBDA * t),
        gamma_ad=1.0 - np.exp(-KAPPA * t),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_cheb_nodes_span_interval():
    nodes = chebyshev_time_nodes(0.2, 6)
    assert nodes[0] == 0.0
    assert np.isclose(nodes[-1], 0.2)
    assert np.all(np.diff(nodes) > 0)


@pytest.mark.parametrize("d", [4, 8, 16])
def test_learns_true_rates(d):
    info = learn_channel_rates(_channel_at_time, d)
    gamma, p, gamma_ad = info["strengths"]
    assert abs(gamma - GAMMA_RATE) < 0.02
    assert abs(p - (1 - np.exp(-LAMBDA))) < 0.02
    assert abs(gamma_ad - (1 - np.exp(-KAPPA))) < 0.02
    assert info["cond"] < 50.0  # well-conditioned inversion


@pytest.mark.parametrize("d", [4, 8, 16])
def test_learned_inversion_beats_coherence_max(d, rng):
    fs_learn, fs_cm = [], []
    for _ in range(8):
        target = haar_random_pure(d, rng=rng)
        noisy = _channel_at_time(target, 1.0)
        est = estimate_learned_inversion(noisy, _channel_at_time, d=d)
        fs_learn.append(fidelity(icec_recover(noisy, est), target))
        fs_cm.append(fidelity(
            icec_recover(noisy, estimate_coherence_max(noisy)), target))
    assert np.mean(fs_learn) >= np.mean(fs_cm) - 1e-3
    assert np.mean(fs_learn) > 0.95


def test_return_info_shape(rng):
    target = haar_random_pure(4, rng=rng)
    noisy = _channel_at_time(target, 1.0)
    est, info = estimate_learned_inversion(
        noisy, _channel_at_time, d=4, return_info=True)
    assert est.shape == (4, 4)
    assert set(info) >= {"rates", "strengths", "residual", "cond"}


@pytest.mark.parametrize("d", [4, 8, 16])
def test_hybrid_large_cap_reduces_to_inversion(d, rng):
    """amp_cap -> infinity recovers pure learned inversion."""
    target = haar_random_pure(d, rng=rng)
    noisy = _channel_at_time(target, 1.0)
    hyb = estimate_learned_hybrid(noisy, _channel_at_time, d=d, amp_cap=1e12)
    inv = estimate_learned_inversion(noisy, _channel_at_time, d=d)
    np.testing.assert_allclose(hyb, inv, atol=1e-6)


def test_hybrid_returns_valid_state(rng):
    target = haar_random_pure(8, rng=rng)
    noisy = _channel_at_time(target, 1.0)
    est = estimate_learned_hybrid(noisy, _channel_at_time, d=8, amp_cap=8.0)
    assert np.isclose(np.trace(est).real, 1.0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh((est + est.conj().T) / 2) > -1e-10)

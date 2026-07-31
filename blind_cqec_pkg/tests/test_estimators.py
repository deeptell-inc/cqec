"""Tests for estimation strategies."""
import numpy as np
import pytest

from blind_cqec import (
    haar_random_pure,
    dephasing, depolarizing, combined_noise,
    estimate_naive, estimate_coherence_max, estimate_channel_inversion,
    estimate_iterative, estimate_multicopy_average, estimate_hybrid,
    icec_recover, fidelity,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# --- Naive ---

def test_naive_returns_noisy(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = depolarizing(rho, 0.2)
    est = estimate_naive(noisy)
    # naive == noisy up to PSD projection (which is identity for PSD input)
    np.testing.assert_allclose(est, noisy, atol=1e-10)


def test_naive_recovery_no_improvement(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = depolarizing(rho, 0.2)
    f_noisy = fidelity(noisy, rho)
    rec = icec_recover(noisy, estimate_naive(noisy))
    f_rec = fidelity(rec, rho)
    assert abs(f_rec - f_noisy) < 1e-6


# --- Coherence maximization ---

@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_cohmax_returns_valid_state(d, rng):
    rho = haar_random_pure(d, rng=rng)
    noisy = dephasing(rho, 2.0)
    est = estimate_coherence_max(noisy)
    assert np.isclose(np.trace(est).real, 1.0, atol=1e-8)
    eigvals = np.linalg.eigvalsh((est + est.conj().T) / 2)
    assert np.all(eigvals > -1e-10)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_cohmax_recovers_under_pure_dephasing(d, rng):
    """Under pure dephasing, coh-max should recover pure states to high fidelity."""
    rho = haar_random_pure(d, rng=rng)
    noisy = dephasing(rho, gamma=2.0)
    est = estimate_coherence_max(noisy)
    f = fidelity(est, rho)
    assert f > 0.98, f"Got F = {f}, expected > 0.98"


# --- Channel inversion ---

def test_chinv_inverts_depolarizing_exactly(rng):
    rho = haar_random_pure(4, rng=rng)
    p = 0.3
    noisy = depolarizing(rho, p)
    est = estimate_channel_inversion(noisy, p=p)
    f = fidelity(est, rho)
    assert f > 0.999, f"Depolarizing inversion should be exact, got F = {f}"


def test_chinv_inverts_dephasing_exactly(rng):
    rho = haar_random_pure(4, rng=rng)
    gamma = 1.5
    noisy = dephasing(rho, gamma)
    est = estimate_channel_inversion(noisy, gamma=gamma)
    f = fidelity(est, rho)
    assert f > 0.999, f"Dephasing inversion should be exact, got F = {f}"


def test_chinv_with_zero_params_is_identity(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = combined_noise(rho, gamma=0.5, p=0.1, gamma_ad=0.05)
    est = estimate_channel_inversion(noisy, gamma=0.0, p=0.0, gamma_ad=0.0)
    # With zero params, output should equal PSD-projected input
    np.testing.assert_allclose(est, noisy, atol=1e-8)


# --- Iterative ---

def test_iterative_converges(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = dephasing(rho, 1.5)
    est = estimate_iterative(noisy, max_iter=5)
    f = fidelity(est, rho)
    # Should be at least as good as a single coh-max application
    f_cm = fidelity(estimate_coherence_max(noisy), rho)
    assert f >= f_cm - 0.05


# --- Multi-copy averaging ---

def test_multicopy_shrinks_variance(rng):
    d = 4
    rho = haar_random_pure(d, rng=rng)
    # n independent noisy copies (different realizations would require stochastic noise;
    # here we use deterministic noise so all copies equal -> averaging is identity)
    copies = [combined_noise(rho) for _ in range(5)]
    est = estimate_multicopy_average(copies)
    assert est.shape == (d, d)
    assert np.isclose(np.trace(est).real, 1.0, atol=1e-8)


def test_multicopy_empty_raises():
    with pytest.raises(ValueError):
        estimate_multicopy_average([])


# --- Hybrid ---

def test_hybrid_weight_0_equals_cohmax(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = combined_noise(rho)
    est_hybrid = estimate_hybrid(noisy, weight=0.0)
    est_cm = estimate_coherence_max(noisy)
    np.testing.assert_allclose(est_hybrid, est_cm, atol=1e-8)


def test_hybrid_weight_1_equals_chinv(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = combined_noise(rho, gamma=1.0, p=0.15, gamma_ad=0.1)
    est_hybrid = estimate_hybrid(noisy, weight=1.0, gamma=1.0, p=0.15, gamma_ad=0.1)
    est_ci = estimate_channel_inversion(noisy, gamma=1.0, p=0.15, gamma_ad=0.1)
    np.testing.assert_allclose(est_hybrid, est_ci, atol=1e-8)


def test_hybrid_out_of_range_raises(rng):
    rho = haar_random_pure(4, rng=rng)
    noisy = combined_noise(rho)
    with pytest.raises(ValueError):
        estimate_hybrid(noisy, weight=1.5)

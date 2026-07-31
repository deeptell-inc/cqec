"""Tests for noise channels: trace preservation, positivity, limiting cases."""
import numpy as np
import pytest

from blind_cqec import (
    dephasing, depolarizing, amplitude_damping, combined_noise,
    haar_random_pure,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_dephasing_trace_preserving(d, rng):
    rho = haar_random_pure(d, rng=rng)
    out = dephasing(rho, gamma=1.0)
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-10)
    # Diagonal unchanged
    np.testing.assert_allclose(np.diag(out).real, np.diag(rho).real, atol=1e-10)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_dephasing_limits(d, rng):
    rho = haar_random_pure(d, rng=rng)
    # gamma=0 is identity
    np.testing.assert_allclose(dephasing(rho, 0.0), rho, atol=1e-10)
    # large gamma kills off-diagonals
    out = dephasing(rho, gamma=50.0)
    off = out - np.diag(np.diag(out))
    assert np.max(np.abs(off)) < 1e-10


@pytest.mark.parametrize("d,p", [(2, 0.3), (4, 0.5), (8, 0.15)])
def test_depolarizing_properties(d, p, rng):
    rho = haar_random_pure(d, rng=rng)
    out = depolarizing(rho, p)
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-10)
    eigvals = np.linalg.eigvalsh((out + out.conj().T) / 2)
    assert np.all(eigvals > -1e-12)
    # p=1 gives maximally mixed
    mm = depolarizing(rho, 1.0)
    np.testing.assert_allclose(mm, np.eye(d) / d, atol=1e-10)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_amplitude_damping_trace_preserving(d, rng):
    rho = haar_random_pure(d, rng=rng)
    out = amplitude_damping(rho, gamma=0.3)
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-8)


@pytest.mark.parametrize("d", [2, 4, 8])
def test_combined_noise_valid_density_matrix(d, rng):
    rho = haar_random_pure(d, rng=rng)
    out = combined_noise(rho, gamma=1.0, p=0.15, gamma_ad=0.1)
    # Trace one
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-8)
    # Hermitian
    np.testing.assert_allclose(out, out.conj().T, atol=1e-8)
    # PSD
    eigvals = np.linalg.eigvalsh((out + out.conj().T) / 2)
    assert np.all(eigvals > -1e-10)


def test_combined_reduces_fidelity(rng):
    rho = haar_random_pure(8, rng=rng)
    out = combined_noise(rho, gamma=1.0, p=0.15, gamma_ad=0.1)
    # Noisy state must not equal input
    assert np.linalg.norm(out - rho) > 0.1

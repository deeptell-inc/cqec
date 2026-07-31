"""Tests for the recovery primitives."""
import numpy as np
import pytest

from blind_cqec import icec_recover, psd_project, haar_random_pure


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_psd_project_valid_state_unchanged(rng):
    rho = haar_random_pure(4, rng=rng)
    out = psd_project(rho)
    np.testing.assert_allclose(out, rho, atol=1e-10)


def test_psd_project_clips_negatives():
    d = 3
    # Hermitian matrix with negative eigenvalue
    M = np.diag([0.6, 0.5, -0.1]).astype(complex)
    out = psd_project(M)
    assert np.isclose(np.trace(out).real, 1.0)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals >= -1e-12)


def test_psd_project_fallback_zero_trace():
    M = np.zeros((4, 4), dtype=complex)
    out = psd_project(M)
    # Should fall back to maximally mixed
    np.testing.assert_allclose(out, np.eye(4) / 4, atol=1e-10)


def test_icec_recover_returns_valid_state(rng):
    noisy = haar_random_pure(4, rng=rng)
    target = haar_random_pure(4, rng=rng)
    rec = icec_recover(noisy, target)
    assert np.isclose(np.trace(rec).real, 1.0, atol=1e-10)
    eigvals = np.linalg.eigvalsh((rec + rec.conj().T) / 2)
    assert np.all(eigvals > -1e-12)

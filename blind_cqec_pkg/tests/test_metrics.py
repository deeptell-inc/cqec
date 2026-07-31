"""Tests for fidelity and coherence metrics."""
import numpy as np
import pytest

from blind_cqec import fidelity, trace_distance, l1_coherence, haar_random_pure


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_fidelity_identity(d, rng):
    rho = haar_random_pure(d, rng=rng)
    assert abs(fidelity(rho, rho) - 1.0) < 1e-8


def test_fidelity_orthogonal():
    d = 4
    v1 = np.zeros(d); v1[0] = 1.0
    v2 = np.zeros(d); v2[1] = 1.0
    rho1 = np.outer(v1, v1)
    rho2 = np.outer(v2, v2)
    assert fidelity(rho1, rho2) < 1e-10


def test_fidelity_symmetric(rng):
    rho = haar_random_pure(6, rng=rng)
    sigma = haar_random_pure(6, rng=rng)
    assert abs(fidelity(rho, sigma) - fidelity(sigma, rho)) < 1e-8


def test_fidelity_bounded(rng):
    for _ in range(10):
        rho = haar_random_pure(6, rng=rng)
        sigma = haar_random_pure(6, rng=rng)
        f = fidelity(rho, sigma)
        assert -1e-10 <= f <= 1.0 + 1e-10


def test_trace_distance_zero_for_equal(rng):
    rho = haar_random_pure(6, rng=rng)
    assert trace_distance(rho, rho) < 1e-10


def test_trace_distance_bounded(rng):
    rho = haar_random_pure(8, rng=rng)
    sigma = haar_random_pure(8, rng=rng)
    d = trace_distance(rho, sigma)
    assert 0.0 <= d <= 1.0 + 1e-10


def test_l1_coherence_diagonal_zero():
    d = 4
    rho = np.diag([0.25, 0.25, 0.25, 0.25])
    assert l1_coherence(rho) < 1e-12


def test_l1_coherence_positive(rng):
    rho = haar_random_pure(6, rng=rng)
    assert l1_coherence(rho) > 0

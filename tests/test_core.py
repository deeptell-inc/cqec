"""Tests for cqec.core module."""

import numpy as np
import pytest
from cqec.core import (
    fidelity, purity, l1_coherence, coherence_modes,
    mode_inclusion, dephasing_channel, depolarizing_channel,
    ensure_density_matrix, concurrence,
)


def test_fidelity_pure():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert abs(fidelity(rho, rho) - 1.0) < 1e-10


def test_fidelity_orthogonal():
    rho0 = np.diag([1.0, 0.0]).astype(complex)
    rho1 = np.diag([0.0, 1.0]).astype(complex)
    assert fidelity(rho0, rho1) < 1e-10


def test_purity_pure():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert abs(purity(rho) - 1.0) < 1e-10


def test_purity_mixed():
    rho = np.eye(4) / 4
    assert abs(purity(rho) - 0.25) < 1e-10


def test_l1_coherence_diagonal():
    rho = np.diag([0.5, 0.3, 0.2]).astype(complex)
    assert l1_coherence(rho) < 1e-10


def test_l1_coherence_plus():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    assert l1_coherence(rho) > 2.0


def test_dephasing_preserves_diagonal():
    rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
    rho_d = dephasing_channel(rho, 2.0)
    np.testing.assert_allclose(np.diag(rho_d), np.diag(rho))


def test_dephasing_reduces_offdiag():
    rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
    rho_d = dephasing_channel(rho, 1.0)
    assert np.abs(rho_d[0, 1]) < np.abs(rho[0, 1])


def test_depolarizing_identity():
    rho = np.diag([0.7, 0.3]).astype(complex)
    rho_d = depolarizing_channel(rho, 0.0)
    np.testing.assert_allclose(rho_d, rho)


def test_depolarizing_full():
    rho = np.diag([0.7, 0.3]).astype(complex)
    rho_d = depolarizing_channel(rho, 1.0)
    np.testing.assert_allclose(rho_d, np.eye(2) / 2, atol=1e-10)


def test_mode_inclusion_identical():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    assert mode_inclusion(rho, rho)


def test_mode_inclusion_dephased():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    rho_d = dephasing_channel(rho, 2.0)
    assert mode_inclusion(rho, rho_d)


def test_mode_inclusion_fails_diagonal():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    rho_diag = np.diag(np.diag(rho))
    assert not mode_inclusion(rho, rho_diag)


def test_concurrence_bell():
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    assert abs(concurrence(rho) - 1.0) < 1e-10


def test_concurrence_separable():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert concurrence(rho) < 1e-10


def test_ensure_density_matrix():
    M = np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)
    rho = ensure_density_matrix(M)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)
    assert abs(np.trace(rho) - 1.0) < 1e-10

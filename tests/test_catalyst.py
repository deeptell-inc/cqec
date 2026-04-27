"""Tests for cqec.catalyst module."""

import numpy as np
from cqec.core import fidelity, l1_coherence, depolarizing_channel
from cqec.catalyst import swap_test, recursive_swap, dd_twirl_pipeline


def test_swap_test_pure():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    out, p = swap_test(rho, rho, 4)
    assert abs(fidelity(rho, out) - 1.0) < 1e-6
    assert abs(p - 1.0) < 1e-6


def test_swap_test_depolarized():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    rho_n = depolarizing_channel(rho, 0.3)
    out, p = swap_test(rho_n, rho_n, 4)
    assert fidelity(rho, out) > fidelity(rho, rho_n)


def test_recursive_monotone():
    psi = np.ones(4, dtype=complex) / 2
    rho = np.outer(psi, psi.conj())
    rho_n = depolarizing_channel(rho, 0.3)
    fids = []
    for r in range(1, 5):
        out, _, _ = recursive_swap(rho_n, 4, r)
        fids.append(fidelity(rho, out))
    for i in range(len(fids) - 1):
        assert fids[i + 1] >= fids[i] - 1e-8


def test_dd_twirl_pipeline():
    d = 4
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    rho_ideal = np.outer(psi, psi.conj())
    result = dd_twirl_pipeline(rho_ideal, d, gamma=2.0, n_copies=8, n_dd=8)
    assert result['fidelity'] > 0.9
    assert result['coherence'] > 0

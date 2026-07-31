"""Tests for cqec.joint — the CPTP joint-space covariant channel.

These tests lock in the properties the adversarial review found missing
from the effective implementation:
  linearity, complete positivity, trace preservation, exact covariance,
  catalyst-by-partial-trace (not copied), and target-independence at
  application time.
"""

import numpy as np
import pytest

from cqec.core import fidelity, dephasing_channel, depolarizing_channel
from cqec.algorithms import make_qkan
from cqec.joint import (JointCQEC, build_joint_unitary, three_layer_pairs,
                        total_hamiltonian, ec_gate_qubits)


def _max_coherent(d):
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


@pytest.fixture(scope="module")
def optimized_channel():
    rho_t, d = make_qkan()
    rho_n = dephasing_channel(rho_t, 2.0)
    cat = _max_coherent(d)
    ch = JointCQEC(d)
    ch.optimize(rho_t, rho_n, cat, maxiter=40, n_restarts=1)
    return ch, rho_t, rho_n, cat


def test_ec_gate_unitary():
    U = ec_gate_qubits(3, 0, 2, 0.7)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(8), atol=1e-12)


def test_joint_unitary_covariant():
    """[U, H_total] = 0 exactly for random angles."""
    rng = np.random.default_rng(0)
    n_s = n_c = 2
    n_a = 2
    th = rng.uniform(-np.pi, np.pi, len(three_layer_pairs(n_s, n_c, n_a)))
    U = build_joint_unitary(n_s, n_c, n_a, th)
    H = total_hamiltonian(n_s + n_c + n_a)
    assert np.max(np.abs(U @ H - H @ U)) < 1e-12


def test_channel_linear(optimized_channel):
    """Lambda(mix) equals the mixture of outputs to machine precision."""
    ch, rho_t, rho_n, cat = optimized_channel
    A = rho_t
    B = depolarizing_channel(rho_t, 0.8)
    assert ch.linearity_residual(A, B, cat) < 1e-12


def test_channel_trace_preserving(optimized_channel):
    ch, rho_t, rho_n, cat = optimized_channel
    out, _ = ch.apply(rho_n, cat)
    assert abs(np.trace(out).real - 1.0) < 1e-10


def test_channel_completely_positive(optimized_channel):
    ch, *_ , cat = optimized_channel
    assert ch.choi_min_eigenvalue(cat) > -1e-8


def test_catalyst_from_partial_trace_not_copy(optimized_channel):
    """The catalyst marginal must come from the propagated joint state:
    for a generic shallow circuit it differs measurably from the input."""
    ch, rho_t, rho_n, cat = optimized_channel
    _, rc = ch.apply(rho_n, cat)
    # it is a valid state ...
    assert abs(np.trace(rc).real - 1.0) < 1e-10
    # ... and it is NOT byte-identical to the input (no copying).
    assert not np.allclose(rc, cat, atol=1e-12)


def test_recovery_improves_fidelity(optimized_channel):
    ch, rho_t, rho_n, cat = optimized_channel
    out, _ = ch.apply(rho_n, cat)
    assert fidelity(rho_t, out) > fidelity(rho_t, rho_n) + 0.05


def test_no_target_input_at_application(optimized_channel):
    """At application time the channel takes (rho, catalyst) only —
    the same fixed map acts on any input state."""
    ch, rho_t, rho_n, cat = optimized_channel
    other = depolarizing_channel(_max_coherent(ch.d), 0.5)
    out, _ = ch.apply(other, cat)   # no target argument exists
    assert abs(np.trace(out).real - 1.0) < 1e-10

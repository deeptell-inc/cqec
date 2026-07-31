"""Tests for n-qubit learning-augmented blind CQEC (Pauli basis)."""
import numpy as np
import pytest

from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    sign, anticommute_parity, pauli_matrix, pauli_eigenvalue, sign_matrix,
    learn_pauli_channel, apply_pauli_channel, invert_pauli_channel,
    psd_normalize, bell_sample_support,
    pauli_generator_eigenvalue, learn_pauli_generator,
)


def test_sign_commutation_basic():
    # XI vs IX commute; X vs Z (single qubit) anticommute
    n = 2
    XI, IX = 1, (1 << 2)  # X on q0, X on q1
    assert sign(XI, IX, n) == 1
    n1 = 1
    assert sign(1, 3, n1) == -1   # X vs Z anticommute
    assert sign(1, 0, n1) == 1    # X vs I commute


def test_full_wht_is_cond_one():
    """The full 4^n sign matrix H satisfies H^2 = 4^n I, hence cond(H) = 1.

    Use integer matmul: H^2 == 4^n I implies all singular values equal 2^n, so
    the condition number is exactly 1.  (Integer arithmetic also avoids spurious
    FP-exception warnings from the platform BLAS on a +-1 float matrix.)
    """
    n = 3
    full = list(range(4 ** n))
    H = sign_matrix(full, full, n).astype(np.int64)
    assert np.array_equal(H @ H, (4 ** n) * np.eye(4 ** n, dtype=np.int64))


def test_pauli_eigenvalue_matches_matrix():
    """lambda_b from rates equals the PTM diagonal computed via matrices."""
    n = 2
    ch = {0: 0.7, 1: 0.1, 11: 0.12, 6: 0.08}
    for b in [1, 6, 11, 5]:
        Pb = pauli_matrix(b, n)
        # E(P_b) = lambda_b P_b  ->  lambda_b = Tr(P_b E(P_b)) / 2^n
        EPb = sum(c * pauli_matrix(a, n) @ Pb @ pauli_matrix(a, n)
                  for a, c in ch.items())
        lam_mat = np.trace(Pb @ EPb).real / 2 ** n
        assert abs(lam_mat - pauli_eigenvalue(ch, b, n)) < 1e-9


def test_inversion_roundtrip_exact():
    n = 2
    rng = np.random.default_rng(0)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    ch = {0: 0.8, 1: 0.07, 11: 0.08, 6: 0.05}
    noisy = apply_pauli_channel(rho, ch, n)
    back = invert_pauli_channel(noisy, ch, n)
    assert np.linalg.norm(back - rho) < 1e-9


def test_learn_recovers_rates_noiseless():
    n = 3
    ch = {0: 0.7, 5: 0.1, 22: 0.12, 41: 0.08}
    info = learn_pauli_channel(ch, n, n_bell=200000, n_shots=10 ** 8,
                               rng=np.random.default_rng(1))
    for a in [5, 22, 41]:
        assert abs(info["rates"].get(a, 0.0) - ch[a]) < 1e-3


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_inversion_well_conditioned(n):
    """The over-determined WHT inversion has O(1) condition number, any n."""
    rng = np.random.default_rng(10 + n)
    cand = rng.choice(np.arange(1, 4 ** n), size=6, replace=False)
    ch = {0: 0.75}
    ch.update({int(a): 0.25 / 6 for a in cand})
    info = learn_pauli_channel(ch, n, n_bell=8000, n_shots=8000, rng=rng)
    assert info["cond"] < 5.0


def test_bell_sampling_finds_support():
    n = 4
    rng = np.random.default_rng(2)
    cand = rng.choice(np.arange(1, 4 ** n), size=5, replace=False)
    ch = {0: 0.8}
    ch.update({int(a): 0.2 / 5 for a in cand})
    counts = bell_sample_support(ch, 5000, rng)
    found = set(a for a in counts if a != 0)
    assert set(int(a) for a in cand) <= found


def test_sql_scaling_slope():
    """Coefficient error scales ~ N^{-1/2} (slope in [-0.65, -0.35])."""
    n = 4
    rng0 = np.random.default_rng(7)
    cand = rng0.choice(np.arange(1, 4 ** n), size=6, replace=False)
    ch = {0: 0.75}
    ch.update({int(a): 0.25 / 6 for a in cand})
    true_supp = set(int(a) for a in cand)
    Ns = [500, 4000, 32000]
    errs = []
    for N in Ns:
        acc = []
        for trial in range(8):
            r = np.random.default_rng(31 * trial + N)
            info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
            acc.append(sum(abs(info["rates"].get(a, 0.0) - ch.get(a, 0.0))
                           for a in true_supp | set(info["rates"])))
        errs.append(np.mean(acc))
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert -0.65 < slope < -0.35, f"slope {slope} not ~ -0.5"


def test_generator_eigenvalue_consistency():
    """Lambda_b = d/dt lambda_b(t)|_0 for lambda_b(t) = exp(t Lambda_b)."""
    n = 3
    rates = {5: 0.3, 22: 0.5, 41: 0.2}
    for b in [5, 22, 41, 1, 7]:
        Lam = pauli_generator_eigenvalue(rates, b, n)
        # finite-difference check of the derivative of exp(t*Lam) at 0
        dt = 1e-6
        fd = (np.exp(dt * Lam) - 1.0) / dt
        assert abs(fd - Lam) < 1e-3


def test_learn_generator_recovers_rates_noiseless():
    n = 4
    rng = np.random.default_rng(5)
    cand = rng.choice(np.arange(1, 4 ** n), size=5, replace=False)
    rates = {int(a): float(g) for a, g in zip(cand, rng.uniform(0.2, 1.0, size=5))}
    info = learn_pauli_generator(rates, n, n_bell=200000, n_shots=10 ** 7,
                                 t_max=0.1, rng=rng)
    for a in rates:
        assert abs(info["rates"].get(a, 0.0) - rates[a]) < 1e-2
    assert info["cond"] < 6.0

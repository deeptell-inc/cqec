r"""Learning-augmented blind CQEC on n qubits (Pauli basis, ansatz-free).

Lifts the d-level toy-model integration B to the n-qubit Pauli-channel setting,
putting the actual primitives of the recent Lindbladian-learning literature in
the estimation stage:

  * Bell sampling (arXiv:2606.20706) -> support learning.  The chi-matrix of a
    Pauli channel is diagonal, ``chi_{aa} = c_a``, so a Bell measurement samples
    the error ``P_a`` with probability ``c_a``.  This discovers the (unknown,
    sparse) support without ever solving a 4^n-dimensional system.
  * Walsh-Hadamard / PTM inversion (arXiv:2606.20535, 2606.23652) -> coefficient
    learning.  A Pauli channel has a diagonal PTM with eigenvalues
    ``lambda_b = sum_a c_a (-1)^{<a,b>}`` --- the Walsh-Hadamard transform of the
    error rates over (Z_2)^{2n}.  Its inverse is an *orthogonal* transform with
    condition number exactly 1, so coefficient recovery does not amplify the
    estimation error.

Why this matters: the d-level dephasing inversion amplified error by
``exp(gamma|i-j|)`` (the conditioning blow-up that wrecked high-d recovery in
ex07).  The Pauli/WHT structure removes that amplification entirely; the only
residual instability is the per-eigenvalue contraction ``1/lambda_b``.  What
remains is the genuine standard-quantum-limit (SQL) wall: estimating each
``lambda_b`` to precision ``eps`` costs ``Omega(1/eps^2)`` samples.

See ``../../Lnl/learning_augmented_cqec_formalization.md`` for the protocol and
the SQL-recovery theorem this module validates numerically.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
PauliChannel = Dict[int, float]  # {pauli_index: rate}, sparse


# --------------------------------------------------------------------------
# Pauli-string bookkeeping (index = n base-4 digits; 0=I,1=X,2=Y,3=Z).
# --------------------------------------------------------------------------
def pauli_digits(idx: int, n: int) -> Tuple[int, ...]:
    out = []
    for _ in range(n):
        out.append(idx & 0b11)
        idx >>= 2
    return tuple(out)


def anticommute_parity(a: int, b: int, n: int) -> int:
    """Parity (0/1) of the number of qubit positions where P_a, P_b anticommute.

    Two single-qubit Paulis anticommute iff both are non-identity and distinct.
    """
    da, db = pauli_digits(a, n), pauli_digits(b, n)
    ac = sum(1 for x, y in zip(da, db) if x != 0 and y != 0 and x != y)
    return ac & 1


def sign(a: int, b: int, n: int) -> int:
    """(-1)^{<a,b>} = +1 if P_a, P_b commute, -1 if they anticommute."""
    return -1 if anticommute_parity(a, b, n) else 1


# --------------------------------------------------------------------------
# Single-qubit / n-qubit Pauli matrices (only for small-n state-level demos).
# --------------------------------------------------------------------------
_PAULI_1Q = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(idx: int, n: int) -> Array:
    M = np.array([[1.0 + 0j]])
    for d in pauli_digits(idx, n):
        M = np.kron(M, _PAULI_1Q[d])
    return M


# --------------------------------------------------------------------------
# Pauli eigenvalues (Walsh-Hadamard transform of the rates) and inversion.
# --------------------------------------------------------------------------
def pauli_eigenvalue(channel: PauliChannel, b: int, n: int) -> float:
    """lambda_b = sum_a c_a (-1)^{<a,b>}."""
    return float(sum(c * sign(a, b, n) for a, c in channel.items()))


def sign_matrix(rows: Sequence[int], cols: Sequence[int], n: int) -> Array:
    """M[i,j] = (-1)^{<rows[i], cols[j]>}; the restricted WHT (orthogonal-ish)."""
    return np.array([[sign(b, a, n) for a in cols] for b in rows], dtype=float)


# --------------------------------------------------------------------------
# Stage 1a: Bell sampling -> support.
# --------------------------------------------------------------------------
def bell_sample_support(
    channel: PauliChannel, n_bell: int, rng: np.random.Generator
) -> Dict[int, int]:
    """Sample errors P_a ~ c_a (the Bell-sampling distribution) n_bell times.

    Returns empirical counts over the observed Paulis.  Identity (a = 0) is
    included; the support estimate is typically ``{a != 0 : count}`` thresholded.
    """
    paulis = np.array(list(channel.keys()))
    probs = np.array(list(channel.values()), dtype=float)
    probs = probs / probs.sum()
    draws = rng.choice(paulis, size=n_bell, p=probs)
    vals, counts = np.unique(draws, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------
# Stage 1b: estimate Pauli eigenvalues (with SQL shot noise) -> WHT inversion.
# --------------------------------------------------------------------------
def estimate_eigenvalue_noisy(
    lam_true: float, n_shots: int, rng: np.random.Generator
) -> float:
    """Finite-shot estimate of a Pauli eigenvalue.

    Measuring P_b on a P_b-eigenstate after the channel is a +-1 random variable
    with mean lambda_b; the empirical mean has std sqrt((1-lambda^2)/N) (SQL).
    """
    var = max(1.0 - lam_true ** 2, 0.0) / max(n_shots, 1)
    return float(lam_true + rng.normal(scale=np.sqrt(var)))


def _probe_set(support: Sequence[int], n: int, oversample: int,
               rng: np.random.Generator) -> list[int]:
    """Probe Paulis B for eigenvalue estimation: support + random extras.

    An *over-determined* set ( |B| >> |support| ) of random Paulis makes the
    sign matrix's columns near-orthogonal random +-1 vectors, so the WHT
    inversion has condition number ~1 --- unlike the singular square submatrix
    obtained from B = support alone.
    """
    target = max(oversample * (len(support) + 1), len(support) + 16)
    target = min(target, 4 ** n - 1)
    chosen = set(support)
    pool = [a for a in range(1, 4 ** n) if a not in chosen]
    rng.shuffle(pool)
    for a in pool:
        if len(chosen) >= target:
            break
        chosen.add(a)
    return sorted(chosen)


def estimate_eigenvalue_spam_robust(
    lam_true: float, n_shots: int, spam_atten: float, rng: np.random.Generator
) -> float:
    """SPAM-robust eigenvalue estimate by a two-depth (cycle-benchmarking) ratio.

    A constant SPAM factor ``A`` attenuates every measured eigenvalue:
    ``meas(depth L) = A * lambda^L``.  The ratio ``meas(2)/meas(1) = lambda``
    cancels ``A``, removing state-preparation and measurement error.
    """
    m1 = spam_atten * lam_true + rng.normal(
        scale=np.sqrt(max(1 - (spam_atten * lam_true) ** 2, 0.0) / max(n_shots, 1)))
    lam2 = lam_true ** 2
    m2 = spam_atten * lam2 + rng.normal(
        scale=np.sqrt(max(1 - (spam_atten * lam2) ** 2, 0.0) / max(n_shots, 1)))
    if abs(m1) < 1e-6:
        return float(m1 / spam_atten) if spam_atten > 0 else float(m1)
    return float(m2 / m1)


def learn_pauli_channel(
    channel: PauliChannel,
    n: int,
    n_bell: int = 4000,
    n_shots: int = 4000,
    support_threshold: int = 1,
    oversample: int = 4,
    spam_atten: float = 1.0,
    spam_robust: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Learn a sparse Pauli channel: Bell-sample support, WHT-invert coefficients.

    ``spam_atten`` (<=1) models a constant SPAM attenuation of measured Pauli
    eigenvalues (state-prep + readout error).  ``spam_robust=True`` cancels it
    via a two-depth ratio (cycle-benchmarking style).

    Returns dict with ``rates`` (estimated {a: c_a} on the learned support),
    ``support`` (learned), ``cond`` (condition number of the over-determined
    sign matrix --- ~1), and ``true_support``.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # 1a. Bell sampling for support (drop identity; threshold the rest).
    counts = bell_sample_support(channel, n_bell, rng)
    support = sorted(a for a, c in counts.items() if a != 0 and c >= support_threshold)
    if not support:
        return {"rates": {}, "support": [], "cond": 1.0,
                "true_support": sorted(a for a in channel if a != 0)}

    # 1b. Estimate Pauli eigenvalues at an over-determined probe set B, then
    #     least-squares invert.  Unknowns: the identity weight c_0 and the
    #     support rates, so cols = [0] + support; lambda_b = sum_a c_a sign(a,b).
    B = _probe_set(support, n, oversample, rng)
    cols = [0] + support
    M = sign_matrix(B, cols, n)                            # random +-1 -> cond ~ 1
    if spam_robust:
        lam_hat = np.array([
            estimate_eigenvalue_spam_robust(
                pauli_eigenvalue(channel, b, n), n_shots, spam_atten, rng)
            for b in B])
    else:
        lam_hat = np.array([
            estimate_eigenvalue_noisy(
                spam_atten * pauli_eigenvalue(channel, b, n), n_shots, rng)
            for b in B])
    c_hat, *_ = np.linalg.lstsq(M, lam_hat, rcond=None)
    rates = {a: max(float(c), 0.0) for a, c in zip(support, c_hat[1:])}

    return {
        "rates": rates,
        "support": support,
        "cond": float(np.linalg.cond(M)),
        "true_support": sorted(a for a in channel if a != 0),
    }


# --------------------------------------------------------------------------
# Stage 2: invert the (learned) Pauli channel and recover a state. (small n)
# --------------------------------------------------------------------------
def apply_pauli_channel(rho: Array, channel: PauliChannel, n: int) -> Array:
    out = np.zeros_like(rho, dtype=complex)
    # errstate guards against spurious FP-exception warnings from some BLAS backends
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for a, c in channel.items():
            Pa = pauli_matrix(a, n)
            out += c * (Pa @ rho @ Pa)
    return out


def invert_pauli_channel(rho: Array, channel: PauliChannel, n: int,
                         lam_floor: float = 1e-3) -> Array:
    """Apply E^{-1}: in the Pauli basis multiply each component r_b by 1/lambda_b.

    rho = (1/2^n) sum_b r_b P_b,  r_b = Tr(P_b rho);  E^{-1}(rho) scales r_b/lam_b.
    """
    dim = rho.shape[0]
    out = np.zeros_like(rho, dtype=complex)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for b in range(4 ** n):
            Pb = pauli_matrix(b, n)
            r_b = np.trace(Pb @ rho)
            lam = pauli_eigenvalue(channel, b, n)
            if abs(lam) < lam_floor:
                lam = lam_floor if lam >= 0 else -lam_floor
            out += (r_b / lam) * Pb
    return out / dim


def psd_normalize(rho: Array) -> Array:
    """Hilbert-Schmidt (Frobenius) metric projection onto the state set.

    Projects the eigenvalues onto the probability simplex ((w - mu)_+ with
    unit sum) rather than clip-and-renormalise. The metric projection onto a
    convex set is non-expansive in the Frobenius norm, which the recovery
    bound's proof requires; clip+renormalise is a different map that can
    expand distances (observed up to 1.7x) and is NOT covered by the theorem.
    """
    rho = (rho + rho.conj().T) / 2
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w, v = np.linalg.eigh(rho)
        w = w.real
        # Euclidean projection of eigenvalues onto the simplex {p >= 0, sum p = 1}
        u = np.sort(w)[::-1]
        css = np.cumsum(u) - 1.0
        idx = np.arange(1, len(u) + 1)
        rho_idx = np.nonzero(u - css / idx > 0)[0]
        if len(rho_idx) == 0:
            return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
        mu = css[rho_idx[-1]] / (rho_idx[-1] + 1)
        w = np.maximum(w - mu, 0.0)
        out = (v * w) @ v.conj().T
    return out


# --------------------------------------------------------------------------
# Short-time / Lindbladian-rate learning (Chebyshev derivative, Sources 2 & 4).
#
# For a Pauli Lindbladian  L(rho) = sum_{a != 0} gamma_a (P_a rho P_a - rho),
# the semigroup e^{tL} is a Pauli channel with eigenvalues
#     lambda_b(t) = exp(t * Lambda_b),   Lambda_b = sum_a gamma_a (sign(a,b) - 1).
# Estimating Lambda_b = d/dt lambda_b(t)|_0 by Chebyshev interpolation in t
# (instead of the snapshot lambda_b) recovers the *generator* rates gamma_a --
# the genuine Lindbladian-learning primitive -- and lets us rebuild the channel
# at any time.  The same WHT inversion (condition number O(1)) maps Lambda_b to
# gamma_a.
# --------------------------------------------------------------------------
def pauli_generator_eigenvalue(rates: PauliChannel, b: int, n: int) -> float:
    """Lambda_b = sum_{a != 0} gamma_a (sign(a,b) - 1) = -2 sum_{a anti b} gamma_a."""
    return float(sum(g * (sign(a, b, n) - 1) for a, g in rates.items() if a != 0))


def learn_pauli_generator(
    rates: PauliChannel,
    n: int,
    n_bell: int = 8000,
    n_shots: int = 8000,
    t_max: float = 0.1,
    m_nodes: int = 6,
    poly_degree: int = 4,
    oversample: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Learn Lindbladian rates gamma_a from short-time probes (Chebyshev deriv).

    ``rates`` is the true dissipator ``{a != 0 : gamma_a}`` (the oracle); the
    learner sees only finite-shot estimates of ``lambda_b(t) = exp(t*Lambda_b)``.

    Returns dict with ``rates`` (learned gamma_a on the learned support),
    ``support``, ``cond``, ``true_support``.
    """
    # local import keeps the Chebyshev primitive shared with generator_learning
    from .generator_learning import chebyshev_time_nodes, _cheb_derivative_at_zero

    if rng is None:
        rng = np.random.default_rng(0)

    # 1a. Bell sampling at a small time t0: error P_a appears with prob ~ t0*gamma_a
    #     (the dissipator support), identity taking the rest.
    t0 = t_max
    snapshot = {a: t0 * g for a, g in rates.items() if a != 0}
    snapshot[0] = max(1.0 - sum(snapshot.values()), 0.0)
    counts = bell_sample_support(snapshot, n_bell, rng)
    support = sorted(a for a in counts if a != 0)
    if not support:
        return {"rates": {}, "support": [], "cond": 1.0,
                "true_support": sorted(rates)}

    # 1b. Estimate Lambda_b at an over-determined probe set via Chebyshev deriv.
    B = _probe_set(support, n, oversample, rng)
    t_nodes = chebyshev_time_nodes(t_max, m_nodes)
    Lambda_hat = []
    for b in B:
        Lam_b = pauli_generator_eigenvalue(rates, b, n)
        vals = []
        for t in t_nodes:
            lam_t = np.exp(t * Lam_b)
            vals.append(estimate_eigenvalue_noisy(lam_t, n_shots, rng))
        Lambda_hat.append(_cheb_derivative_at_zero(t_nodes, np.array(vals), poly_degree))
    Lambda_hat = np.array(Lambda_hat)

    # Lambda_b = sum_{a in support} gamma_a (sign(a,b) - 1)
    M = sign_matrix(B, support, n) - 1.0
    g_hat, *_ = np.linalg.lstsq(M, Lambda_hat, rcond=None)
    learned = {a: max(float(g), 0.0) for a, g in zip(support, g_hat)}

    return {
        "rates": learned,
        "support": support,
        "cond": float(np.linalg.cond(M)),
        "true_support": sorted(rates),
    }

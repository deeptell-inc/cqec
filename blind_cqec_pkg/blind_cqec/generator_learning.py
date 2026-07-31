r"""Generator learning for blind CQEC --- closing the "known-noise" gap.

The original :func:`blind_cqec.estimators.estimate_channel_inversion` requires
the noise *strengths* ``(gamma, p, gamma_ad)`` to be supplied by hand.  That is
an oracle assumption: in a genuinely blind setting the noise model is unknown.

This module removes that assumption by *learning* the generator of the noise
channel from black-box short-time probes, in the spirit of the recent
ansatz-free / near-optimal Lindbladian-learning literature:

    * Arad, Chen, Guo, Rebentrost, Yu, "Near-Optimal Learning of Local
      Lindbladians", arXiv:2606.20535 --- estimate the PTM generator
      ``L = d/dt e^{tL}|_{t=0}`` from logarithmically short-time data by
      *Chebyshev interpolation in time*, then recover coefficients by a stable
      (condition-number-controlled) inversion.
    * Mobus, Bergamaschi, Stilck Franca, Rouze, "Robust Structure Learning of
      k-local Lindbladians", arXiv:2606.23652 --- the same short-time
      derivative-by-robust-polynomial-interpolation primitive, with a
      well-conditioned local inversion (here: an over-determined least squares
      over the known mechanism generators).

Mapping to this package's ``d``-level, energy-eigenbasis noise model:

    The unknown channel is assumed to lie in the model class
    ``combined_noise`` = dephasing -> depolarizing -> amplitude damping, with
    unknown rates ``theta = (gamma_rate, lambda_depol, kappa_ad)``.  Its
    one-parameter semigroup ``e^{tL}`` is exposed as a black box
    ``channel_at_time(rho, t)`` for ``t in (0, 1]`` (``t = 1`` reproduces the
    actual noisy channel).  We:

      1. probe the black box on a small set of informative states at
         Chebyshev time-nodes in a short interval ``[0, t_max]`` (the noiseless
         density-matrix read-out is the same idealization used elsewhere in
         this package);
      2. estimate the generator action ``L[sigma] = d/dt e^{tL}[sigma]|_{t=0}``
         entrywise by fitting a Chebyshev series in ``t`` and differentiating
         at ``t = 0``;
      3. solve the (over-determined, well-conditioned) linear system
         ``L[sigma] = gamma_rate * L_deph[sigma]
                    + lambda_depol * L_depol[sigma]
                    + kappa_ad   * L_ad[sigma]``
         for ``theta`` by least squares;
      4. integrate the semigroup to ``t = 1`` to obtain the channel strengths
         and feed them to :func:`estimate_channel_inversion`.

Because generators add at first order in ``t`` (composition order only enters
at ``O(t^2)``), the design matrix in step 3 is built analytically from the
*known mechanism forms* with unit rates --- only the rates are learned.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from numpy.polynomial import chebyshev as _C

from .estimators import estimate_channel_inversion, estimate_coherence_max
from .recovery import psd_project

Array = np.ndarray
ChannelAtTime = Callable[[Array, float], Array]


# --------------------------------------------------------------------------
# Known mechanism generators (unit rate), acting on a density matrix sigma.
# These encode the model class; the *rates* multiplying them are learned.
# --------------------------------------------------------------------------
def _gen_dephasing(sigma: Array) -> Array:
    """d/dgamma [dephasing(sigma, gamma)] at gamma=0:  -|i-j| sigma_{ij}."""
    d = sigma.shape[0]
    idx = np.arange(d)
    gap = np.abs(idx[:, None] - idx[None, :])
    return -gap * sigma


def _gen_depolarizing(sigma: Array) -> Array:
    """d/dp [(1-p) sigma + p I/d] at p=0:  -sigma + I/d."""
    d = sigma.shape[0]
    return -sigma + np.eye(d, dtype=complex) / d


def _gen_amplitude_damping(sigma: Array) -> Array:
    """d/dgamma_ad [amplitude_damping(sigma, gamma_ad)] at gamma_ad=0.

    From the cascaded-decay channel:
        [0,0]  ->  sum_{k>=1} sigma_{kk}
        [i,i]  ->  -sigma_{ii}            (i >= 1)
        [i,j]  ->  -((i+j)/2) sigma_{ij}  (i != j)
    """
    d = sigma.shape[0]
    out = np.zeros_like(sigma, dtype=complex)
    out[0, 0] = sum(sigma[k, k] for k in range(1, d))
    for i in range(1, d):
        out[i, i] = -sigma[i, i]
    for i in range(d):
        for j in range(d):
            if i != j:
                out[i, j] = -((i + j) / 2.0) * sigma[i, j]
    return out


_MECHANISMS = (_gen_dephasing, _gen_depolarizing, _gen_amplitude_damping)


# --------------------------------------------------------------------------
# Informative probe states (real, so generator images are real).
# --------------------------------------------------------------------------
def _default_probes(d: int) -> list[Array]:
    """A small, diverse probe set: basis populations + coherent superpositions.

    * computational basis states  -> populations (isolate depolarizing / AD)
    * uniform coherent state       -> all gaps at once
    * nearest/!far pair coherences -> gap dependence (isolate dephasing slope)
    """
    probes: list[Array] = []
    for i in range(d):
        e = np.zeros(d, dtype=complex)
        e[i] = 1.0
        probes.append(np.outer(e, e.conj()))
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    probes.append(np.outer(psi, psi.conj()))
    for (i, j) in ((0, 1), (0, d - 1), (1, d - 1)):
        if i < d and j < d and i != j:
            v = np.zeros(d, dtype=complex)
            v[i] = v[j] = 1.0 / np.sqrt(2.0)
            probes.append(np.outer(v, v.conj()))
    return probes


# --------------------------------------------------------------------------
# Chebyshev derivative-at-zero (the short-time interpolation primitive).
# --------------------------------------------------------------------------
def _cheb_derivative_at_zero(
    t_nodes: Array, values: Array, degree: int
) -> float:
    """Fit a Chebyshev series values(t) and return its derivative at t=0.

    This is the stable short-time derivative estimator of arXiv:2606.20535 /
    arXiv:2606.23652: rather than a finite difference (which amplifies
    sampling error), interpolate over several short-time nodes and read off
    the t->0 slope.
    """
    deg = min(degree, len(t_nodes) - 1)
    series = _C.Chebyshev.fit(t_nodes, values, deg)
    return float(series.deriv(1)(0.0))


def chebyshev_time_nodes(t_max: float, m: int) -> Array:
    """``m`` Chebyshev-Lobatto nodes mapped onto ``[0, t_max]`` (includes 0).

    ``t = 0`` is the identity channel, an exact anchor for the interpolation.
    """
    k = np.arange(m)
    cheb = -np.cos(np.pi * k / (m - 1))  # in [-1, 1], endpoints exact
    nodes = (cheb + 1.0) * 0.5 * t_max     # in [0, t_max]
    nodes[0] = 0.0
    return nodes


# --------------------------------------------------------------------------
# The learner.
# --------------------------------------------------------------------------
def default_t_max(d: int) -> float:
    """Adaptive short-time window.

    The interpolation is only valid in the genuinely short-time regime.  The
    fastest signal here is dephasing of the largest mode gap ``d-1``, whose
    phase grows like ``gamma_rate * (d-1) * t``.  To keep that phase ``O(1)``
    (so every gap still carries derivative information) the window must shrink
    with dimension --- mirroring how the Lindbladian-learning protocols pick
    the probe time relative to the interaction strength/locality.
    """
    return float(min(0.25, 1.5 / max(d - 1, 1)))


def learn_channel_rates(
    channel_at_time: ChannelAtTime,
    d: int,
    t_max: Optional[float] = None,
    m_nodes: int = 6,
    poly_degree: int = 4,
    probes: Optional[Sequence[Array]] = None,
) -> dict:
    """Learn the noise-generator rates from black-box short-time probes.

    Parameters
    ----------
    channel_at_time : callable ``(rho, t) -> rho_t``
        Black-box one-parameter semigroup ``e^{tL}`` of the unknown noise.
        ``t = 1`` corresponds to the actual noisy channel.
    d : int
        Hilbert-space dimension.
    t_max, m_nodes, poly_degree :
        Short-time interpolation window, number of Chebyshev time-nodes, and
        Chebyshev fit degree.
    probes : optional sequence of (d, d) density matrices
        Informative probe states; defaults to :func:`_default_probes`.

    Returns
    -------
    dict with keys
        ``rates`` = (gamma_rate, lambda_depol, kappa_ad),
        ``strengths`` = (gamma, p, gamma_ad) evaluated at t = 1,
        ``residual`` = least-squares residual,
        ``cond`` = design-matrix condition number (inversion stability).
    """
    if probes is None:
        probes = _default_probes(d)
    if t_max is None:
        t_max = default_t_max(d)
    t_nodes = chebyshev_time_nodes(t_max, m_nodes)

    a_cols: list[Array] = []  # analytic generator images (design matrix)
    g_vec: list[Array] = []   # estimated generator images (observed)

    for sigma in probes:
        sigma = np.asarray(sigma, dtype=complex)

        # observed derivative L[sigma], entrywise, via Chebyshev interpolation
        probed = np.stack(
            [np.asarray(channel_at_time(sigma, float(t)), dtype=complex)
             for t in t_nodes],
            axis=0,
        )  # (m_nodes, d, d)
        Lsig = np.zeros((d, d), dtype=complex)
        for i in range(d):
            for j in range(d):
                Lsig[i, j] = _cheb_derivative_at_zero(
                    t_nodes, probed[:, i, j].real, poly_degree
                )
        g_vec.append(Lsig.real.reshape(-1))

        # analytic generator images for the three unit-rate mechanisms
        cols = [mech(sigma).real.reshape(-1) for mech in _MECHANISMS]
        a_cols.append(np.stack(cols, axis=1))  # (d*d, 3)

    A = np.concatenate(a_cols, axis=0)   # (n_obs, 3)
    g = np.concatenate(g_vec, axis=0)    # (n_obs,)

    theta, residual, _, _ = np.linalg.lstsq(A, g, rcond=None)
    gamma_rate, lambda_depol, kappa_ad = (max(float(x), 0.0) for x in theta)

    # integrate the semigroup to t = 1 -> channel strengths
    gamma = gamma_rate                       # dephasing: gamma(t) = gamma_rate * t
    p = 1.0 - np.exp(-lambda_depol)          # depolarizing: 1 - (1-p) ; (1-p)=e^{-lambda}
    gamma_ad = 1.0 - np.exp(-kappa_ad)       # amplitude damping likewise

    cond = float(np.linalg.cond(A))
    res = float(residual[0]) if residual.size else float(
        np.linalg.norm(A @ theta - g) ** 2
    )
    return {
        "rates": (gamma_rate, lambda_depol, kappa_ad),
        "strengths": (gamma, float(p), float(gamma_ad)),
        "residual": res,
        "cond": cond,
    }


def estimate_learned_inversion(
    noisy: Array,
    channel_at_time: ChannelAtTime,
    d: Optional[int] = None,
    t_max: Optional[float] = None,
    m_nodes: int = 6,
    poly_degree: int = 4,
    probes: Optional[Sequence[Array]] = None,
    return_info: bool = False,
):
    """Blind target-state estimate via *learned* channel inversion.

    Drop-in companion to :func:`estimate_channel_inversion`, but the noise
    strengths are learned from short-time probes (this module's docstring)
    instead of being supplied by hand.

    Parameters
    ----------
    noisy : (d, d) complex array
        The observed noisy state (= ``channel_at_time(target, 1)``).
    channel_at_time : callable ``(rho, t) -> rho_t``
        Black-box noise semigroup used only for learning.
    return_info : bool
        If True, return ``(estimate, info_dict)`` where ``info_dict`` is the
        output of :func:`learn_channel_rates`.
    """
    noisy = np.asarray(noisy, dtype=complex)
    if d is None:
        d = noisy.shape[0]
    info = learn_channel_rates(
        channel_at_time, d, t_max=t_max, m_nodes=m_nodes,
        poly_degree=poly_degree, probes=probes,
    )
    gamma, p, gamma_ad = info["strengths"]
    est = estimate_channel_inversion(noisy, gamma=gamma, p=p, gamma_ad=gamma_ad)
    est = psd_project(est)
    if return_info:
        return est, info
    return est


def estimate_learned_hybrid(
    noisy: Array,
    channel_at_time: ChannelAtTime,
    d: Optional[int] = None,
    amp_cap: float = 8.0,
    t_max: Optional[float] = None,
    m_nodes: int = 6,
    poly_degree: int = 4,
    probes: Optional[Sequence[Array]] = None,
    return_info: bool = False,
):
    r"""Gap-adaptive blend of learned channel inversion and coherence maximization.

    The dephasing inversion multiplies each coherence ``rho_{ij}`` by
    ``exp(gamma * |i-j|)``.  For a mode gap ``g = |i-j|`` this *amplification*
    ``A(g) = exp(gamma * g)`` also amplifies any estimation/readout error in
    that element --- the dominant instability at large dimension.  Coherence
    maximization, by contrast, is unconditionally stable (it caps each
    coherence at the physicality bound ``sqrt(p_i p_j)``) but ignores the
    learned noise model.

    We therefore blend *per matrix element*, trusting the inversion only where
    its amplification is mild:

        est_{ij} = w_{ij} * CI_{ij} + (1 - w_{ij}) * CM_{ij},
        w_{ij}   = min(1, amp_cap / A(|i-j|)),    w_{ii} = 1.

    Small gaps (A(g) <= amp_cap) are fully inverted; large gaps fall back to
    coherence maximization.  ``amp_cap`` is the largest error-amplification the
    inversion is allowed to incur --- set it near the inverse of the readout
    precision (shot-noise scale).

    Parameters
    ----------
    amp_cap : float
        Amplification ceiling for trusting the inversion (default 8.0).
        ``amp_cap -> infinity`` recovers pure learned inversion
        (:func:`estimate_learned_inversion`).
    return_info : bool
        If True, also return the :func:`learn_channel_rates` info dict.

    Findings (honest negative result)
    ---------------------------------
    Benchmarking (``examples/ex07_learned_hybrid.py``) shows this blend is
    *dominated* in every regime tested:

      * With accurate learning (noiseless / high-precision read-out) the
        inversion is already excellent at all gaps, so any finite ``amp_cap``
        merely discards good inverted information --- pure inversion
        (``amp_cap = inf``) wins.
      * Under finite-statistics read-out, high-dimensional blind recovery of
        Haar-random states collapses for *all* strategies, because the
        dephasing inversion ``exp(gamma*g)`` is intrinsically ill-conditioned
        (``exp(15) ~ 1e6`` at ``d = 16``) --- the standard-quantum-limit /
        error-amplification wall, not something a blend can repair.

    The premise that coherence maximization is a safe large-gap fallback is
    false here: it uses the *noisy* phases and damped populations, so it is
    worse than an accurately learned inversion.  The function is retained as a
    tunable (and as the documented negative result); for production use prefer
    :func:`estimate_learned_inversion`.
    """
    noisy = np.asarray(noisy, dtype=complex)
    if d is None:
        d = noisy.shape[0]
    info = learn_channel_rates(
        channel_at_time, d, t_max=t_max, m_nodes=m_nodes,
        poly_degree=poly_degree, probes=probes,
    )
    gamma, p, gamma_ad = info["strengths"]
    ci = estimate_channel_inversion(noisy, gamma=gamma, p=p, gamma_ad=gamma_ad)
    cm = estimate_coherence_max(noisy)

    idx = np.arange(d)
    gap = np.abs(idx[:, None] - idx[None, :])
    amp = np.exp(gamma * gap)                       # A(g) per element
    w = np.minimum(1.0, amp_cap / np.maximum(amp, 1e-15))
    np.fill_diagonal(w, 1.0)                          # populations: always invert

    est = psd_project(w * ci + (1.0 - w) * cm)
    if return_info:
        return est, info
    return est

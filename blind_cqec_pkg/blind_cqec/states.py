"""State generators used in benchmarks."""
from __future__ import annotations

from typing import Optional

import numpy as np

Array = np.ndarray


def haar_random_pure(d: int, rng: Optional[np.random.Generator] = None) -> Array:
    """Haar-random pure state as a density matrix (rank 1, trace 1).

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    rng : numpy Generator, optional
        Seedable RNG. If None, uses the global default.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def werner_state(
    d: int,
    purity: float,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Werner-like state: rho = v |psi><psi| + (1-v) I/d with Haar-random |psi>.

    Parameters
    ----------
    d : int
    purity : float
        Mixing parameter v in [0, 1]. v = 1 is pure, v = 0 is maximally mixed.
    """
    if not 0.0 <= purity <= 1.0:
        raise ValueError("purity must be in [0, 1]")
    pure = haar_random_pure(d, rng=rng)
    return purity * pure + (1.0 - purity) * np.eye(d, dtype=complex) / d

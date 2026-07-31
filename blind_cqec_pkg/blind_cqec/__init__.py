"""
Blind Catalytic Quantum Error Correction (blind-cqec).

A reference implementation of blind CQEC: a two-stage protocol that
(1) estimates an unknown target state from a noisy density matrix and
(2) applies the density-matrix-level recovery interface toward that
estimate, restricted to the coherence modes that survive in the noisy
state (the mode-inclusion constraint of the Shiraishi--Takagi theorem).

Semantics (v0.2.0):
    * ``icec_recover(noisy, estimate)`` zeroes estimated coherences whose
      modes decoherence has annihilated in ``noisy`` (threshold 1e-10),
      then projects onto the PSD cone. Recovery therefore genuinely
      depends on the noisy state, and F_rec <= F_est in general; the gap
      is the mode-survival penalty Delta_mode of the companion paper.
    * All estimators assume oracle access to the exact noisy density
      matrix. Acquiring such access operationally costs Theta(d^2/eps^2)
      measurement shots; no measurement model is simulated here.
    * The catalytic covariant transformation itself (the physical
      mechanism) is NOT simulated; this package implements its idealized
      density-matrix interface only.

Companion paper:
    Hikaru Wakaura and Taiki Tanimae,
    "Blind Catalytic Quantum Error Correction: Target-State Estimation
     and Fidelity Recovery Without A Priori Knowledge",
    2026.
    arXiv: https://arxiv.org/abs/2604.11857
    DOI:   https://doi.org/10.48550/arXiv.2604.11857

Repository:
    https://github.com/deeptell-inc/blind_cqec_pkg
"""

from .noise import dephasing, depolarizing, amplitude_damping, combined_noise
from .estimators import (
    estimate_naive,
    estimate_coherence_max,
    estimate_channel_inversion,
    estimate_iterative,
    estimate_multicopy_average,
    estimate_hybrid,
)
from .generator_learning import (
    learn_channel_rates,
    estimate_learned_inversion,
    estimate_learned_hybrid,
    chebyshev_time_nodes,
)
from .recovery import icec_recover, psd_project
from .metrics import fidelity, trace_distance, l1_coherence
from .states import haar_random_pure, werner_state

__version__ = "0.2.0"
__author__ = "Hikaru Wakaura and Taiki Tanimae"

__all__ = [
    # noise channels
    "dephasing",
    "depolarizing",
    "amplitude_damping",
    "combined_noise",
    # estimators
    "estimate_naive",
    "estimate_coherence_max",
    "estimate_channel_inversion",
    "estimate_iterative",
    "estimate_multicopy_average",
    "estimate_hybrid",
    # generator learning (blind channel inversion)
    "learn_channel_rates",
    "estimate_learned_inversion",
    "estimate_learned_hybrid",
    "chebyshev_time_nodes",
    # recovery
    "icec_recover",
    "psd_project",
    # metrics
    "fidelity",
    "trace_distance",
    "l1_coherence",
    # states
    "haar_random_pure",
    "werner_state",
]

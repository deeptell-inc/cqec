"""
CQEC — Catalytic Quantum Error Correction
==========================================

Companion code for:
    Wakaura, "Catalytic Quantum Error Correction: Theory, Efficient
    Catalyst Preparation, and Numerical Benchmarks"
    arXiv:2603.25774 — https://doi.org/10.48550/arXiv.2603.25774

Repository: https://github.com/deeptell-inc/cqec

A quantum state recovery protocol based on catalytic covariant
transformations (Shiraishi & Takagi, PRL 132, 180202, 2024).

Modules
-------
core
    Coherence resource theory: density matrices, noise channels,
    fidelity, coherence measures, mode analysis.
protocol
    CQEC recovery protocol: catalytic covariant recovery with
    energy-conserving gates.
catalyst
    Catalyst preparation strategies: variational, swap test,
    covariant swap test, DD+Twirl pipeline.
algorithms
    Target state generators for benchmark quantum algorithms.
"""

from cqec.core import (
    fidelity,
    purity,
    l1_coherence,
    coherence_modes,
    mode_inclusion,
    dephasing_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    random_density_matrix,
)

from cqec.protocol import (
    CQECRecovery,
    ec_gate,
    build_ec_circuit,
)

from cqec.catalyst import (
    swap_test,
    recursive_swap,
    dd_twirl_pipeline,
    twirl_analytical,
    variational_catalyst,
)

from cqec.joint import JointCQEC

__version__ = "0.2.0"
__author__ = "Hikaru Wakaura"
__email__ = "h.wakaura@qiri.co.jp"

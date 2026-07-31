# blind-cqec

Reference implementation of **Blind Catalytic Quantum Error Correction**:
a two-stage protocol that (i) estimates an unknown target state from a
noisy density matrix and (ii) applies the density-matrix-level recovery
interface toward that estimate, restricted to the coherence modes that
survive in the noisy state.

## Scope (read this first)

This package implements the **idealized density-matrix interface** of
catalytic recovery, not the physical protocol:

- `icec_recover(noisy, estimate)` zeroes estimated coherences whose modes
  decoherence has annihilated in the noisy state (mode-inclusion
  constraint of the Shiraishi–Takagi theorem; threshold `1e-10`), then
  projects onto the PSD cone. Consequently **`F_rec <= F_est`**, and the
  gap is the mode-survival penalty `Delta_mode` of the companion paper.
- The catalytic covariant transformation itself (catalyst construction,
  joint-system unitary) is **not simulated**; see the companion CQEC
  framework preprint for its construction.
- All estimators assume **oracle access to the exact noisy density
  matrix**. Acquiring such access operationally requires on the order of
  `d^2/eps^2` measurement shots; no measurement model is simulated.

Version 0.2.0 incorporates the corrections from an adversarial numerical
audit (see `CHANGELOG.md`): unified forward/inverse noise conventions,
the mode-inclusion restriction above, and dimension caps at `d <= 64`
(the exact dephasing inversion overflows double precision beyond that).

## Companion paper

**Title:** *Blind Catalytic Quantum Error Correction: Target-State Estimation and Fidelity Recovery Without A Priori Knowledge*
**Authors:** Hikaru Wakaura, Taiki Tanimae (QIRI Inc.), 2026
**Preprint:** https://arxiv.org/abs/2604.11857 (DOI: [10.48550/arXiv.2604.11857](https://doi.org/10.48550/arXiv.2604.11857))
**Companion paper (foundational CQEC framework):** Wakaura & Tanimae, *Catalytic Quantum Error Correction: Theory, Efficient Catalyst Preparation, and Numerical Benchmarks* — [arXiv:2603.25774](https://arxiv.org/abs/2603.25774v4)
**Repository:** https://github.com/deeptell-inc/blind_cqec_pkg

## Install

```bash
pip install blind-cqec
```

Development install from source:

```bash
git clone https://github.com/deeptell-inc/blind_cqec_pkg
cd blind-cqec
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import numpy as np
from blind_cqec import (
    haar_random_pure, combined_noise,
    estimate_coherence_max, estimate_channel_inversion,
    icec_recover, fidelity,
)

rng = np.random.default_rng(42)
target = haar_random_pure(d=8, rng=rng)

# Apply decoherence
noisy = combined_noise(target, gamma=1.0, p=0.15, gamma_ad=0.1)
print("Noisy fidelity:", fidelity(noisy, target))

# Blind CQEC with coherence maximization (no noise-model knowledge)
est_cm   = estimate_coherence_max(noisy)
rec_cm   = icec_recover(noisy, est_cm)
print("Coh. max  recovery:", fidelity(rec_cm, target))

# Blind CQEC with channel inversion (requires noise-model knowledge)
est_ci   = estimate_channel_inversion(noisy, gamma=1.0, p=0.15, gamma_ad=0.1)
rec_ci   = icec_recover(noisy, est_ci)
print("Ch. inv.  recovery:", fidelity(rec_ci, target))
```

Typical output (seed=42, d=8; at this dimension all modes survive, so
exact channel inversion reaches the ceiling):

```
Noisy fidelity:          0.2516
Coh. max  recovery:      0.9248
Ch. inv.  recovery:      1.0000
```

At higher dimension the mode-survival ceiling caps every strategy,
including exact inversion (`examples/ex02_dimension_sweep.py`,
20 Haar-random states per dimension, combined noise):

```
   d     NoCor     CohMax     ChInv
   2    0.7309    0.9894    1.0000
   4    0.4968    0.9762    1.0000
   8    0.2506    0.9630    1.0000
  16    0.1340    0.9586    1.0000
  32    0.0599    0.5353    0.6176
  64    0.0309    0.1862    0.3028
```

## Package layout

```
blind_cqec/
├── noise.py        # dephasing / depolarizing / amplitude damping / combined
│                   #   (gap-dependent conventions, forward == inverse)
├── estimators.py   # naive, coherence-max, channel-inversion, iterative,
│                   #   multi-copy averaging, hybrid, learned variants
├── recovery.py     # mode-inclusion-restricted recovery + PSD projection
├── metrics.py      # Uhlmann fidelity, trace distance, l1 coherence
└── states.py       # Haar-random pure states, Werner states
```

## Reproducing the paper figures

Example scripts under `examples/` reproduce the headline numbers:

| Script                           | Paper section                          |
|----------------------------------|----------------------------------------|
| `examples/ex01_quickstart.py`    | sanity check (prints the numbers)      |
| `examples/ex02_dimension_sweep.py` | Haar-random dimension sweep          |
| `examples/ex03_qem_compare.py`   | QEM comparison (idealized)             |
| `examples/ex04_sensitivity.py`   | noise-parameter sensitivity            |
| `examples/ex05_hybrid.py`        | hybrid strategy (oracle analysis)      |

Run with:
```bash
python examples/ex01_quickstart.py
```

## Testing & reproducibility

All randomness routes through `numpy.random.Generator` and is seeded inside
tests/examples. The test suite (`pytest`, 91 tests) validates:

- Channel properties (trace-preserving, PSD-preserving)
- Estimator correctness on pure-state targets
- Channel inversion is the exact inverse under each noise component
  (round-trip `F > 0.999` at all `d <= 64`)
- Mode-restriction invariants: `F_rec <= F_est` always; annihilated modes
  stay dead even under an oracle estimate; larger `mode_tol` never
  improves fidelity
- Recovery-fidelity thresholds pinned to the paper's reported values
  (tuned to the default `mode_tol = 1e-10`)

## Citation

If you use this software, please cite both the package and the paper:

```bibtex
@article{Wakaura2026blindCQEC,
  author  = {Wakaura, Hikaru and Tanimae, Taiki},
  title   = {Blind Catalytic Quantum Error Correction:
             Target-State Estimation and Fidelity Recovery Without
             {A Priori} Knowledge},
  year    = {2026},
  eprint        = {2604.11857},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2604.11857},
  url           = {https://arxiv.org/abs/2604.11857},
}

@software{blind_cqec_software,
  author  = {Wakaura, Hikaru and Tanimae, Taiki},
  title   = {{blind-cqec}: Reference implementation of Blind
             Catalytic Quantum Error Correction},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/deeptell-inc/blind_cqec_pkg},
}
```

## License

MIT — see `LICENSE`.

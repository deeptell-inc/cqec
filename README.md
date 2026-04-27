# CQEC — Catalytic Quantum Error Correction

[![arXiv](https://img.shields.io/badge/arXiv-2603.25774-b31b1b.svg)](https://doi.org/10.48550/arXiv.2603.25774)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code for **Wakaura, "Catalytic Quantum Error Correction: Theory, Efficient Catalyst Preparation, and Numerical Benchmarks"** ([arXiv:2603.25774](https://doi.org/10.48550/arXiv.2603.25774)).

A Python package for quantum state recovery via catalytic covariant transformations, based on the theoretical framework of [Shiraishi & Takagi (PRL 132, 180202, 2024)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.180202).

## Overview

CQEC recovers a known target quantum state from noisy copies without an error magnitude threshold. Recovery succeeds whenever the coherent modes of the target are preserved in the noisy state, regardless of noise strength. A reusable catalyst mediates the transformation.

Key features:
- **Threshold-free recovery**: No minimum fidelity required — any nonzero coherence enables recovery
- **DD+Twirl+Swap Test pipeline**: 10^9-fold reduction in catalyst preparation cost over distillation
- **Four preparation strategies**: Variational (0 copies), standard/covariant swap test, DD+Twirl
- **Benchmark suite**: qDRIFT, QKAN, CF-QPE, Regev across dephasing, depolarizing, and combined noise

## Installation

```bash
pip install cqec
```

For development (with tests and plotting):

```bash
pip install cqec[dev]
```

## Quick Start

```python
import numpy as np
from cqec import fidelity, dephasing_channel, mode_inclusion
from cqec.algorithms import make_qdrift
from cqec.catalyst import dd_twirl_pipeline
from cqec.protocol import CQECRecovery

# 1. Create target state (qDRIFT, d=8)
rho_target, d = make_qdrift()

# 2. Apply noise
rho_noisy = dephasing_channel(rho_target, gamma=2.0)
print(f"F_noisy = {fidelity(rho_target, rho_noisy):.4f}")
print(f"Modes preserved: {mode_inclusion(rho_target, rho_noisy)}")

# 3. Prepare catalyst via DD+Twirl pipeline
psi_cat = np.ones(d, dtype=complex) / np.sqrt(d)
rho_cat_ideal = np.outer(psi_cat, psi_cat.conj())
result = dd_twirl_pipeline(rho_cat_ideal, d, gamma=2.0, n_copies=8, n_dd=8)
print(f"F_cat = {result['fidelity']:.4f}")

# 4. CQEC recovery
recovery = CQECRecovery(d, n_gates=5)
rec = recovery.recover(rho_target, rho_noisy, result['rho_cat'])
print(f"F_rec = {rec['fidelity']:.4f}")
```

## Package Structure

```
cqec/
  __init__.py          # Public API
  core.py              # Fidelity, coherence measures, noise channels
  protocol.py          # CQEC recovery with EC gates
  catalyst.py          # Catalyst preparation (variational, swap test, DD+Twirl)
  algorithms.py        # Benchmark algorithm states
tests/
  test_core.py         # Core module tests (16 tests)
  test_catalyst.py     # Catalyst preparation tests (4 tests)
scripts/
  benchmark_prxq.py    # Paper benchmark (200 configurations)
  benchmark_nature_supplement.py  # Gate noise, depth scan, entanglement
  plot_prxq_figures.py # Figure generation
  plot_dd_scaling.py   # DD+Twirl scaling figures
paper/
  paper_unified.tex    # Manuscript (revtex4-2, 19 pages)
  paper_unified.bib    # References (40+ entries)
```

## Reproducing Paper Results

### Quick reproduction (~35s)

The fastest way to verify all key paper claims:

```bash
python scripts/reproduce_paper.py
```

This runs in ~35 seconds and reproduces:
- Sharp threshold (Sec VI.B): mode_inclusion ε > 0 vs ε = 0
- Recursive swap test under depolarizing (Table VI): F_cat at n = 2-64
- DD+Twirl pipeline (Table VII): F_cat > 0.96 for all d ∈ {4, 8, 16, 64}
- Finite-n recovery (Table X): monotone F_rec(n) for QKAN, qDRIFT
- Bell state concurrence recovery (Table XIII): 5.0× improvement

### Full reproduction (~10 min)

```bash
# Asymptotic recovery + sharp threshold + 200-config noise sweep
python scripts/benchmark_prxq.py

# DD+Twirl pipeline benchmarks
python scripts/dd_purification.py

# Finite-n actual recovery, gate noise, gate depth, entanglement
python scripts/benchmark_nature_supplement.py

# Generate all figures (16 figures + TikZ circuit diagram)
python scripts/plot_prxq_figures.py
python scripts/plot_finite_copy.py
python scripts/plot_dd_scaling.py
```

### Verification via tests

```bash
python -m pytest tests/ -v   # 44 tests, all passing
```

## Method Summary

| Method | Copies | F_cat (gamma=2) | Best for |
|--------|--------|-----------------|----------|
| Variational | 0 | 1.000 (d<=4) | NISQ, no copies available |
| Standard swap test | 8-64 | 0.93 (depol) | Depolarizing noise |
| DD+Twirl+Swap Test | 8 | 0.96 (all d) | Dephasing noise (recommended) |
| Distillation | ~10^10 | 0.99 | Theoretical baseline |

## Citation

Paper: <https://doi.org/10.48550/arXiv.2603.25774>

```bibtex
@article{Wakaura2026cqec,
  author  = {Wakaura, Hikaru},
  title   = {Catalytic Quantum Error Correction: Theory, Efficient Catalyst
             Preparation, and Numerical Benchmarks},
  journal = {arXiv preprint arXiv:2603.25774},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.25774},
  url     = {https://doi.org/10.48550/arXiv.2603.25774},
}
```

## License

MIT License. See [LICENSE](LICENSE).

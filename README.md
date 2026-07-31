# CQEC — Catalytic Quantum Error Correction

[![arXiv](https://img.shields.io/badge/arXiv-2603.25774-b31b1b.svg)](https://doi.org/10.48550/arXiv.2603.25774)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-deeptell--inc%2Fcqec-blue?logo=github)](https://github.com/deeptell-inc/cqec)

Companion code for **Wakaura, "Catalytic Quantum Error Correction: Theory, Efficient Catalyst Preparation, and Numerical Benchmarks"** ([arXiv:2603.25774](https://doi.org/10.48550/arXiv.2603.25774)).

- **Repository:** <https://github.com/deeptell-inc/cqec>
- **Install:** `pip install cqec` (or `pip install git+https://github.com/deeptell-inc/cqec.git`)

A Python package for quantum state recovery via catalytic covariant transformations, based on the theoretical framework of [Shiraishi & Takagi (PRL 132, 180202, 2024)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.180202).

## Overview

CQEC recovers a known target quantum state from noisy copies without an error magnitude threshold. Recovery succeeds whenever the coherent modes of the target are preserved in the noisy state, regardless of noise strength. A reusable catalyst mediates the transformation.

Key features:
- **Threshold-free recovery** (mode-inclusion condition): success depends on the *support* of coherence, not its magnitude
- **CPTP joint-channel implementation** (`cqec.joint.JointCQEC`, v0.2.0): a genuinely linear, completely positive, exactly covariant recovery channel on the joint system–catalyst–ancilla register, with the catalyst obtained by partial trace (never copied). Verified properties: `[U, H] = 0` to machine precision, linearity residual < 1e-12, Choi matrix PSD
- **Effective surrogate model** (`cqec.protocol.CQECRecovery`): fast target-parameterized interpolation model used for large-d sweeps — explicitly *not* a CPTP channel; see the paper's methods caveat
- **DD+Twirl+Swap Test pipeline**: 10^4–10^9-fold copy-count reduction at matched catalyst fidelity (dimension-dependent)
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

# 4. Recovery, two ways:
# (a) effective surrogate model (fast, NOT a CPTP channel — see paper caveat)
recovery = CQECRecovery(d, n_gates=5)
rec = recovery.recover(rho_target, rho_noisy, result['rho_cat'])
print(f"F_rec (effective model) = {rec['fidelity']:.4f}")

# (b) genuine CPTP joint-space covariant channel (v0.2.0, d <= 8 practical)
from cqec.joint import JointCQEC
ch = JointCQEC(d)
ch.optimize(rho_target, rho_noisy, result['rho_cat'])   # offline, target-aware
rho_out, rho_cat_out = ch.apply(rho_noisy, result['rho_cat'])  # fixed linear map
print(f"F_rec (CPTP channel) = {fidelity(rho_target, rho_out):.4f}")
print(f"F_cat marginal (partial trace) = {fidelity(result['rho_cat'], rho_cat_out):.4f}")
print(f"covariance ||[U,H]|| = {ch.covariance_residual():.1e}")   # exactly 0
```

## Package Structure

```
cqec/
  __init__.py          # Public API (v0.2.0)
  core.py              # Fidelity, coherence measures, noise channels
  protocol.py          # Effective recovery model (surrogate; not CPTP)
  joint.py             # CPTP joint-space covariant channel (v0.2.0)
  catalyst.py          # Catalyst preparation (variational, swap test, DD+Twirl)
  algorithms.py        # Benchmark algorithm states
tests/
  test_core.py         # Core module tests (16 tests)
  test_catalyst.py     # Catalyst preparation tests (4 tests)
  test_joint.py        # CPTP channel properties: linearity, CP, TP,
                       #   exact covariance, catalyst-by-partial-trace (8 tests)
  test_reproduce_paper.py  # Paper-claim regression tests (24 tests)
scripts/
  reproduce_paper.py   # One-command reproduction of key paper claims
  benchmark_joint.py   # CPTP joint-channel validation + target-swap control
  benchmark_prxq.py    # Effective-model benchmark (200 configurations)
  benchmark_nature_supplement.py  # Gate noise, depth scan, entanglement
  plot_prxq_figures.py # Figure generation
  plot_dd_scaling.py   # DD+Twirl scaling figures
paper/
  paper_quantum.tex    # Manuscript (quantumarticle, 21 pages)
  paper_quantum.bib    # References
```

## Reproducing Paper Results

### Quick reproduction (~1 min)

The fastest way to verify all key paper claims:

```bash
python scripts/reproduce_paper.py
```

This reproduces:
- Mode-inclusion threshold scan: ε > 0 vs ε = 0
- Recursive swap test under depolarizing: F_cat at n = 2-64
- DD+Twirl pipeline: F_cat > 0.96 for all d ∈ {4, 8, 16, 64}
- Finite-n recovery: monotone F_rec(n) for QKAN, qDRIFT
- Bell state concurrence recovery (5.0×, effective model)
- CPTP joint-channel validation (d = 4): F 0.54 → 0.72, exact covariance,
  machine-precision linearity, Choi positivity

### CPTP joint-channel benchmarks (~10 min)

```bash
python scripts/benchmark_joint.py
```

Runs the genuine joint-space channel for d = 4 and d = 8 (dephasing +
depolarizing, ideal and DD+Twirl catalysts) and the target-swap control
experiment. Honest headline: under dephasing at d = 4 the channel truly
recovers fidelity (0.54 → 0.77 with the DD+Twirl catalyst); under
depolarizing and at d = 8 the shallow 3-layer ansatz does not yet improve
fidelity, and the catalyst marginal degrades (0.48–0.84) — the current
circuit transfers rather than catalytically borrows coherence.

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
python -m pytest tests/ -v   # 52 tests, all passing
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

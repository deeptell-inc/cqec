# Quantum journal submission package

This directory contains the manuscript and supporting files for submission to
[**Quantum**](https://quantum-journal.org/).

## Files

| File | Purpose |
|---|---|
| `paper_blind_cqec_quantum.tex` | Main manuscript (`quantumarticle` class, single-column, A4) |
| `paper_blind_cqec_quantum.bib` | Bibliography (`quantum.bst` style) |
| `paper_blind_cqec_quantum.pdf` | Compiled PDF (27 pages) |
| `cover_letter_quantum.tex` / `.pdf` | Cover letter for the editors |
| `fig_*.pdf` | All figures referenced by the manuscript (17 files) |

## Revision status (2026-07-31)

The manuscript incorporates the corrections from an **adversarial numerical
audit** (two independent reviewer models, cross-checked by reproduction).
Key changes relative to the pre-audit draft, all recorded in the manuscript
itself (Computational Details section):

- Recovery is stated explicitly as the mode-inclusion-restricted interface
  (`F_rec <= F_est`, mode threshold 1e-10); the catalytic transformation
  itself is not simulated.
- The closed-form crossover equation was retracted (no positive solution);
  the crossover between estimators is now empirical and calibration-dependent.
- All noise channels unified into a single forward/inverse-consistent module;
  the dimension-sweep, sensitivity, per-channel, hybrid, and main-table
  numbers were regenerated (channel inversion now coincides with the oracle
  under exact calibration; the Regev ceiling 0.738 is the mode-survival limit).
- Dimensions d >= 128 withdrawn (double-precision overflow of the exact
  dephasing inversion).
- Negative results reported: idealized PEC reaches F = 1.000 at every
  dimension; LiH VQE null result; coherence maximization harmful for target
  purity v <~ 0.5; Regev unrecoverable under dephasing-only noise.
- "Single-copy" resource claims replaced by oracle density-matrix access
  accounting (Theta(d^2/eps^2) shots).

The companion package `blind_cqec` is released as **v0.2.0** with matching
semantics (see `blind_cqec_pkg/CHANGELOG.md`); the test suite (91 tests)
pins the manuscript's headline numbers.

## Compile from source

```bash
export PATH="/usr/local/texlive/2025/bin/universal-darwin:$PATH"
pdflatex paper_blind_cqec_quantum.tex
bibtex   paper_blind_cqec_quantum
pdflatex paper_blind_cqec_quantum.tex
pdflatex paper_blind_cqec_quantum.tex
```

Result: 27-page PDF, zero errors / zero overfull boxes / zero undefined
references.

## Submission via Quantum's Scholastica portal

1. **Manuscript file**: upload `paper_blind_cqec_quantum.pdf`.
2. **LaTeX source archive**: upload a `.zip` containing
   `paper_blind_cqec_quantum.tex`, `paper_blind_cqec_quantum.bib`, and all
   `fig_*.pdf` files referenced in the manuscript.
3. **Cover letter**: upload `cover_letter_quantum.pdf`.
4. **Suggested referees**: see cover letter.
5. **Code/Data availability statement**: source code at
   <https://github.com/deeptell-inc/blind_cqec_pkg>; the `blind_cqec` Python
   package (v0.2.0) is published on PyPI under the MIT license.

## Note on the older RevTeX draft

`paper_blind_cqec.tex` (PRX-style, RevTeX 4-2) predates the numerical audit
and is **superseded**: it still contains the retracted crossover equation and
the pre-audit numbers. Do not submit or circulate it; the Quantum manuscript
is the canonical version. If a RevTeX version is ever needed, regenerate it
from `paper_blind_cqec_quantum.tex`.

## Reproducing the results

The headline numbers can be reproduced from the open-source repository:

```bash
git clone https://github.com/deeptell-inc/blind_cqec_pkg.git
cd blind_cqec_pkg
pip install -e ".[test]"

# Reproduce the main numbers (under 30 s on a laptop):
python examples/ex01_quickstart.py
python examples/ex02_dimension_sweep.py
python examples/ex03_qem_compare.py
python examples/ex04_sensitivity.py
python examples/ex05_hybrid.py

# Run the full test suite (91 tests):
pytest
```

All randomness is seeded (`numpy.random.default_rng` / recorded seeds) and
the test suite pins the headline fidelities, including the mode-restriction
invariants (`F_rec <= F_est`; annihilated modes stay dead; larger `mode_tol`
never improves fidelity).

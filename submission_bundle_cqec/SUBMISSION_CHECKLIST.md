# Quantum submission bundle — CQEC (v0.2.0 revision)

Manuscript: **"Catalytic Quantum Error Correction: Theory, Efficient Catalyst
Preparation, and Numerical Benchmarks"**
Authors: Hikaru Wakaura, Taiki Tanimae (QIRI Inc., Tokyo)

## Bundle contents

| File | Purpose |
|---|---|
| `manuscript_paper_quantum.pdf` | Compiled manuscript (quantumarticle, 21 pages, A4 twocolumn) |
| `cover_letter.pdf` | Cover letter with suggested referees (incl. contact emails) |
| `arxiv_v2_source.tar.gz` | Self-contained arXiv source (tex + bbl + bib + 16 figures) |
| `arxiv_source/` | Unpacked copy of the same source |

Verified before packaging:
- Compiles standalone with `pdflatex` twice (no bibtex needed — `.bbl` included): 21 pages, zero errors, zero overfull boxes, zero undefined references.
- Abstract length: 1,765 characters plain text (`abstract_plain.txt`; < 1,920 limit).
- Package claims match `cqec` v0.2.0 (52 tests passing) at
  https://github.com/deeptell-inc/cqec

## Submission steps

### Step 1 — Update the arXiv record (arXiv:2603.25774 → v2)

1. Log in to arXiv → "Replace" on abs/2603.25774.
2. Upload `arxiv_v2_source.tar.gz`.
3. Update the abstract field with the manuscript's current abstract
   (copy from `paper_quantum.tex`, strip LaTeX cite commands).
4. Comments field suggestion:
   `21 pages, 17 figures (16 files + 1 TikZ). v2: adversarial-review revision; adds CPTP
   joint-channel validation (Sec. VI), corrected finite-copy bounds,
   and explicit effective-model scope. Code: v0.2.0 at
   https://github.com/deeptell-inc/cqec`
5. Note the announcement date; Quantum requires the paper to be on arXiv.

### Step 2 — Submit to Quantum (Scholastica)

1. https://quantum-journal.org/ → Submit → Scholastica portal.
2. Enter the arXiv ID (2603.25774) and the v2 version number.
3. Upload `cover_letter.pdf` in the cover-letter field.
4. Suggested referees (from the cover letter):
   - Naoto Shiraishi (U. Tokyo) — shiraishi@phys.c.u-tokyo.ac.jp
   - Ryuji Takagi (NTU) — ryuji.takagi@ntu.edu.sg
   - Tim Byrnes (NYU Shanghai) — tim.byrnes@nyu.edu
   - Iman Marvian (Duke) — iman.marvian@duke.edu
   - Andrew M. Childs (UMD) — amchilds@umd.edu
   - Aram W. Harrow (MIT) — aram@mit.edu
5. Declare the concurrent related submission ("Blind Catalytic Quantum
   Error Correction") as stated in the cover letter.

### Step 3 — After submission

- `git tag v0.2.0 && git push --tags` on the code repository so the
  reviewed code state is pinned.
- Optional: `twine upload dist/cqec-0.2.0*` to publish v0.2.0 on PyPI
  (requires PyPI token; run locally).

## Claim-integrity notes for referee response

The manuscript's headline claims were revised after an internal
adversarial review; the current claims are scoped as follows and each is
backed by the repository:

1. Effective-model sweep F > 0.99 (200 points) — `scripts/benchmark_prxq.py`
   (legacy effective map; scope stated in Secs. III.D, VI.A).
2. CPTP joint-channel validation, d = 4 dephasing: 0.54 → 0.77 —
   `scripts/benchmark_joint.py`, `results_joint.json`; properties locked by
   `tests/test_joint.py`.
3. Copy-count reduction 10^4–10^9 at matched F_cat ≥ 0.99 endpoint
   (n = 32), postselection P ≈ 0.7 included — Sec. IV.E.
4. Known open gaps stated in the paper: depolarizing / d = 8 recovery,
   catalyst-marginal degradation (0.48–0.84), assumed DD suppression law.

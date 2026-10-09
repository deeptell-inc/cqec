# Physical Review A submission bundle — CQEC

Manuscript: **"Catalytic quantum error correction: efficient catalyst
preparation and a completely positive covariant recovery channel"**
Authors: Hikaru Wakaura, Taiki Tanimae (QIRI Inc., Tokyo)
Target: PRA Regular Article, Quantum information section.

## Restructure relative to the arXiv version

- revtex4-2 (`pra` option), 21 pages twocolumn.
- Main text leads with the two primary contributions: the DD+Twirl+Swap-Test
  catalyst pipeline and the CPTP joint-channel validation.
- All effective-model asymptotic benchmarks (200-point sweep, sharp
  threshold, QEC comparison, TTN, durability) are demoted to
  Appendix A, explicitly labeled as surrogate-model results.
- Submission-hygiene trace check passed:
  `grep -inE "corrected|earlier (revision|draft)|previously reported|referee|phase-1"`
  returns only the technical term "Corrected state" (QEC output column of
  the paradigm-comparison table).

## Bundle contents

| File | Purpose |
|---|---|
| `manuscript_paper_pra.pdf` | Compiled manuscript (21 pp, zero errors / overfull / undefined refs) |
| `cover_letter.pdf` | PRA cover letter with 5 suggested referees (emails included) |
| `source/` | Self-contained source: `paper_pra.tex` + `.bbl` + `.bib` + 16 figures. Verified: compiles standalone with pdflatex twice (no bibtex needed) |

## Submission steps

1. arXiv: replace abs/2603.25774 with this version (v3) OR submit PRA from
   the existing v2 and upload this source — PRA accepts direct source upload
   at https://authors.aps.org/Submissions/
2. Journal: Physical Review A → Regular Article → Quantum information.
3. Upload `source/` contents (tex + bbl + bib + figures) or a tarball of it.
4. Paste cover letter text; enter suggested referees from `cover_letter.pdf`.
5. Data availability: point to https://github.com/deeptell-inc/cqec (v0.2.0,
   52 tests, one-command reproduction).

## Claim-evidence map (for referee response)

1. Pipeline 10^4–10^9-fold reduction at matched F_cat ≥ 0.99 (n = 32),
   P_succ ≈ 0.7 — Sec. IV.E; `scripts/reproduce_paper.py`.
2. CPTP joint channel: covariance exactly 0, linearity < 4e-16, Choi ≥ -6e-16,
   0.54 → 0.77 at d = 4 dephasing, target-swap control 0.403 vs 0.717 —
   Sec. VI; `results_joint.json`; `tests/test_joint.py`.
3. Open gaps stated in-paper: no improvement under depolarizing / d = 8;
   catalyst marginal 0.48–0.84 (consumption); DD law is an assumption.
4. Effective-model results: Appendix A only, never cited as channel-level
   evidence.

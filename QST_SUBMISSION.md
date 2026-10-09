# Quantum Science and Technology (IOP) submission package

Canonical manuscript for submission after the Quantum decision (2026-09-14).

| File | Purpose |
|---|---|
| `paper_blind_cqec_qst.tex` / `.bib` | Manuscript, `article` class + numeric `natbib` (IOP accepts standard classes; `iopart.cls` not required) |
| `paper_blind_cqec_qst.pdf` | Compiled PDF |
| `cover_letter_qst.tex` / `.pdf` | Cover letter (paste into the IOP form or upload) |
| `fig_*.pdf` | 17 figures referenced by the manuscript |
| `blind_cqec_pkg/` | v0.2.0, 91 tests, `results/*.json`, `REPRODUCE.md` |

Compile:
```bash
export PATH="/usr/local/texlive/2025/bin/universal-darwin:$PATH"
pdflatex paper_blind_cqec_qst.tex && bibtex paper_blind_cqec_qst && pdflatex paper_blind_cqec_qst.tex && pdflatex paper_blind_cqec_qst.tex
```

IOP form fields: article type **Paper**; keywords are in the manuscript after the abstract; the
"Data availability statement" section satisfies IOP's mandatory DAS; ORCID 0009-0001-2341-7710.

Superseded files (do not submit): `paper_blind_cqec.tex` (pre-audit RevTeX draft),
`paper_blind_cqec_quantum.*`, `cover_letter_quantum.*`, `response_to_reviewers_quantum.*`
(kept for the record; the response letter documents the review history).

Before/alongside submission: post `paper_blind_cqec_qst.tex` as arXiv v2 of 2604.11857 — v1 still
carries the retracted crossover and correlation claims.

# Citation audit — paper_blind_cqec_qst.tex (2026-09-14)

Method: all 47 sentences carrying a `\cite` were extracted; each cited primary source was fetched
(arXiv/publisher PDF via hyperresearch) and read; locations are as printed in the source. Books not
online (Nielsen–Chuang) and paywalled papers (Shor 1995 full text, Davies 1988 full text) are labeled
"Not Found in Uploaded Sources" — no location was guessed.

Labels: SS = Strongly Supported · PS = Partially Supported · CT = Contradicted · NF = Not Found in Uploaded Sources
Totals: SS 22 · PS 19 · CT 2 · NF 4

| ID | Claim (gist) | Cite(s) | Label | Source passage / location | Flags |
|---|---|---|---|---|---|
| C01 | Coherence must be protected (intro opener) | NielsenChuang2010 | NF | Book offline | Generic; low risk |
| C02 | QEC = encoding overhead + noise threshold | Shor1995, Gottesman1997, Knill2005 | SS (jointly) | Gottesman Ch.1 p.12 "arbitrarily long calculations as long as the basic error rate is below some threshhold"; Sec.2.3 p.21 "encode k qubits in n qubits"; Knill abstract p.1 "above 3 % per gate… resources… excessive"; Shor1995 abstract only | Multi-cite strengthens; Shor supports encoding half only (no threshold result) |
| C03 | CQEC amplifies coherence catalytically; threshold-free; catalyst reusable "across unlimited cycles" | Shiraishi2024, Wakaura2026unified, Wakaura2026catalyst | PS | Shiraishi Suppl. Sec.I p.4 "extremely small but non-zero coherence can be amplified arbitrarily, while exactly zero coherence never becomes non-zero"; Thm 2 p.3. Unified Sec.2.3 p.3 "catalyst is unchanged… enabling infinite reuse" **but** Sec.6.6 p.11 "guaranteed only in the asymptotic limit… drift O(1/√n) per cycle", Sec.8 p.18 "the catalyst is consumed rather than preserved", Table 14 p.14 F_cat^marg = 0.48–0.84 | **Overgeneralization** (reuse); Shiraishi proves sufficiency only, converse conjectured (Thm 3 p.4 rational-span only); Wakaura2026catalyst is not a catalyst paper (see C45) |
| C04 | Post-hoc CQEC "particularly attractive" for VQAs/iterative protocols | Peruzzo2014, Cerezo2021 | PS | Peruzzo p.3 "prepare unknown eigenvectors"; Cerezo abstract "classical optimizer to train a parametrized quantum circuit" | **Unsupported causal link** ("attractive" is authors' inference); **missing counterpoint**: Unified Sec.7.3 pp.16–17 says CQEC is post-hoc and cannot interleave; better cite Unified Sec.7.4 p.17 (VQE/QAOA use cases) |
| C05 | Target knowledge "identified as the principal conceptual gap" | Wakaura2026unified | PS | Sec.2.3 p.3, Sec.7.4 p.17 "CQEC assumes knowledge of ρ0", Table 17 p.17 | Phrase "principal conceptual gap" not in source (0 hits); source's stated gaps are theorem→protocol and copy count |
| C06 | Generic QST methods exist | Hradil1997, Gross2010, Haah2017 | SS | Hradil abstract p.1; Gross abstract p.1; Haah abstract p.1 | — |
| C07 | Four algorithms "from recent literature" | Chen2021, Ivashkov2024, Clinton2026, Regev2024 | PS | Instantiations appear verbatim in Unified Sec.5.1 p.8 ("four algorithms… d = 4–64"); primaries contain the algorithms, not the instances | Re-attribute instances to Unified Sec.5 |
| C08 | Definition of covariant channel | GourSpekkens2008, Marvian2014 | PS | G–S Eq.(15) p.7 "G-invariant if T(g)∘E∘T†(g) = E"; M–S Sec.II p.5 "U(1)-covariant super-operator" | Sources treat compact groups (U(1)); "time-translation symmetric" generic-H form is Shiraishi Def. S.2 (Suppl. p.2) — co-cite |
| C09 | Mode decomposition ρ = Σ_ω ρ^(ω), e^{−iωt} | Marvian2014 | PS | Sec.II p.5 Eq.(2.5) "U(θ)AU†(θ) = Σ_k e^{ikθ}A^(k)" | Source: integer modes k, sign e^{+ikθ}; manuscript generalizes to real ω with opposite sign — say "generalizing", fix sign; ω-set D(ρ) is Shiraishi Def. S.5 (Suppl. p.4) |
| C10 | D(ρ), resonant modes C(ρ)=⟨D(ρ)⟩_Z | Shiraishi2024 | SS | Main p.2 "all linear combinations of non-zero coherent modes with integer coefficients"; Def. S.5–S.6, Eqs.(4)–(5), Suppl. p.4 | Exact match |
| C11 | Covariant channels cannot create absent modes | Marvian2014 | SS | Prop.1 Eq.(2.11) p.6 "Modes(σ) ⊆ Modes(ρ)"; p.5 | Multi-cite: with catalyst, the no-go is Shiraishi Thm 3 p.4 — co-cite |
| C12 | Theorem 1 statement (sufficiency, correlated catalysis, ε-accuracy, exact iff target full rank) | Shiraishi2024 | SS | Thm 2 p.3 "convertible… with a correlated catalyst with an arbitrarily small error if C(ρ') ⊆ C(ρ)… exact if ρ' is full rank"; Thm S.24 Suppl. p.15 | Full-rank condition correctly on **target**; source uses δ (accuracy)/ε (correlation); omitted clause "correlation can be made arbitrarily small"; for pure targets exactness clause is vacuous |
| C13 | CQEC applies theorem to pair (ρn, ρt) | Wakaura2026unified | SS | Sec.2.3 p.3 Steps 0–2, Eq.(7) "Λ(ρnoisy ⊗ c) = τ, TrC[τ] ≈ ρ0, TrS[τ] = c" | Source numbers it "Theorem 2" |
| C14 | "ICEC protocol" builds catalyst with C(c)⊇C(ρe); d_c = d; joint cost O(d⁶) | Wakaura2026unified | PS | Sec.2.3 p.3 "Step 0… full-rank catalyst c with D(c) ⊇ D(ρ0)"; "simplified catalyst of dimension dC = 2^nC with nC = nS"; Sec.6.13 p.14 register nS+nC+nA, nA = 2 | "ICEC" never defined in source (only filename icec.py p.5, p.19); O(d⁶) not in source (authors' estimate; register has 2 ancillas); d_c = d is the simplified choice, formal catalyst "grows exponentially" (Eq.6 p.3); condition is D(c)⊇D(ρ0), not C(c)⊇C(ρe) |
| C15 | Trace norm / trace distance definitions (N1) | NielsenChuang2010 §9.2 | NF | Book offline | Standard; low risk |
| C16 | Variational characterization of ‖·‖₁ (N2) | Watrous2018 "Thm 1.14, Eq.(1.173)" | PS | Eq.(1.173) Sec.1.1.3 p.33 (Schatten duality); Eq.(1.182) p.34 "‖X‖₁ = max{|⟨U,X⟩| : U ∈ U(X)}" | **Wrong theorem number**: Thm 1.14 is Jensen's inequality (p.50). Hermitian −I≤Y≤I form is a corollary, not stated |
| C17 | Jordan decomposition (N3) | Watrous2018 §1.1.3 | SS | "Jordan–Hahn decompositions", Eqs.(1.143)–(1.144), pp.26–27 | Name is "Jordan–Hahn"; ‖A‖₁ = trA₊+trA₋ follows from Eq.(1.181) p.34 |
| C18 | Squared fidelity "convention of Nielsen–Chuang" (N4) | NielsenChuang2010 | NF | Book offline | **Likely misattribution**: N&C use unsquared F = tr√(ρ^{1/2}σρ^{1/2}); squared form is Jozsa 1994 — verify/re-attribute |
| C19 | Fuchs–van de Graaf inequality (N5) | FuchsGraaf1999 Thm 1 | SS | Theorem 1, Eq.(46), p.17 "1 − B ≤ ½Tr|ρ0−ρ1| ≤ √(1−B²)"; B defined Prop.4 Eq.(24) p.13 | State explicitly B = √F in manuscript convention |
| C20 | M removes modes no covariant channel can create | Marvian2014 | SS (cite) | Prop.1 Eq.(2.11) p.6 | **Internal consistency**: Theorem 1 guarantees integer combinations (Shiraishi p.2, Suppl. p.4 "1 and √2 … create 1+√2"); if M prunes outside D(ρn) rather than C(ρn) it over-prunes; stronger no-go is Shiraishi Thm 3 p.4 |
| C21 | Two-sided clipping bound false; |·| not operator-Lipschitz | Davies1988 | PS | Abstract: "In the particular case… f(u) = |u| this result is false for p = 1 and p = ∞" | **Missing counterpoint** from same abstract: logarithmically weaker bounds hold in finite dimension; 3×3 counterexamples are the manuscript's, not Davies's; full text paywalled |
| C22 | O(n^{-1/2}) concentration pointer | Tropp2015 §1 | SS | Sec.1.6.2 Thm 1.6.2 (Matrix Bernstein) p.8; Sec.1.6.3 p.11 | Tropp's bounds are spectral-norm; √d conversion is manuscript's |
| C23 | Haar populations ~ Dir(1,…,1) | ZyczkowskiSommers2001 §2 | SS | Sec.II.C p.3 after Eq.(2.8) "uniform on the simplex"; Eq.(2.9) p.4; Appendix A p.12 | Cite "Sec. II.C, Eqs.(2.8)–(2.9), Appendix A" |
| C24 | Moments E[p_i], Var, E[p_ip_j] | ZyczkowskiSommers2001 "Eq.(2.8) and Appendix" | PS | Distribution as C23; moments **not stated** (Eq.(2.8) is only λ_i = |V_{i1}|²; Sec.III.B Eqs.(3.10)–(3.20) pp.8–9 are HS-measure moments) | Values are correct Dirichlet facts but need a Dirichlet reference or "standard" |
| C25 | Coh-max is a special case of constrained QSE | Hradil1997 | PS | Abstract p.1 (positivity of ML); Eq.(3) p.2 | Hradil is ML, not constrained projection — soften |
| C26 | "Tomographic methods reconstruct from informationally complete measurements" | Gross2010, Haah2017 | PS / CT (Gross) | Gross abstract p.1 "O(rd log² d) measurement settings, compared to standard methods that require d² settings"; Thm 1 p.2 | **Contradicted for Gross**: compressed sensing is explicitly *not* IC; Haah uses collective POVM — rephrase |
| C27 | Acquiring ρn costs Θ(d²/ε²) shots | Haah2017 | SS (up to log) | Table I p.2 "n ≤ O(d²/ε²) ln(d/ε)" / "n ≥ Ω(d²/ε²)"; Thm 3 p.7 | Θ only up to ln(d/ε) (exact Θ: O'Donnell–Wright, cited Haah p.12); requires **collective** measurements — independent measurements need Ω(dr²/δ²) (abstract, Thm 4 p.7) |
| C28 | Multi-copy averaging is not standard QST | Hradil1997, Haah2017 | SS | Hradil abstract; Haah abstract | — |
| C29 | qDRIFT 3-qubit Heisenberg d=8 | Chen2021 | PS | Abstract p.1 "qDRIFT… random product formulas"; Sec.II.D p.7 Heisenberg chain (general n) | Instance from Unified Sec.5.1 p.8; credit Campbell PRL 123, 070503 (2019) for qDRIFT |
| C30 | QKAN, Chebyshev degree 3, approximating sin(πx), d=4 | Ivashkov2024 | PS | p.2 Chebyshev-polynomial edge activations; p.5 degree d; degree 3 only in Fig.10 pp.18–19 (Gaussian prep) | "sin(πx)" not in source (0 hits); **inconsistent** with companion Unified Sec.5.1 p.8 (T0..T3 at x=0.5) — reconcile |
| C31 | Control-free QPE, 3-site Fermi–Hubbard d=16 | Clinton2026 (arXiv:2410.21517) | PS | Abstract p.1 vectorial phase retrieval, Fermi-Hubbard spectrum; Sec.2.1.2 p.7 (1×5), p.9 (2×2) | No 3-site instance; verify bib author list against arXiv:2410.21517 |
| C32 | Regev factoring N=15, d=64 | Regev2024 | PS | Sec.3 p.3 discrete Gaussian state; p.4 exponentiation; p.5 QFT | "N = 15" not in source (theory paper); **missing counterpoint** (Unified Sec.7.4 p.17): knowing ρ0 implies knowing the factorization → CQEC circular |
| C33 | Gap-dependent dephasing from bath with J(ω) ∝ ω | NielsenChuang2010 | NF | Book offline | **Likely misattribution**: spin-boson result (Leggett 1987; Breuer–Petruccione), not N&C |
| C34 | C_ℓ1 definition; "standard monotone" | Baumgratz2014, Streltsov2017 | SS | Baumgratz Eq.(2) p.1 / Eq.(9) p.3; Streltsov Eq.(38) p.11 "fulfills C1–C4 for IO" | **Missing counterpoint**: Streltsov p.12 — C_ℓ1 violates C2 for DIO/MIO at d>2 (Bu & Xiong 2016); qualify "under IO" |
| C35 | Power-law ansatz motivated by O(1/n) infidelity in QST; "standard quantum limit" | Haah2017 | PS | p.2 "For d = 2 … optimal infidelity … 1/n [9–13] … generalized to qudits in [14]" | 1/n is Haah citing prior work, not Haah's result; holds for collective/adaptive measurements; "standard quantum limit" not in source |
| C36 | RB must pin dephasing rate to few % (γ dominates) | Emerson2005 | PS | p.1 "overall noise strength… estimated by a scalable experimental method" | "randomized benchmarking" not in source (0 hits); estimates average gate fidelity, not a dephasing rate; accuracy figure is manuscript's own result — add Knill 2008 / Magesan 2011 |
| C37 | Characterization of ZNE / PEC / VD | Temme2017, Li2017, Endo2018, Koczor2021, Huggins2021 | SS | Temme p.1 "extrapolation to the zero noise limit… Richardson"; Li p.1; Temme p.1 "quasi-probability distribution"; Endo p.1; Huggins p.1 "ρ^M/Tr(ρ^M) without explicitly preparing it"; Koczor p.1 | Multi-cite strengthens; VD/ESD are observable-level protocols (manuscript's state-level ρ²/Trρ² is an idealization — already stated); **missing counterpoint**: convergence to dominant eigenvector, "coherent mismatch" (Koczor p.8; Huggins p.1); Koczor's name is ESD |
| C38 | PEC overhead exponential; catalytic cost per Ref. | Wakaura2026unified | SS (cited part) | Sec.6.15 p.15 gate count "orders of magnitude more than surface-code overhead" | PEC half uncited: add Temme suppl. p.11 (γ_β = ∏γ_βℓ), Endo p.8 ((1+2ε)^{2N} repetitions); do not imply CQEC is cheaper — source concludes exponential copy overhead (μ ~ 2^{2n}e^{2γ}) |
| C39 | Tomography needs Θ(d²/ε²) copies | Haah2017 | SS (up to log) | Table I p.2; Thm 3 p.7 | Same flags as C27 |
| C40 | p_eff would come from RB in deployment | Emerson2005 | SS (substance) | p.1 "direct estimation of the average gate fidelity" | Global depolarizing ↔ average fidelity matches; naming flag as C36 |
| C41 | RB provides noise characterization at required accuracy | Emerson2005 | PS | Same passage p.1 | Isotropic noise strength only; no dephasing-specific accuracy |
| C42 | Positioning vs stabilizer QEC | Shor1995, Gottesman1997 | SS | Gottesman Ch.1 p.12; Shor abstract | Generic; adequate |
| C43 | PQES: SWAP-test purification, full record, no postselection, no noise model | Raghoonanan2026 (v2) | SS | Abstract p.1 "multiple noisy copies of an unknown state"; "full outcome record… without postselection"; Sec.V p.9 "independently of any particular error model" | Name PQES ✓; bib title should be v2 title; **counterpoint to state**: thresholds p_th = 3/4 (depol.), 1/2 (dephasing) (p.1) |
| C44 | Explicit CPTP map "with verifiable catalyst preservation" exists in Ref. | Wakaura2026unified | PS | Sec.6.13 p.14 Eq.(24) covariance ‖[U,H_total]‖ = 0, Choi positivity, catalyst marginal; **but** p.14 "catalyst marginal degrades substantially (F = 0.48–0.84)… genuine correlated catalysis… is not achieved" | **Misleading**: check exists and *fails*; construction d ≤ 8 only |
| C45 | Companion preprints: catalyst-preparation methods in Wakaura2026catalyst (rs-9366084) | Wakaura2026unified, Wakaura2026catalyst, Raghoonanan2026 | **CT** | doi 10.21203/rs.3.rs-9366084/v1 = "Biologically Plausible Quantum Error Correction in a Three-Layer Neural Spin Model" (12 Apr 2026); catalyst preparation is Unified Sec.4 pp.5–8; "ICEC" appears only in rs-9366084 Sec.III.E p.7 | **Wrong DOI/description**; "unified CQEC and ICEC framework" — Unified never defines ICEC |
| C46 | Interface acts on system density matrix alone | Wakaura2026unified | SS | Sec.3.4 p.5 "acts on the system density matrix alone; the catalyst enters only through scalar summaries" | Carry source caveat p.5: "target-parameterized and nonlinear, so it is not itself a CPTP channel" |
| C47 | Open problem: integrate Ref.'s CPTP map end-to-end | Wakaura2026unified | SS | Abstract p.1 "explicitly CPTP, exactly covariant joint-channel… 0.54 → 0.77"; Sec.6.13 p.14 | Construction includes ancilla, d ≤ 8, does not achieve catalysis — "integrated end-to-end" is a goal |

## Corrections required before submission (ordered by severity)

1. **C45 — wrong reference.** rs-9366084 is the neural-spin-model preprint. Cite Unified Sec. 4 for catalyst preparation (or the correct preprint); drop rs-9366084 as support for catalyst reuse (C03).
2. **C14/C45 — "ICEC".** Not defined in Unified; use "CQEC" or define ICEC from rs-9366084 Sec. III.E.
3. **C16 — Watrous.** "Thm 1.14" → "Sec. 1.1.3, Eqs. (1.173) and (1.182)". C17: "Jordan–Hahn".
4. **C18, C33 — Nielsen–Chuang.** Squared-fidelity convention (→ Jozsa 1994) and ohmic-bath dephasing (→ Leggett 1987 / Breuer–Petruccione) are likely misattributed; verify or re-cite.
5. **C24 — ZS 2001.** Cite for the Dirichlet distribution only (Sec. II.C, Eqs. 2.8–2.9, App. A); moments need a Dirichlet reference.
6. **C26 — Gross.** Compressed sensing is not informationally complete; rephrase.
7. **C27/C39/C35 — Haah.** Θ up to ln(d/ε); collective vs independent measurements (Ω(dr²/δ²)); 1/n is cited prior work; drop "standard quantum limit"; co-cite O'Donnell–Wright.
8. **C03, C44, C47 — catalyst reuse/preservation.** State as asymptotic/in-principle; report that the source's own CPTP test shows the catalyst is consumed (F_cat^marg 0.48–0.84, d ≤ 8).
9. **C05 — "principal conceptual gap".** Not in source; rephrase to Table 17 / Sec. 7.4 wording.
10. **C07, C29–C32 — algorithm instances.** Attribute to Unified Sec. 5.1; reconcile QKAN target (sin(πx) vs T0..T3 at x = 0.5); credit Campbell 2019 for qDRIFT; add Unified's circularity caveat for factoring; verify Clinton2026 authors.
11. **C36/C40/C41 — RB.** Emerson 2005 never says "randomized benchmarking"; add Knill 2008 / Magesan 2011; make clear the few-% γ requirement is your sensitivity result.
12. **C08/C09/C11/C20 — asymmetry sources.** Co-cite Shiraishi Def. S.2 (time-translation covariance) and Thm 3 (catalytic no-go); mark ω-decomposition as a generalization of integer U(1) modes and fix the sign convention; ensure M prunes relative to the lattice C(ρn), not D(ρn).
13. **C34 — C_ℓ1.** Monotone "under IO"; note DIO/MIO failure for d > 2 (Streltsov p.12).
14. **C21 — Davies.** Add the log-weaker finite-d bound; mark the 3×3 counterexamples as yours.
15. **C38 — PEC overhead.** Add Temme suppl. p.11 and Endo p.8; avoid implying CQEC is cheaper (Unified Sec. 6.15).
16. **C43 — PQES.** Bib title to v2; state the source's thresholds when contrasting.
17. **C04 — VQA motivation.** Hedge "particularly attractive"; cite Unified Sec. 7.4; acknowledge the post-hoc/no-interleaving caveat (Sec. 7.3).
18. **C02 — Shor 1995** supports encoding only; threshold rests on Gottesman/Knill.

## Status (2026-09-14)
Corrections 1–9, 11, 13–15, 17–18 applied to `paper_blind_cqec_qst.tex` / `.bib`; #10 (QKAN attribution), #12 (lattice-based pruning) and #16 (PQES eprint/url) were applied only on 2026-09-15 after the adversarial accuracy panel. Compiles clean; 0 "ICEC"; the CLAUDE.md hygiene grep returns 5 raw hits, all benign technical uses ("uncorrected", "corrected expectation values"), 0 violations. Audit-side errata: C08 location is Prop. S.3 (Suppl. p.3), not Def. S.2 p.2; C12 summary cell should read "exact *if* target full rank"; C27 flag: Haah Thm 4 full-rank clause has no log factor; C45 posting date is 13 Apr 2026. New references: Jozsa1994, Leggett1987, BreuerPetruccione2002, Campbell2019, Knill2008, Magesan2011, ODonnellWright2016; Wakaura2026catalyst entry corrected to its actual title/author.

## Addendum (2026-10-06): 20 related-work references added, 2022–2025

Each entry was fetched as a primary source with `hyperresearch fetch` (vault tag `cqec-refs`, notes under `~/research/notes/`), and title, author list, venue, volume/pages, and DOI were read from the arXiv record, the publisher page, or Crossref (noted per entry in the vault). Where the paper is cited for a specific claim, the supporting abstract sentence is recorded in the vault note summary. Two corrections made during verification: Liao et al. 2024 has no Czarnik among its authors; the arXiv `journal_ref` of Google Quantum AI 2023 misprints the first page (676, not 678; that paper is not cited in the end).

| Key | Venue | Used for | Status |
|---|---|---|---|
| Cai2023qem | Rev. Mod. Phys. 95, 045005 | QEM review | Strongly Supported |
| Takagi2022limits | npj Quantum Inf. 8, 114 | exponential QEM sampling overhead | Strongly Supported |
| Quek2024tighter | Nat. Phys. 20, 1648 | superpolynomial samples at shallow depth | Strongly Supported |
| Tsubouchi2023costbound | PRL 131, 210601 | estimation-theoretic QEM cost bound | Strongly Supported |
| Takagi2023sampling | PRL 131, 210602 | universal sampling lower bound | Strongly Supported |
| vandenBerg2023pec | Nat. Phys. 19, 1116 | PEC with learned noise models; QEM targets expectation values | Strongly Supported |
| Kim2023utility | Nature 618, 500 | ZNE at 127 qubits | Strongly Supported |
| OBrien2023purification | Nat. Phys. 19, 1787 | purification/VD in experiment | Strongly Supported |
| Liao2024mlqem | Nat. Mach. Intell. 6, 1478 | learning-based QEM | Strongly Supported |
| GoogleQuantumAI2025belowthreshold | Nature 638, 920 | below-threshold surface code, 101 qubits/logical | Strongly Supported |
| LipkaBartosik2024catalysis | Rev. Mod. Phys. 96, 025005 | catalysis review | Strongly Supported |
| Datta2023catalysis | Rep. Prog. Phys. 86, 116002 | catalysis review | Strongly Supported |
| Takagi2022correlation | PRL 128, 240501 | correlated catalysts enable arbitrary coherence manipulation | Strongly Supported |
| Rubboli2022limits | PRL 129, 120506 | catalyst must be resourceful when residual correlation is small | Strongly Supported |
| Marvian2022restrictions | Nat. Phys. 18, 283 | symmetry restricts realizable operations; circumvented by catalysis | Strongly Supported |
| Kondra2024coherence | PRL 132, 200201 | catalysis + free energy amplifies coherence arbitrarily | Strongly Supported |
| Huang2023learning | PRX Quantum 4, 040337 | learning unknown processes from data | Strongly Supported |
| Elben2023randomized | Nat. Rev. Phys. 5, 9 | randomized measurements / classical shadows | Strongly Supported |
| Anshu2024survey | Nat. Rev. Phys. 6, 59 | survey of state-learning complexity | Strongly Supported |
| Chen2023adaptivity | FOCS 2023, 391 | Ω(d³/ε²) for incoherent (even adaptive) measurements | Strongly Supported |

Verified but not cited: Yamaguchi & Tajima PRL 131, 200203 (2023) (QFI-based asymmetry; tangential) and Google Quantum AI, Nature 614, 676 (2023) (superseded by the 2025 result in the text).

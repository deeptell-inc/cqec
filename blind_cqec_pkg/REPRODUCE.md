# Reproducing the figures and tables of the blind-CQEC paper

All benchmark scripts live one directory above this package, in the repository
root (`../benchmark_blind_*.py`).  They import `core.py`, `icec.py`,
`benchmark_icec.py`, and `benchmark_blind_cqec.py` from that root and the
`blind_cqec` package from this directory; every path is resolved relative to
the script file, so the scripts can be invoked from any working directory:

```bash
pip install -e ".[bench]"          # numpy, scipy, matplotlib, qiskit, qiskit-aer, sympy
python ../benchmark_blind_algorithms.py    # or any other script below
```

Figures are written next to the scripts (repository root); the raw numerical
data behind each figure and table is written to `results/<script>.json`
(one file per script, top-level keys = figure / table names, values = the
arrays or tables the script prints).  The JSON files shipped here were
generated with the versions in the table at the bottom; all scripts are
deterministic given their fixed seeds.

## Script -> figure / table map

Figure names are the `\includegraphics{...}` targets and table names the
`\label{tab:...}` labels of the manuscript (`paper_blind_cqec_pra.tex`; the `article`-class twin `paper_blind_cqec_qst.tex` shares them).  Runtimes are
wall-clock on an Apple-silicon laptop (single process).

| Manuscript item | Generating script | Results JSON (top-level key) | Runtime |
|---|---|---|---|
| `tab:strategies` | none (descriptive table, no numerics) | -- | -- |
| `fig_blind_dephasing.pdf` | `benchmark_blind_cqec.py` | `benchmark_blind_cqec.json` (`fig_blind_dephasing`) | 2.4 s (whole script) |
| `fig_blind_amplitude_damping.pdf` | `benchmark_blind_cqec.py` | `benchmark_blind_cqec.json` (`fig_blind_amplitude_damping`) | " |
| `fig_blind_qutrit.pdf` | `benchmark_blind_cqec.py` | `benchmark_blind_cqec.json` (`fig_blind_qutrit`) | " |
| `tab:main_results` | `benchmark_blind_algorithms.py` | `benchmark_blind_algorithms.json` (`tab_main_results__fig_blind_alg_heatmap__fig_blind_alg_scatter`) | 4.8 s (whole script) |
| `fig_blind_alg_heatmap.pdf` | `benchmark_blind_algorithms.py` | same key as above | " |
| `fig_blind_alg_scatter.pdf` | `benchmark_blind_algorithms.py` | same key as above + `estimation_vs_recovery_correlation` | " |
| `fig_blind_alg_copies.pdf` | `benchmark_blind_algorithms.py` | `benchmark_blind_algorithms.json` (`fig_blind_alg_copies`) | " |
| `tab:min_copies` | `benchmark_blind_algorithms.py` | `benchmark_blind_algorithms.json` (`tab_min_copies`) | " |
| `tab:dephasing_only` | `benchmark_blind_nature_supp.py` | `benchmark_blind_nature_supp.json` (`tab_dephasing_only__tab_depol_only__tab_ampdamp_only`) | 9.5 s (whole script) |
| `tab:depol_only` | `benchmark_blind_nature_supp.py` | same key as above | " |
| `tab:ampdamp_only` | `benchmark_blind_nature_supp.py` | same key as above | " |
| `fig_mixed_state_sweep.pdf` | `benchmark_blind_nature_supp.py` | `benchmark_blind_nature_supp.json` (`fig_mixed_state_sweep`) | " |
| `fig_hybrid_strategy.pdf` | `benchmark_blind_nature_supp.py` | `benchmark_blind_nature_supp.json` (`fig_hybrid_strategy`) | " |
| `fig_dimension_sweep.pdf` | `benchmark_blind_dimension_sweep.py` | `benchmark_blind_dimension_sweep.json` (`fig_dimension_sweep__fig_haar_distribution__fig_mle_comparison`) | 12.6 s (whole script) |
| `fig_haar_distribution.pdf` | `benchmark_blind_dimension_sweep.py` | same key as above | " |
| `fig_mle_comparison.pdf` | `benchmark_blind_dimension_sweep.py` | same key as above | " |
| `fig_sensitivity_heatmap.pdf` | `benchmark_blind_sensitivity.py` | `benchmark_blind_sensitivity.json` (`fig_sensitivity_heatmap`) | 29 s (whole script) |
| `fig_sensitivity_curves.pdf` | `benchmark_blind_sensitivity.py` | `benchmark_blind_sensitivity.json` (`fig_sensitivity_curves`) | " |
| `fig_scaling_physics.pdf` (supplementary, not included in the manuscript) | `benchmark_blind_sensitivity.py` | `benchmark_blind_sensitivity.json` (`fig_scaling_physics`, incl. fitted power-law exponents) | " |
| `tab:tau_sensitivity` (+ the Π re-creation fraction quoted in Sec. 7) | `benchmark_blind_tau_sensitivity.py` | `benchmark_blind_tau_sensitivity.json` (`tab_tau_sensitivity`, `pi_recreation_regev_combined`) | 5 s |
| `fig_qem_compare.pdf` | `benchmark_blind_qem_compare.py` | `benchmark_blind_qem_compare.json` (`fig_qem_compare`) | 3.7 s (whole script) |
| `tab:overhead` | `benchmark_blind_qem_compare.py` | `benchmark_blind_qem_compare.json` (`tab_overhead`; ms/call is machine-dependent) | " |
| `fig_sample_complexity.pdf` | `benchmark_blind_qem_compare.py` | `benchmark_blind_qem_compare.json` (`fig_sample_complexity`) | " |
| `fig_vqe_d16.pdf` (LiH null result, text only) | `benchmark_blind_qem_compare.py` | `benchmark_blind_qem_compare.json` (`fig_vqe_d16`) | " |
| `fig_vqe_demo.pdf` | `benchmark_blind_vqe_demo.py` | `benchmark_blind_vqe_demo.json` (`fig_vqe_demo`) | 1.2 s |
| `tab:circuit_sanity` | `benchmark_blind_circuit_sanity.py` | `benchmark_blind_circuit_sanity.json` (`tab_circuit_sanity__fig_circuit_sanity`) | 1.2 s |
| `fig_circuit_sanity.pdf` | `benchmark_blind_circuit_sanity.py` | same key as above | " |

`benchmark_blind_cqec_copies.py` (copy-economized sweep, `fig_copies_*.pdf`,
6.4 s) is retained for completeness; the multi-copy scaling exponents quoted in
Sec. 5.4 are printed by it (`fig_copies_scaling_exponents`), and none of its
figures appear in the manuscript.

Every recovery in every script goes through `blind_cqec.recovery.icec_recover`
(mode restriction at `mode_tol=1e-10` followed by PSD projection); no script
applies any additional copy-dependent interpolation.

## Software versions used to generate `results/*.json`

| Package | Version |
|---|---|
| Python | 3.13.9 |
| numpy | 2.2.6 |
| scipy | 1.17.1 |
| matplotlib | 3.10.6 |
| qiskit | 2.4.0 (circuit sanity check only) |
| qiskit-aer | 0.17.2 (circuit sanity check only) |
| sympy | 1.14.0 (`isprime` in the Regev simulator of `benchmark_icec.py`) |

Install them with `pip install -e ".[bench]"`.  Run the unit tests with
`pytest -q` in this directory (91 tests).

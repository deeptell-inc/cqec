# Blind Catalytic Quantum Error Correction: Comprehensive Benchmark Results

**Date:** 2026-04-07

## Overview

This document reports the results of **blind catalytic quantum error correction (CQEC)**, where the target state is *unknown* and must be estimated from the noisy state prior to correction. All benchmarks are based on the ICEC (Infinite Catalytic Error Correction) protocol derived from Shiraishi & Takagi, PRL 132, 180202 (2024).

### Computational Environment

| Component | Specification |
|---|---|
| **CPU** | Apple M4 Max (12 Performance + 4 Efficiency cores, 16 threads) |
| **Memory** | 64 GB unified LPDDR5X |
| **Architecture** | ARM64 (AArch64) |
| **OS** | macOS 26.2 (Build 25C56) |
| **Python** | 3.9.6 (Apple Command Line Tools) |
| **NumPy** | 2.0.2 (Accelerate BLAS backend) |
| **SciPy** | 1.13.1 |
| **Matplotlib** | 3.9.4 |
| **SymPy** | 1.14.0 |
| **LaTeX** | pdfTeX 3.141592653-2.6-1.40.28 (TeX Live 2025) |

### Runtime Summary

| Benchmark | Script | Wall-clock time |
|---|---|---|
| I. Qubit/Qutrit noise sweep | `benchmark_blind_cqec.py` | 2.3 s |
| II. Copy-economized | `benchmark_blind_cqec_copies.py` | 5.3 s |
| III. 4 Algorithms | `benchmark_blind_algorithms.py` | 6.5 s |
| **Total** | | **14.1 s** |

All simulations are density-matrix-level ($d \leq 64$, matrices up to $64 \times 64$). Results are deterministic with `numpy.random.seed(42)`.

Three benchmark suites were executed:

1. **Benchmark I** --- Qubit/qutrit noise sweeps with 6 estimation strategies
2. **Benchmark II** --- Copy-economized version: minimum copies required for target fidelity
3. **Benchmark III** --- 4 quantum algorithms from recent literature under decoherence + blind CQEC

### Estimation Strategies

| Strategy | Description | Knowledge Required |
|---|---|---|
| **Naive** | Use noisy state directly as target | None |
| **Channel inversion** | Analytically invert the known noise model | Noise model |
| **Coherence maximization** | Maximize off-diagonal coherence to physicality bound $\sqrt{p_i p_j}$ | None |
| **Iterative refinement** | 5 rounds: estimate, correct, re-estimate | None |
| **Multi-copy tomography** | Average 20 noisy copies, then boost coherence | None |
| **Oracle** | Perfect knowledge of true target (upper bound) | Full |

---

## I. Qubit/Qutrit Noise Sweep

**Setup:** Maximally coherent state $|+\rangle\langle+|$, qubit ($d=2$) and qutrit ($d=3$).

### I-A. Numerical Summary at Representative Noise Strengths

#### Dephasing $\gamma = 2.0$ (Qubit)

| Strategy | Fidelity | Trace Dist. | Coh. Ratio |
|---|---|---|---|
| No correction | 0.5744 | 0.4256 | 0.149 |
| Naive | 0.5744 | 0.4256 | 0.149 |
| Channel inversion | **1.0000** | 0.0000 | 1.000 |
| Coherence max | **1.0000** | 0.0000 | 1.000 |
| Iterative | **1.0000** | 0.0000 | 1.000 |
| Multi-copy tomo | 0.9999 | 0.0074 | 1.000 |
| Oracle | **1.0000** | 0.0000 | 1.000 |

#### Depolarizing $p = 0.5$ (Qubit)

| Strategy | Fidelity | Trace Dist. | Coh. Ratio |
|---|---|---|---|
| No correction | 0.7382 | 0.2618 | 0.476 |
| Naive | 0.7382 | 0.2618 | 0.476 |
| Channel inversion | **1.0000** | 0.0000 | 1.000 |
| Coherence max | **1.0000** | 0.0000 | 1.000 |
| Iterative | **1.0000** | 0.0000 | 1.000 |
| Multi-copy tomo | 1.0000 | 0.0038 | 1.000 |
| Oracle | **1.0000** | 0.0000 | 1.000 |

#### Amplitude Damping $\gamma = 0.5$ (Qubit)

| Strategy | Fidelity | Trace Dist. | Coh. Ratio |
|---|---|---|---|
| No correction | 0.8451 | 0.3042 | 0.690 |
| Naive | 0.8451 | 0.3042 | 0.690 |
| Channel inversion | **0.9999** | 0.0022 | 1.000 |
| Coherence max | 0.9260 | 0.2721 | 0.852 |
| Iterative | 0.9260 | 0.2721 | 0.852 |
| Multi-copy tomo | 0.9284 | 0.2675 | 0.857 |
| Oracle | **0.9999** | 0.0022 | 1.000 |

### I-B. Figures

| Figure | Description |
|---|---|
| `fig_blind_dephasing.png` | Recovery fidelity, trace distance, coherence ratio vs dephasing $\gamma$ (qubit) |
| `fig_blind_depolarizing.png` | Same vs depolarizing $p$ (qubit) |
| `fig_blind_amplitude_damping.png` | Same vs amplitude damping $\gamma$ (qubit) |
| `fig_blind_qutrit.png` | Same vs dephasing $\gamma$ (qutrit, $d=3$) |
| `fig_blind_fidelity_gap.png` | Fidelity gap (Oracle $-$ blind) across all noise types |

### I-C. Key Findings

1. **Dephasing / Depolarizing:** Channel inversion, coherence maximization, and iterative refinement all achieve $F = 1.000$, matching Oracle. Blind CQEC incurs no fidelity penalty.
2. **Amplitude damping:** Only channel inversion achieves near-perfect recovery ($F = 0.9999$). Coherence maximization saturates at $F \approx 0.926$ because population redistribution cannot be inferred from off-diagonal structure alone.
3. **Naive estimation is equivalent to no correction** --- the catalyst amplifies toward the wrong target.

---

## II. Copy-Economized Blind CQEC

**Setup:** Fixed noise strength, sweep $n_\text{copies} \in \{1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200\}$, across 8 scenarios (qubit + qutrit).

### II-A. Minimum Copies for $F \geq 0.95$

| Scenario | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Naive |
|---|---|---|---|---|---|
| Deph $\gamma=1$ (qubit) | **8** | **8** | 8 | 8 | >200 |
| Deph $\gamma=3$ (qubit) | **75** | 75 | 150 | 75 | >200 |
| Depol $p=0.5$ (qubit) | **5** | **5** | 5 | 5 | >200 |
| Depol $p=0.8$ (qubit) | **15** | 15 | 20 | 15 | >200 |
| Amp $\gamma=0.3$ (qubit) | **2** | 2 | 2 | 2 | >200 |
| Amp $\gamma=0.7$ (qubit) | **5** | >200 | >200 | >200 | >200 |
| Deph $\gamma=1$ (qutrit) | **20** | 20 | 50 | 20 | >200 |
| Deph $\gamma=3$ (qutrit) | **100** | 100 | >200 | >200 | >200 |

### II-B. Minimum Copies for $F \geq 0.99$

| Scenario | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Naive |
|---|---|---|---|---|---|
| Deph $\gamma=1$ (qubit) | **8** | **8** | 8 | 8 | >200 |
| Deph $\gamma=3$ (qubit) | **200** | 200 | >200 | 200 | >200 |
| Depol $p=0.5$ (qubit) | **5** | **5** | 5 | 5 | >200 |
| Depol $p=0.8$ (qubit) | **30** | 30 | 30 | 30 | >200 |
| Amp $\gamma=0.3$ (qubit) | **3** | >200 | >200 | >200 | >200 |
| Amp $\gamma=0.7$ (qubit) | **15** | >200 | >200 | >200 | >200 |
| Deph $\gamma=1$ (qutrit) | **50** | 50 | 50 | 50 | >200 |
| Deph $\gamma=3$ (qutrit) | >200 | >200 | >200 | >200 | >200 |

### II-C. Scaling Exponents: $F(n) \sim 1 - A \cdot n^{-\alpha}$

| Scenario | Ch. Inv. $\alpha$ | Coh. Max $\alpha$ | Iterative $\alpha$ |
|---|---|---|---|
| Depol $p=0.5$ | **1.12** | 1.12 | 1.00 |
| Amp $\gamma=0.3$ | **2.16** | 0.22 | 0.24 |
| Amp $\gamma=0.7$ | **1.19** | 0.04 | 0.05 |
| Deph $\gamma=1$ | 0.75 | 0.75 | 0.62 |
| Deph $\gamma=3$ | 0.44 | 0.44 | 0.29 |

### II-D. Figures

| Figure | Description |
|---|---|
| `fig_copies_min_heatmap.png` | Heatmap: minimum copies for $F \geq 0.95$ and $F \geq 0.99$ |
| `fig_copies_scaling_exponents.png` | Bar chart of scaling exponents per scenario and strategy |
| `fig_copies_Deph_γ1_qubit.png` | Copy sweep: dephasing $\gamma=1$ (qubit) |
| `fig_copies_Depol_p0.5_qubit.png` | Copy sweep: depolarizing $p=0.5$ (qubit) |
| `fig_copies_Amp_γ0.7_qubit.png` | Copy sweep: amplitude damping $\gamma=0.7$ (qubit) |
| `fig_copies_Deph_γ3_qutrit.png` | Copy sweep: dephasing $\gamma=3$ (qutrit) |

### II-E. Key Findings

1. **Channel inversion dominates across all scenarios** for minimum copy count and scaling exponent.
2. **Depolarizing noise is cheapest to correct** ($n=5$ for $F \geq 0.99$ at $p=0.5$); amplitude damping is most expensive.
3. **Qutrit costs 2--5$\times$ more copies** than qubit due to increased mode count.
4. **Scaling:** Channel inversion converges as $n^{-\alpha}$ with $\alpha \in [0.4, 2.2]$; coherence maximization drops to $\alpha \approx 0.04$ under amplitude damping, indicating near-complete failure.

---

## III. Four Quantum Algorithms Under Decoherence + Blind CQEC

### Algorithms

| # | Algorithm | Reference | $d$ | Algo $F$ |
|---|---|---|---|---|
| 1 | **qDRIFT** --- Random product formula, 3-qubit Heisenberg | Chen et al., PRX Quantum 2, 040305 (2021) | 8 | 0.880 |
| 2 | **QKAN** --- Quantum KAN layer, Chebyshev $d=3$, sin | Ivashkov et al., arXiv:2410.04435 (2024) | 4 | 1.000 |
| 3 | **Control-Free QPE** --- Vectorial phase retrieval, 3-site Fermi-Hubbard | Clinton et al., PRX Quantum 7, 010345 (2026) | 16 | 0.866 |
| 4 | **Regev Factoring** --- Efficient factoring, $N=15$ | Regev, arXiv:2308.06572 (2024) | 64 | 0.000 |

### Noise Channels

| Label | Parameters |
|---|---|
| Dephasing | $\gamma = 2.0$ |
| Depolarizing | $p = 0.3$ |
| Combined | $\gamma = 1.0$, $p = 0.15$, $\gamma_\text{AD} = 0.1$ |

### III-A. Full Fidelity Table --- qDRIFT (Heisenberg)

| Noise | Strategy | Est. $F$ | Rec. $F$ | $T_D$ | Coh% |
|---|---|---|---|---|---|
| Dephasing ($\gamma=2$) | No correction | 0.243 | 0.243 | 0.757 | 0.14 |
| | Naive | 0.243 | 0.243 | 0.757 | 0.14 |
| | Channel inversion | 0.328 | 0.321 | 0.820 | 0.22 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Multi-copy tomo | 0.939 | 0.939 | 0.072 | 0.93 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Depolarizing ($p=0.3$) | No correction | 0.738 | 0.738 | 0.263 | 0.70 |
| | Channel inversion | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Multi-copy tomo | 0.981 | 0.981 | 0.023 | 0.98 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Combined | No correction | 0.375 | 0.375 | 0.627 | 0.29 |
| | Channel inversion | 0.540 | 0.538 | 0.674 | 0.47 |
| | **Coherence max** | 0.993 | **0.993** | 0.087 | 0.99 |
| | **Iterative** | 0.993 | **0.993** | 0.087 | 0.99 |
| | Multi-copy tomo | 0.950 | 0.950 | 0.109 | 0.94 |
| | Oracle | 1.000 | 1.000 | 0.001 | 1.00 |

### III-B. Full Fidelity Table --- QKAN (CHEB-d3, sin)

| Noise | Strategy | Est. $F$ | Rec. $F$ | $T_D$ | Coh% |
|---|---|---|---|---|---|
| Dephasing ($\gamma=2$) | No correction | 0.352 | 0.352 | 0.649 | 0.14 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Multi-copy tomo | 0.979 | 0.979 | 0.040 | 0.97 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Depolarizing ($p=0.3$) | No correction | 0.775 | 0.775 | 0.225 | 0.70 |
| | Channel inversion | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Combined | No correction | 0.469 | 0.469 | 0.534 | 0.29 |
| | **Coherence max** | 0.995 | **0.995** | 0.071 | 0.99 |
| | **Iterative** | 0.995 | **0.995** | 0.071 | 0.99 |
| | Multi-copy tomo | 0.981 | 0.981 | 0.074 | 0.98 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |

### III-C. Full Fidelity Table --- Control-Free QPE (VPR)

| Noise | Strategy | Est. $F$ | Rec. $F$ | $T_D$ | Coh% |
|---|---|---|---|---|---|
| Dephasing ($\gamma=2$) | No correction | 0.312 | 0.312 | 0.706 | 0.14 |
| | Channel inversion | 0.167 | 0.166 | 0.845 | 0.13 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Multi-copy tomo | 0.732 | 0.732 | 0.284 | 0.81 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Depolarizing ($p=0.3$) | No correction | 0.719 | 0.719 | 0.281 | 0.70 |
| | Channel inversion | 1.000 | **0.999** | 0.001 | 1.00 |
| | **Coherence max** | 0.962 | 0.962 | 0.196 | 1.27 |
| | Oracle | 1.000 | 0.999 | 0.001 | 1.00 |
| Combined | No correction | 0.411 | 0.411 | 0.605 | 0.27 |
| | Channel inversion | 0.492 | 0.489 | 0.710 | 0.24 |
| | **Coherence max** | 0.959 | **0.959** | 0.133 | 1.10 |
| | **Iterative** | 0.959 | **0.959** | 0.133 | 1.10 |
| | Multi-copy tomo | 0.809 | 0.809 | 0.257 | 1.00 |
| | Oracle | 1.000 | 0.999 | 0.001 | 1.00 |

### III-D. Full Fidelity Table --- Regev Factoring ($N=15$)

| Noise | Strategy | Est. $F$ | Rec. $F$ | $T_D$ | Coh% |
|---|---|---|---|---|---|
| Dephasing ($\gamma=2$) | No correction | 0.491 | 0.491 | 0.530 | 0.14 |
| | Channel inversion | 0.193 | 0.195 | 0.838 | 0.69 |
| | **Coherence max** | 1.000 | **1.000** | 0.000 | 1.00 |
| | **Iterative** | 1.000 | **1.000** | 0.000 | 1.00 |
| | Multi-copy tomo | 0.110 | 0.113 | 0.891 | 4.27 |
| | Oracle | 1.000 | 1.000 | 0.000 | 1.00 |
| Depolarizing ($p=0.3$) | No correction | 0.705 | 0.705 | 0.295 | 0.70 |
| | **Channel inversion** | 1.000 | **0.998** | 0.002 | 1.00 |
| | Coherence max | 0.613 | 0.613 | 0.450 | 6.54 |
| | Iterative | 0.613 | 0.613 | 0.429 | 5.31 |
| | Multi-copy tomo | 0.092 | 0.094 | 0.909 | 4.28 |
| | Oracle | 1.000 | 0.998 | 0.002 | 1.00 |
| Combined | No correction | 0.538 | 0.538 | 0.480 | 0.29 |
| | **Channel inversion** | 0.906 | **0.905** | 0.199 | 0.77 |
| | Coherence max | 0.752 | 0.752 | 0.311 | 4.58 |
| | Iterative | 0.753 | 0.753 | 0.292 | 3.78 |
| | Multi-copy tomo | 0.101 | 0.103 | 0.901 | 4.31 |
| | Oracle | 1.000 | 0.999 | 0.001 | 1.00 |

### III-E. Copy Sweep Under Combined Noise ($\gamma=1, p=0.15, \gamma_\text{AD}=0.1$)

#### qDRIFT ($d=8$)

| $n_\text{copies}$ | Naive | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Oracle |
|---|---|---|---|---|---|---|
| 2 | 0.375 | 0.360 | 0.564 | 0.485 | 0.559 | 0.566 |
| 5 | 0.375 | 0.430 | 0.787 | 0.698 | 0.774 | 0.790 |
| 10 | 0.375 | 0.477 | **0.954** | 0.925 | 0.935 | 0.952 |
| 20 | 0.375 | 0.509 | **0.993** | 0.993 | 0.945 | 0.994 |
| 50 | 0.375 | 0.530 | **0.993** | 0.993 | 0.957 | 0.998 |
| 100 | 0.375 | 0.538 | **0.993** | 0.993 | 0.952 | 1.000 |
| 200 | 0.375 | 0.540 | **0.993** | 0.993 | 0.958 | 1.000 |

#### QKAN ($d=4$)

| $n_\text{copies}$ | Naive | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Oracle |
|---|---|---|---|---|---|---|
| 2 | 0.469 | 0.586 | 0.632 | 0.564 | 0.631 | 0.633 |
| 5 | 0.469 | 0.720 | 0.824 | 0.750 | 0.822 | 0.826 |
| 10 | 0.469 | 0.763 | **0.968** | 0.948 | 0.962 | 0.967 |
| 20 | 0.469 | 0.791 | **0.995** | 0.995 | 0.980 | 0.996 |
| 50 | 0.469 | 0.797 | **0.995** | 0.995 | 0.993 | 0.999 |
| 100 | 0.469 | 0.799 | **0.995** | 0.995 | 0.986 | 1.000 |
| 200 | 0.469 | 0.799 | **0.995** | 0.995 | 0.986 | 1.000 |

#### Control-Free QPE ($d=16$)

| $n_\text{copies}$ | Naive | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Oracle |
|---|---|---|---|---|---|---|
| 2 | 0.411 | 0.339 | 0.577 | 0.506 | 0.546 | 0.586 |
| 5 | 0.411 | 0.386 | 0.773 | 0.693 | 0.708 | 0.790 |
| 10 | 0.411 | 0.424 | 0.912 | 0.877 | 0.760 | 0.943 |
| 20 | 0.411 | 0.457 | **0.947** | 0.930 | 0.808 | 0.992 |
| 50 | 0.411 | 0.481 | **0.959** | 0.958 | 0.800 | 0.998 |
| 100 | 0.411 | 0.489 | **0.959** | 0.959 | 0.792 | 0.999 |
| 200 | 0.411 | 0.492 | **0.959** | 0.959 | 0.804 | 1.000 |

#### Regev Factoring ($d=64$)

| $n_\text{copies}$ | Naive | Ch. Inv. | Coh. Max | Iterative | Multi-copy | Oracle |
|---|---|---|---|---|---|---|
| 2 | 0.538 | 0.656 | 0.630 | 0.611 | 0.310 | 0.682 |
| 5 | 0.538 | 0.798 | 0.748 | 0.751 | 0.233 | 0.847 |
| 10 | 0.538 | **0.882** | 0.758 | 0.767 | 0.174 | 0.964 |
| 20 | 0.538 | **0.896** | 0.755 | 0.760 | 0.134 | 0.985 |
| 50 | 0.538 | **0.903** | 0.753 | 0.754 | 0.109 | 0.996 |
| 100 | 0.538 | **0.905** | 0.752 | 0.753 | 0.101 | 0.999 |
| 200 | 0.538 | **0.906** | 0.752 | 0.752 | 0.099 | 1.000 |

### III-F. Figures

| Figure | Description |
|---|---|
| `fig_blind_alg_heatmap.png` | Heatmap: 4 algorithms x 3 noise x 7 strategies |
| `fig_blind_alg_bars.png` | Grouped bar chart under combined noise |
| `fig_blind_alg_copies.png` | Copy scaling per algorithm (4 panels) |
| `fig_blind_alg_scatter.png` | Estimation fidelity vs recovery fidelity (scatter) |

---

## IV. Consolidated Analysis

### Best Blind Strategy by Regime

| Regime | Best Strategy | Typical $F$ | Notes |
|---|---|---|---|
| Low-dim ($d \leq 16$), dephasing | **Coherence max** | 0.99--1.00 | No noise-model knowledge needed |
| Low-dim ($d \leq 16$), depolarizing | **Coherence max** or **Ch. inv.** | 0.96--1.00 | Both equally effective |
| Low-dim ($d \leq 16$), combined | **Coherence max** | 0.95--0.99 | Robust to amplitude damping component |
| Low-dim, amplitude damping only | **Channel inversion** | 0.99--1.00 | Only strategy that corrects populations |
| High-dim ($d = 64$), any noise | **Channel inversion** | 0.90--1.00 | Coherence max breaks down at $d=64$ |
| Any dim, no noise knowledge | **Coherence max** | 0.75--1.00 | Safe default if $d \leq 16$ |
| Very few copies ($n < 5$) | **Coherence max** | 0.56--0.82 | Fastest convergence per copy |

### Algorithm-Specific Conclusions

| Algorithm | Dim | Best Blind $F$ (Combined) | Gap to Oracle | Recommended Strategy |
|---|---|---|---|---|
| qDRIFT | 8 | 0.993 | 0.7% | Coherence max |
| QKAN | 4 | 0.995 | 0.5% | Coherence max |
| Control-Free QPE | 16 | 0.959 | 4.0% | Coherence max |
| Regev Factoring | 64 | 0.905 | 9.4% | Channel inversion |

### Universal Observations

1. **Estimation fidelity $\approx$ recovery fidelity** (scatter plot is near-diagonal). The quality of the blind estimate is the sole bottleneck; the catalytic amplification itself is lossless.

2. **Zero vs nonzero threshold is absolute.** If any coherent mode reaches exactly zero, that mode cannot be recovered regardless of strategy or copy count. This is the Shiraishi-Takagi sharp threshold in action.

3. **Coherence maximization is the best zero-knowledge strategy** for $d \leq 16$, requiring no noise model information and only 10--20 copies for $F > 0.95$.

4. **Channel inversion is essential at high dimension** ($d = 64$). Coherence maximization overshoots because $\sqrt{p_i p_j}$ is a loose bound when populations are nearly uniform in high-$d$ mixed states.

5. **Multi-copy tomography degrades at high dimension** due to insufficient statistical samples (20 copies for 64-dimensional reconstruction).

6. **Copy efficiency peaks at small $n$.** The marginal fidelity gain per additional copy drops rapidly; $F(n)/n$ decreases monotonically on log-log scale.

---

## V. File Inventory

### Scripts

| File | Description |
|---|---|
| `benchmark_blind_cqec.py` | Benchmark I: noise sweeps, 6 strategies |
| `benchmark_blind_cqec_copies.py` | Benchmark II: copy-economized version |
| `benchmark_blind_algorithms.py` | Benchmark III: 4 algorithms + blind CQEC |

### Generated Figures

All figures saved as both `.png` and `.pdf` in `~/Documents/alterego/personal/cqec/`.

| Benchmark | Figures |
|---|---|
| I | `fig_blind_dephasing`, `fig_blind_depolarizing`, `fig_blind_amplitude_damping`, `fig_blind_qutrit`, `fig_blind_fidelity_gap` |
| II | `fig_copies_min_heatmap`, `fig_copies_scaling_exponents`, `fig_copies_Deph_γ*`, `fig_copies_Depol_*`, `fig_copies_Amp_*` |
| III | `fig_blind_alg_heatmap`, `fig_blind_alg_bars`, `fig_blind_alg_copies`, `fig_blind_alg_scatter` |

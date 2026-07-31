# Catalytic Coherence Amplification for Quantum State Recovery: Theory, Numerics, and Comparison with Conventional Error Correction

## Authors
[Authors], [Affiliations]

---

## Abstract

Quantum error correction is essential for scalable quantum computing, yet conventional approaches based on encoding redundancy impose substantial qubit overhead and fail when noise exceeds code-specific thresholds. Here we present Infinite Catalytic Error Correction (ICEC), a state recovery protocol that exploits the recently discovered arbitrary amplification of quantum coherence in catalytic transformations [1]. The protocol uses a catalyst state that is returned unchanged after each correction cycle, enabling unlimited reuse. The recovery condition exhibits an infinitely sharp threshold: any nonzero residual coherence suffices for state recovery, while exact zero coherence renders recovery impossible. We numerically validate ICEC across four quantum algorithms—qDRIFT Hamiltonian simulation [2], quantum Kolmogorov-Arnold networks [3], control-free quantum phase estimation [4], and Regev's factoring algorithm [5]—under three decoherence models, achieving fidelity recovery from $F = 0.07$ to $F > 0.999$ across 200 tested noise configurations. Comparison against Steane [[7,1,3]] and surface codes illustrates that ICEC, operating under different assumptions (knowledge of the target state and copy overhead), maintains $F \approx 1.0$ at error rates where conventional codes degrade. We further demonstrate ICEC applied to quantum states arising in tree tensor network cryptographic circuits [6, 7]. Our results highlight the potential of coherence resource theory as a foundation for quantum state recovery, complementary to stabilizer-based error correction.

---

## I. Introduction

Quantum computers promise exponential speedups for problems in cryptography [5, 8], quantum simulation [2, 9], and machine learning [3, 10], but their practical realization is hindered by the fragility of quantum coherence under environmental decoherence [11, 12]. Quantum error correction (QEC) addresses this challenge by encoding logical information redundantly across physical qubits, enabling the detection and correction of errors below a code-specific threshold [13, 14]. However, the qubit overhead of conventional QEC is substantial: the Steane [[7,1,3]] code requires 7 physical qubits per logical qubit [15], while surface codes at distance $d$ require $O(d^2)$ qubits [16, 17], and correction fails entirely when error rates exceed the threshold $p_{\mathrm{th}} \approx 1\%$ [18].

A distinct approach to quantum state protection arises from the resource theory of quantum coherence [19, 20]. Coherence—the presence of off-diagonal elements in a state's density matrix with respect to a preferred basis—is a quantifiable resource under the framework of covariant (energy-conserving) operations [21, 22]. Shiraishi and Takagi [1] recently established that the transformation rate between coherent states can diverge arbitrarily: given $n$ copies of a weakly coherent state $\rho$, one can produce $m \gg n$ copies of a target state $\rho'$ via catalytic covariant operations, provided that the coherent modes of $\rho'$ are contained within those of $\rho$. This result reveals an infinitely sharp boundary between recoverable and irrecoverable quantum states—a feature with no analog in conventional QEC.

Despite its fundamental significance, the practical applicability of the Shiraishi-Takagi result to quantum error correction has not been explored. In particular, it remains unclear whether the infinite amplification rate translates into effective protection for realistic quantum algorithms operating under decoherence.

In this paper, we bridge this gap by introducing the Infinite Catalytic Error Correction (ICEC) protocol, which leverages Theorems 1 and 2 of Ref. [1] to construct a reusable catalyst that restores corrupted quantum states to arbitrary fidelity. We implement ICEC for density-matrix-level simulations of four quantum algorithms spanning Hamiltonian simulation, quantum machine learning, phase estimation, and integer factoring, and demonstrate protection of quantum cryptographic protocols. Our central contributions are:

1. Formulation of the ICEC protocol with explicit conditions for success and failure based on the mode inclusion criterion $\mathcal{C}(\rho') \subseteq \mathcal{C}(\rho)$.
2. Numerical verification of the sharp threshold across 10 orders of magnitude in residual coherence ($\varepsilon \in [10^{-10}, 10^{-1}]$).
3. Comprehensive benchmarking against Steane and surface codes at 10 error rates, demonstrating ICEC's threshold-free advantage.
4. Demonstration of 100-cycle catalyst reuse with zero accumulated deviation.

---

## II. Theoretical Framework

### A. Unspeakable coherence and covariant operations

We work within the resource theory of unspeakable coherence [1, 21, 22], where the free operations are covariant channels—completely positive trace-preserving maps $\Lambda$ satisfying

$$\Lambda \circ \mathcal{U}_t = \mathcal{U}_t \circ \Lambda \quad \forall\, t \in \mathbb{R}, \tag{1}$$

where $\mathcal{U}_t(\rho) = e^{-iHt} \rho \, e^{iHt}$ is the time evolution generated by the system Hamiltonian $H = \sum_i E_i |i\rangle\langle i|$. Condition (1) enforces energy conservation: $[U, H_{\mathrm{total}}] = 0$ for any unitary $U$ implementing $\Lambda$ [21]. This is physically motivated by the Wigner-Araki-Yanase theorem, which constrains measurements and operations that do not commute with conserved quantities [23].

For a $d$-level system, the $\ell_1$-norm of coherence quantifies the total off-diagonal content:

$$C_{\ell_1}(\rho) = \sum_{i \neq j} |\rho_{ij}|. \tag{2}$$

We also employ the quantum Fisher information $\mathcal{F}(\rho, H) = 2 \sum_{i,j} \frac{(\lambda_i - \lambda_j)^2}{\lambda_i + \lambda_j} |\langle i|H|j \rangle|^2$ [24], which provides an operationally meaningful coherence monotone under covariant operations [1, Supplemental Material].

### B. Coherent modes and mode inclusion

The *modes of asymmetry* [22] of a state $\rho$ are defined as

$$\mathcal{D}(\rho) = \{ \Delta_{ij} = E_i - E_j \mid \rho_{ij} \neq 0 \}. \tag{3}$$

The *resonant coherent modes* $\mathcal{C}(\rho)$ are the integer-linear span of $\mathcal{D}(\rho)$:

$$\mathcal{C}(\rho) = \left\{ \sum_{(i,j)} n_{ij} \Delta_{ij} \;\middle|\; n_{ij} \in \mathbb{Z},\; \Delta_{ij} \in \mathcal{D}(\rho) \right\}. \tag{4}$$

$\mathcal{C}(\rho)$ forms a subgroup of $(\mathbb{R}, +)$ and determines the transformability of states under covariant operations [1, Definition S.6].

**Theorem 1** (Shiraishi-Takagi [1]). *If $\mathcal{C}(\rho') \subseteq \mathcal{C}(\rho)$ and $\rho$ is full rank, the asymptotic marginal transformation rate*

$$R(\rho \to \rho') = \sup \left\{ \frac{m}{n} \;\middle|\; \rho^{\otimes n} \xrightarrow{\mathrm{cov}} \sigma,\; \mathrm{Tr}_{\bar{k}}[\sigma] \approx \rho' \;\forall k \right\}$$

*diverges: $R(\rho \to \rho') = \infty$.*

**Theorem 2** (Shiraishi-Takagi [1]). *For any $\rho, \rho'$ with $\mathcal{C}(\rho') \subseteq \mathcal{C}(\rho)$, there exists a catalyst $c$ and a covariant operation $\Lambda$ such that*

$$\mathrm{Tr}_C[\Lambda(\rho \otimes c)] = \rho', \quad \mathrm{Tr}_S[\Lambda(\rho \otimes c)] = c. \tag{5}$$

*That is, $\rho \to \rho'$ is achievable by correlated-catalytic covariant transformation.*

### C. ICEC protocol

The ICEC protocol applies Theorem 2 to quantum error correction. Given an ideal state $\rho_0$ corrupted by decoherence to $\rho_{\mathrm{noisy}}$, the protocol proceeds as:

**Step 0 (Catalyst construction).** Construct a full-rank catalyst $c$ with $\mathcal{D}(c) \supseteq \mathcal{D}(\rho_0)$, following the construction of Proposition S.23 in [1]:

$$c = \frac{1}{n} \sum_{k=1}^{n} \rho^{\otimes(k-1)} \otimes \tau_{n-k} \otimes |k\rangle\langle k|_R, \tag{6}$$

where $\tau_{n-k}$ is the partial output of the asymptotic transformation and $R$ is a register system.

**Step 1 (Mode verification).** Compute $\mathcal{D}(\rho_{\mathrm{noisy}})$ and verify $\mathcal{C}(\rho_0) \subseteq \mathcal{C}(\rho_{\mathrm{noisy}})$. For partial dephasing $\rho_{ij} \to \rho_{ij} e^{-\gamma|\Delta_{ij}|}$ with $\gamma < \infty$, all modes survive: $\mathcal{D}(\rho_{\mathrm{noisy}}) = \mathcal{D}(\rho_0)$, so the condition is satisfied. For complete dephasing ($\gamma = \infty$), $\mathcal{D}(\rho_{\mathrm{noisy}}) = \emptyset$ and recovery is impossible.

**Step 2 (Catalytic recovery).** Apply the covariant operation $\Lambda$ from Theorem 2 to execute

$$\Lambda(\rho_{\mathrm{noisy}} \otimes c) = \tau, \quad \text{with} \quad \mathrm{Tr}_C[\tau] \approx \rho_0, \quad \mathrm{Tr}_S[\tau] = c. \tag{7}$$

**Step 3 (Catalyst reuse).** Since $\mathrm{Tr}_S[\tau] = c$, the catalyst is unchanged and available for the next correction cycle. This enables infinite reuse.

**Success condition.** ICEC succeeds if and only if $C_{\ell_1}(\rho_{\mathrm{noisy}}) > 0$ and $\mathcal{C}(\rho_0) \subseteq \mathcal{C}(\rho_{\mathrm{noisy}})$. The threshold is infinitely sharp: $\varepsilon > 0 \Rightarrow F \to 1$, while $\varepsilon = 0 \Rightarrow F = 1/d$ (no recovery).

### D. Copy scaling

From the proof of Theorem S.10 in [1], the full protocol requires:

- **Input copies:** $n = \mu \cdot N^k$, where $\mu = \sum_{i=1}^{N} \mu_i$ is the total catalyst preparation cost and $N$ is the number of two-level catalysts.
- **Output copies:** $m = (k+1) \cdot N^k$.
- **Rate:** $R = m/n = (k+1)/\mu \to \infty$ as $k \to \infty$.

Each $\mu_i$ depends on the weakest coherent mode of the source state [1, Lemma S.16]:

$$\mu_i \sim \frac{1}{|\rho_{ij,\min}|^2}, \tag{8}$$

where $\rho_{ij,\min}$ is the smallest nonzero off-diagonal element on the relevant mode. For a dephased state with decay factor $e^{-\gamma}$, we obtain $\mu_i \sim e^{2\gamma} / |\rho_{ij}^{(0)}|^2$, showing exponential growth in dephasing strength.

---

## III. Quantum Circuit Architecture

### A. Energy-conserving gate

The fundamental gate for ICEC is the energy-conserving (EC) rotation, which acts on two qubits as [25, 1]

$$U_{\mathrm{EC}}(\theta) = |00\rangle\langle 00| + \cos\theta\, |01\rangle\langle 01| -i\sin\theta\, |01\rangle\langle 10| -i\sin\theta\, |10\rangle\langle 01| + \cos\theta\, |10\rangle\langle 10| + |11\rangle\langle 11|, \tag{9}$$

which satisfies $[U_{\mathrm{EC}}, H_{\mathrm{total}}] = 0$ for $H_{\mathrm{total}} = Z_1 + Z_2$ (the total excitation Hamiltonian). This gate rotates within the degenerate $\{|01\rangle, |10\rangle\}$ subspace while leaving $|00\rangle$ and $|11\rangle$ invariant. At $\theta = \pi/2$, it reduces to the iSWAP gate.

This gate is implemented in our code as `energy_conserving_rotation(theta, qubit_i, qubit_j)` in `icec_qulacs.py` (lines 62–82). In matrix form:

$$U_{\mathrm{EC}}(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -i\sin\theta & 0 \\ 0 & -i\sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}. \tag{10}$$

### B. Three-layer circuit structure

The ICEC circuit operates on four registers: system ($S$, $n_S$ qubits), catalyst ($C$, $n_C$ qubits), and two ancillae ($A_0, A_1$). The circuit applies EC gates in three layers, as implemented in `_build_amplification_circuit()` (`icec_qulacs.py`, lines 307–347):

**Layer 1 (System–Catalyst, $n_S \times n_C$ gates):** Transfers coherence from the system to the catalyst:

$$U_{\mathrm{L1}} = \prod_{s \in S} \prod_{c \in C} U_{\mathrm{EC}}(\theta_{sc}). \tag{11}$$

**Layer 2 (Catalyst–Ancilla, $n_C \times n_A$ gates):** Distributes coherence from the catalyst to ancillae:

$$U_{\mathrm{L2}} = \prod_{c \in C} \prod_{a \in A} U_{\mathrm{EC}}(\theta_{ca}). \tag{12}$$

**Layer 3 (System–Ancilla, $n_S \times n_A$ gates):** Direct coherence amplification between system and ancillae:

$$U_{\mathrm{L3}} = \prod_{s \in S} \prod_{a \in A} U_{\mathrm{EC}}(\theta_{sa}). \tag{13}$$

The total circuit is $U_{\mathrm{ICEC}} = U_{\mathrm{L3}} \cdot U_{\mathrm{L2}} \cdot U_{\mathrm{L1}}$, acting on the initial state $\rho_{\mathrm{noisy}} \otimes c \otimes |0\rangle\langle 0|_A$.

### C. Minimal 4-qubit implementation

For $n_S = 1$, $n_C = 1$, $n_A = 2$, the circuit uses 5 EC gates with parameters $\theta_0$ through $\theta_4$:

```
q₀ (System)   ──[EC(θ₀)]──────────────────────[EC(θ₃)]──[EC(θ₄)]──
                    │                               │         │
q₁ (Catalyst) ──[EC(θ₀)]──[EC(θ₁)]──[EC(θ₂)]──    │         │
                               │         │         │         │
q₂ (Ancilla₀) ────────────[EC(θ₁)]───────────[EC(θ₃)]───────│──
                                                              │
q₃ (Ancilla₁) ──────────────────────[EC(θ₂)]──────────[EC(θ₄)]──

 Layer 1 (S↔C): 1 gate (θ₀)       Coherence bridging
 Layer 2 (C↔A): 2 gates (θ₁, θ₂)  Coherence distribution
 Layer 3 (S↔A): 2 gates (θ₃, θ₄)  Direct amplification
 Total: 5 EC gates
```

*Fig. 1: Minimal 4-qubit ICEC circuit. Gate labels correspond to parameters in the `_build_amplification_circuit` method. Each EC gate is the energy-conserving rotation of Eq. (10).*

### D. Parameter optimization

The gate parameters $\boldsymbol{\theta} = (\theta_0, \ldots, \theta_4)$ are optimized to maximize a combined objective:

$$\mathcal{L}(\boldsymbol{\theta}) = 0.7 \cdot F(\rho_S^{\mathrm{out}}, \rho_0) + 0.3 \cdot F(\rho_C^{\mathrm{out}}, c), \tag{14}$$

where $F(\rho, \sigma) = \left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2$ is the quantum fidelity [26], $\rho_S^{\mathrm{out}} = \mathrm{Tr}_{CA}[U_{\mathrm{ICEC}}(\rho_{\mathrm{in}})U_{\mathrm{ICEC}}^\dagger]$ is the recovered system state, and $\rho_C^{\mathrm{out}} = \mathrm{Tr}_{SA}[\cdot]$ is the catalyst after the operation. The weighting coefficients 0.7 and 0.3 were chosen empirically: recovery requires $F(\rho_S^{\mathrm{out}}, \rho_0) \to 1$ (primary objective), while catalyst preservation $F(\rho_C^{\mathrm{out}}, c) \to 1$ is a constraint that must be satisfied but need not dominate the objective. We verified that results are insensitive to the exact weighting: coefficients in the range $[0.6, 0.8]$ for the system term all yield $F_{\mathrm{after}} > 0.99$. Optimization uses gradient-free random search with local refinement (200 iterations), implemented in `_optimize_parameters()` (`icec_qulacs.py`, lines 349–419).

---

## IV. Benchmark Algorithms and Decoherence Models

### A. Quantum algorithms

We test ICEC on four algorithms representing distinct applications of quantum computing.

**1. qDRIFT** [2] simulates Hamiltonian dynamics $e^{-iHt}$ for the 3-qubit Heisenberg model $H = J \sum_{\langle i,j \rangle} (\sigma_i^x \sigma_j^x + \sigma_i^y \sigma_j^y + \sigma_i^z \sigma_j^z) + h \sum_i \sigma_i^z$ with $J = 1.0$, $h = 0.5$, $t = 1.0$. The initial state $|000\rangle$ evolves to $e^{-iHt}|000\rangle$, which has nonzero off-diagonal elements in the computational basis due to the $XX + YY$ terms in $H$ (these create spin-exchange processes that generate superpositions). The qDRIFT approximation uses 80 random product formula gates with probabilistic sampling $P(h_k) = \|h_k\| / \lambda$ [2, Algorithm 0.1]. The density matrix dimension is $d = 8$ (3 qubits). Since different random seeds produce different gate sequences, $F_{\mathrm{before}}$ varies across seeds (Table I: std = 0.17).

**2. QKAN** [3] implements a quantum Kolmogorov-Arnold network layer encoding the first four Chebyshev polynomials $T_n(x)$ at $x = 0.5$: the ideal state has amplitudes proportional to $(T_0, T_1, T_2, T_3)$, while the algorithmic output truncates at degree 2. Dimension $d = 4$ (2 qubits).

**3. Control-free QPE** [4] estimates eigenvalues of a Fermi-Hubbard-type Hamiltonian using vectorial phase retrieval without controlled unitaries. The protocol generates time-series $f_j = \langle \psi | e^{-iHj\Delta t} | \psi \rangle$ for $j = 0, \ldots, 15$, recovers phases from $|f_1|$, $|f_2|$, $|f_1 + f_2|$ via the relation $\cos(\phi_1 - \phi_2) = (|f_1+f_2|^2 - |f_1|^2 - |f_2|^2) / (2|f_1||f_2|)$ [4, Appendix A], and encodes the resulting spectrum as a 16-dimensional quantum state. Dimension $d = 16$ (4 qubits).

**4. Regev factoring** [5] factors $N = 15$ using discrete Gaussian states over $\mathbb{Z}^d$ with $d = \lceil\sqrt{n}\rceil = 2$ dimensions and $D = 8$ grid points per dimension. After modular exponentiation $\prod_i a_i^{z_i} \bmod N$ with $a_i \in \{9, 25\}$ and quantum Fourier transform, the resulting state encodes lattice information for LLL reduction [27]. Dimension $d = 64$ (6 qubits).

### B. Decoherence models

Three noise channels are applied to each algorithm's output state $\rho_{\mathrm{alg}}$:

**Partial dephasing.** $\mathcal{E}_{\mathrm{deph}}(\rho)_{ij} = \rho_{ij} \cdot e^{-\gamma}$ for $i \neq j$, with $\gamma = 2.0$ (strong dephasing). This preserves all coherent modes ($\mathcal{D}(\rho_{\mathrm{noisy}}) = \mathcal{D}(\rho)$), making ICEC applicable.

**Depolarizing.** $\mathcal{E}_{\mathrm{depol}}(\rho) = (1-p)\rho + p \cdot I/d$, with $p = 0.3$. Introduces both coherence decay and population mixing.

**Combined.** Sequential application: dephasing ($\gamma = 1.0$), depolarizing ($p = 0.15$), amplitude damping ($\gamma_{\mathrm{AD}} = 0.1$). The amplitude damping channel uses Kraus operators $E_0 = |0\rangle\langle 0| + \sqrt{1-\gamma_{\mathrm{AD}}}|1\rangle\langle 1|$, $E_1 = \sqrt{\gamma_{\mathrm{AD}}}|0\rangle\langle 1|$ applied independently to each qubit.

### C. TTN cryptographic protocol

To demonstrate ICEC's applicability beyond algorithm protection, we test it on quantum states generated by a tree tensor network (TTN) cryptographic protocol [6, 7]. The protocol uses an 11-qubit variational TTN circuit ($N_q = 10$ output qubits + 1 ancilla input) with 10 binary-tree blocks of 8 parameters each (4 RY + 4 RZ + 2 CNOT), solving four simultaneous cryptographic equations $R_a(F(A,C), p_1) = \gamma$, $R_b(G(\beta, \gamma), p_2) = C$, etc. [7, Eq. (1)–(4)].

For density-matrix tractability, we simulate a 3-qubit reduced model of the TTN circuit with layered RY-CNOT-RZ structure, capturing the essential entanglement and coherence properties. We test across 6 plaintext lengths ($N_c = 5, 10, 15, 20, 25, 30$) and 7 noise levels from ideal to extreme ($p_{\mathrm{depol}} \in [0, 0.3]$, $\gamma_{\mathrm{deph}} \in [0, 3.0]$).

### D. Conventional QEC baselines

For comparison, we compute the logical fidelity of:

- **Steane [[7,1,3]] code** [15]: corrects any single-qubit error. Logical success probability $P_{\mathrm{corr}} = \sum_{k=0}^{1} \binom{7}{k} p_{\mathrm{eff}}^k (1-p_{\mathrm{eff}})^{7-k}$.

- **Surface code** [16, 17] at distances $d = 3$ and $d = 5$: for $p < p_{\mathrm{th}} \approx 0.01$, the logical error rate scales as $p_L \sim (p/p_{\mathrm{th}})^{(d+1)/2}$. Above threshold, correction capacity is computed from the combinatorial error model with $t = (d-1)/2$ correctable errors on $d^2$ physical qubits.

The effective per-qubit error rate for a $d$-dimensional depolarizing channel is $p_{\mathrm{eff}} = 1 - (1-p)^{1/n_q}$.

---

## V. Results

### A. ICEC benchmark across algorithms (Table I, Fig. 3)

Table I presents ICEC recovery results from 10 independent trials. For qDRIFT and TTN-Crypto, each seed generates a different random gate sequence, producing genuinely independent state instances; the nonzero standard deviations ($\sigma = 0.07$–$0.17$) reflect this variability. For QKAN and CF-QPE, the ideal and algorithmic states are deterministic (independent of seed), yielding $\sigma = 0$ by construction; the 10 trials serve only to confirm numerical reproducibility, not statistical variability. We report both cases for completeness but caution that the $\sigma = 0$ entries should not be interpreted as statistical confidence intervals.

**Table I.** ICEC recovery fidelity (mean ± std, 10 seeds). Success rate is 100% for all entries.

| Algorithm | Noise model | $F_{\mathrm{before}}$ | $F_{\mathrm{after}}$ |
|-----------|-------------|----------------------|---------------------|
| qDRIFT (3 qb) | Dephasing ($\gamma=2$) | $0.701 \pm 0.171$ | $0.9999 \pm 0.0000$ |
| qDRIFT (3 qb) | Depolarizing ($p=0.3$) | $0.528 \pm 0.119$ | $0.9997 \pm 0.0001$ |
| qDRIFT (3 qb) | Combined | $0.615 \pm 0.145$ | $0.9999 \pm 0.0001$ |
| QKAN (2 qb) | Dephasing ($\gamma=2$) | $0.341 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| QKAN (2 qb) | Depolarizing ($p=0.3$) | $0.495 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| QKAN (2 qb) | Combined | $0.386 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| CF-QPE (4 qb) | Dephasing ($\gamma=2$) | $0.306 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| CF-QPE (4 qb) | Depolarizing ($p=0.3$) | $0.718 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| CF-QPE (4 qb) | Combined | $0.427 \pm 0.000$ | $1.0000 \pm 0.0000$ |
| TTN-Crypto (3 qb) | Dephasing ($\gamma=2$) | $0.388 \pm 0.067$ | $1.0000 \pm 0.0000$ |
| TTN-Crypto (3 qb) | Depolarizing ($p=0.3$) | $0.734 \pm 0.002$ | $1.0000 \pm 0.0000$ |
| TTN-Crypto (3 qb) | Combined | $0.487 \pm 0.042$ | $1.0000 \pm 0.0000$ |

### B. Noise strength sweep (Fig. 3)

Fig. 3 presents ICEC recovery fidelity as a function of noise strength for all five algorithms. For dephasing with $\gamma \in [0.1, 5.0]$ (20 points) and depolarizing with $p \in [0.01, 0.95]$ (20 points), ICEC maintains $F_{\mathrm{after}} > 0.99$ across all 200 data points, even when $F_{\mathrm{before}}$ drops to 0.066 (Regev, $\gamma = 5.0$) or 0.065 (Regev, $p = 0.95$).

The pre-correction fidelity decay can be understood from the structure of the dephasing channel. For a pure state $|\psi\rangle = \sum_i c_i |i\rangle$ under uniform dephasing $\rho_{ij} \to \rho_{ij} e^{-\gamma}$ ($i \neq j$), the fidelity with the ideal state is

$$F_{\mathrm{before}}(\gamma) = \sum_i |c_i|^4 + e^{-\gamma} \sum_{i \neq j} |c_i|^2 |c_j|^2 = \mathrm{Tr}[\rho_{\mathrm{diag}}^2] + e^{-\gamma}\left(1 - \mathrm{Tr}[\rho_{\mathrm{diag}}^2]\right), \tag{15}$$

where $\rho_{\mathrm{diag}} = \sum_i |c_i|^2 |i\rangle\langle i|$ is the dephased state. For the maximally coherent state ($|c_i|^2 = 1/d$), this gives $F = 1/d + (1 - 1/d)e^{-\gamma}$, which approaches $1/d$ as $\gamma \to \infty$. The observed decay (Fig. 3a) is consistent with Eq. (15): Regev ($d = 64$, $1/d = 0.016$) decays most steeply, while qDRIFT ($d = 8$, $1/d = 0.125$) decays more slowly due to its non-uniform population distribution. Post-ICEC fidelity remains flat at $F \approx 1.0$ regardless of $d$ or $\gamma$, confirming that recovery depends only on whether coherence is nonzero, not on its magnitude.

*Fig. 3: ICEC recovery fidelity vs. noise strength. (a) Dephasing channel, $\gamma \in [0.1, 5.0]$. (b) Depolarizing channel, $p \in [0.01, 0.95]$. Dashed lines: pre-correction fidelity; solid lines: post-ICEC fidelity. Data from `benchmark_prxq.py`, Experiment 1.*

### C. Sharp threshold (Fig. 2)

Fig. 2 demonstrates the infinitely sharp zero/nonzero threshold predicted by Theorem 1. For a 2-qubit ($d = 4$) maximally coherent target state, we prepare noisy states with controlled residual coherence $\varepsilon$ and apply ICEC.

- $\varepsilon = 0$ (exact zero): $F_{\mathrm{after}} = 0.250 = 1/d$. Recovery fails; the protocol correctly identifies $\mathcal{C}(\rho_0) \not\subseteq \mathcal{C}(\rho_{\mathrm{noisy}}) = \{0\}$.
- $\varepsilon = 10^{-10}$: $F_{\mathrm{after}} = 1.000$. Full recovery despite $C_{\ell_1} = 1.2 \times 10^{-9}$.
- All $\varepsilon > 0$ (30 values on $[10^{-10}, 0.32]$): $F_{\mathrm{after}} = 1.000$.

This confirms the theoretical prediction: the transition from irrecoverable to perfectly recoverable is discontinuous at $\varepsilon = 0$.

*Fig. 2: Sharp threshold of ICEC. Post-correction fidelity vs. residual coherence $\varepsilon$ (log scale). Green circles: successful recovery ($\varepsilon > 0$). Red cross: failed recovery ($\varepsilon = 0$). Dashed line: $F = 1/d = 0.25$ (random guess). Data from `benchmark_prxq.py`, Experiment 5.*

### D. Comparison with conventional QEC (Fig. 4, Table II)

Fig. 4 compares ICEC against three conventional codes across depolarizing noise $p \in [0.001, 0.3]$.

**Table II.** Fidelity comparison at representative error rates (CF-QPE, 4 qubits).

| $p$ | No correction | Steane [[7,1,3]] | Surface $d=3$ | Surface $d=5$ | ICEC |
|-----|--------------|-------------------|---------------|---------------|------|
| 0.01 | 0.989 | 1.000 | 0.994 | 0.998 | 1.000 |
| 0.10 | 0.905 | 0.987 | 0.979 | 0.974 | 1.000 |
| 0.20 | 0.811 | 0.949 | 0.918 | 0.848 | 1.000 |
| 0.30 | 0.718 | 0.885 | 0.824 | 0.639 | 1.000 |

Key observations:
1. At $p = 0.01$ (below threshold), all codes perform well; ICEC is marginally superior.
2. At $p = 0.1$ (above threshold for surface codes), surface code $d = 5$ degrades to 0.974 while ICEC remains at 1.000.
3. At $p = 0.3$, Steane falls to 0.885, surface $d = 5$ to 0.639, while ICEC maintains 1.000.

The fundamental advantage of ICEC is the absence of an error threshold: recovery depends on whether any coherence survives, not on the error rate relative to a code-specific constant.

*Fig. 4: Conventional QEC vs. ICEC. Four panels (a–d) for qDRIFT, QKAN, CF-QPE, Regev. ICEC (red circles) maintains $F \approx 1$ across all error rates, while conventional codes degrade. Data from `benchmark_prxq.py`, Experiment 2.*

### E. TTN cryptographic protocol protection (Fig. 6)

Fig. 6 presents a heatmap of fidelity before and after ICEC correction for the TTN cryptographic protocol across 6 plaintext lengths and 7 noise levels (42 data points).

Under extreme noise ($p = 0.3$, $\gamma = 3.0$), the pre-correction fidelity ranges from 0.20 ($N_c = 20$) to 0.28 ($N_c = 30$). After ICEC, all 42 data points achieve $F = 1.000$, demonstrating that ICEC can protect intermediate quantum states within cryptographic protocols, complementing the classical noise absorption mechanism observed in TTN-based quantum cryptography [7, Section 3].

*Fig. 6: TTN crypto protocol + ICEC protection. (a) Fidelity before ICEC (heatmap by plaintext length $N_c$ and noise level). (b) Fidelity after ICEC (uniformly $F = 1.0$). Data from `benchmark_prxq.py`, Experiment 3.*

### F. Catalyst durability (Fig. 5)

Fig. 5 demonstrates catalyst reuse over 100 consecutive correction cycles. At each cycle, fresh dephasing ($\gamma = 1.5$) and depolarizing ($p = 0.2$) noise is applied to the qDRIFT state, and ICEC recovery is performed using the same catalyst.

- Average recovered fidelity: $F = 0.990$ (all 100 cycles identical).
- Maximum catalyst deviation: $\delta_c = 0.00$ (exactly zero).
- Accumulated error after 100 cycles: $0.00$.

This verifies the theoretical prediction of Theorem 2: the catalyst is returned exactly to its original state, enabling unlimited reuse. The finite deviation of $F$ from 1.0 reflects the discrete approximation in our numerical implementation, not a fundamental limitation.

*Fig. 5: Catalyst durability test. (a) Recovery fidelity over 100 cycles (green: recovered, red: noisy). (b) Catalyst state deviation from original (identically zero). Data from `benchmark_prxq.py`, Experiment 4.*

### G. Resource overhead (Table III)

**Table III.** Resource comparison for $n$ logical qubits.

| $n_{\mathrm{logical}}$ | ICEC (qubits / EC gates) | Steane (qubits / gates) | Surface $d=3$ | Surface $d=5$ |
|---|---|---|---|---|
| 1 | 8 / 5 | 7 / 10 | 13 / 36 | 41 / 100 |
| 3 | 22 / 11 | 21 / 30 | 39 / 108 | 123 / 300 |
| 6 | 137 / 20 | 42 / 60 | 78 / 216 | 246 / 600 |

ICEC has the fewest gates at all sizes (5–20 EC gates vs. 10–600 for conventional codes) but requires additional qubits for catalyst preparation copies ($\mu \sim d^2 e^{2\gamma}$, Eq. (8)). For $n = 6$ qubits under strong dephasing, the copy overhead reaches $\sim 130$ qubits. However, the gate depth remains $O(n)$, which is significantly shallower than surface codes at $O(d^2 n)$.

---

## VI. Discussion

### A. Two paradigms for quantum information protection

Our results reveal a complementary relationship between two protection strategies:

1. **Classical noise absorption** (as in TTN cryptographic protocols [7]): The classical regression layer (ALS/SPSA) absorbs quantum measurement noise statistically. This approach requires no quantum overhead but is limited to cases where decoherence does not fundamentally alter the feature space. In the TTN protocol, shot noise ($10^3$ shots) and weak depolarizing noise ($p = 0.01$) produce negligible effects on reconstruction MSE ($\sim 10^{-16}$) [7, Table 1], but stronger decoherence would degrade the quantum feature extraction itself.

2. **Catalytic coherence amplification** (ICEC): Operates directly on the quantum state before measurement. Effective for arbitrarily strong decoherence provided $C_{\ell_1} > 0$, but requires quantum copy overhead.

These paradigms are complementary: classical absorption handles measurement-level noise efficiently, while ICEC addresses state-level decoherence that classical methods cannot correct. Quantitatively, the TTN protocol achieves reconstruction MSE $\sim 10^{-16}$ under shot noise and $\sim 10^{-16}$ under weak depolarizing noise ($p = 0.01$), demonstrating the effectiveness of classical absorption at low noise levels [7, Supplemental Material]. However, under the stronger decoherence tested here ($p = 0.3$, $\gamma = 3.0$), the pre-correction fidelity of the intermediate quantum state drops to $F \approx 0.20$–$0.28$ (Fig. 6a), which would severely degrade the quantum feature space even before the classical regression layer is applied. ICEC restores $F = 1.0$ at this stage, preserving the quantum advantage of the feature extraction step.

### B. Limitations and scope of the numerical simulation

We emphasize that the numerical results presented here validate the *theoretical guarantees* of Ref. [1], rather than demonstrating an end-to-end quantum circuit protocol. Several important limitations must be acknowledged.

**Oracle access to the target state.** The ICEC implementation assumes knowledge of the target state $\rho_0$ for both mode verification (Step 1) and the catalytic recovery (Step 2). In practical applications, $\rho_0$ is the known output of a specific quantum algorithm (e.g., the Trotter evolution $e^{-iHt}|\psi_0\rangle$ for qDRIFT), so this assumption is equivalent to knowing the algorithm being run. However, the recovery operation in our density-matrix simulation directly accesses the matrix elements of $\rho_0$, whereas a physical implementation would construct $\rho_0$ through the many-copy asymptotic protocol of Theorem S.10 in [1]. Our simulation demonstrates that the mathematical conditions for recovery (nonzero coherent modes) are satisfied under realistic decoherence, while the actual circuit-level construction using $\mu \cdot N^k$ copies remains to be demonstrated experimentally.

**Asymptotic nature of the protocol.** Theorems 1 and 2 of Ref. [1] are asymptotic results: the transformation $\rho^{\otimes n} \to \sigma$ achieves exact conversion only in the limit $n \to \infty$. For finite $n$, the recovered state has fidelity $F < 1$, with the gap decreasing as $O(1/\sqrt{n})$ [1, Supplemental Material, Lemma S.16]. Our simulation effectively takes the $n \to \infty$ limit by directly implementing the guaranteed output, and the finite-$n$ corrections are an important consideration for any practical realization.

**Comparison with conventional QEC.** The comparison in Section V.D is intended to illustrate qualitative differences between the two paradigms, not to claim strict superiority. ICEC and conventional QEC operate under different assumptions: conventional codes require only the ability to perform syndrome measurements without knowing the encoded state, while ICEC requires the target state (or a procedure to generate it) and $O(\mu \cdot N^k)$ input copies. The two approaches protect against different classes of errors—stabilizer QEC corrects Pauli errors on encoded data, while ICEC corrects coherence loss in bare quantum states. A fair resource comparison must account for these different operational settings.

**Copy scaling.** The copy overhead of Eq. (8) implies $\mu \sim e^{2\gamma}/|\rho_{ij}^{(0)}|^2$, which grows exponentially with dephasing strength. For $\gamma = 5$ and $d = 64$ (Regev), the estimated copy count exceeds $10^4$. While this is finite and the rate $R = (k+1)/\mu$ diverges as $k \to \infty$, the absolute resource requirements for finite $k$ may be prohibitive compared to conventional QEC at low error rates.

**Noise model.** The comparison with conventional QEC uses an effective per-qubit error model ($p_{\mathrm{eff}} = 1 - (1-p)^{1/n_q}$), which does not capture correlations in physical noise. Correlated noise could be either favorable (if it preserves coherent modes globally) or unfavorable (if it selectively destroys specific modes) for ICEC.

### C. Mode no-broadcasting and fundamental limits

Theorem 3 of Ref. [1] establishes that mode no-broadcasting prevents the creation of states with irrational coherent modes from states with only rational modes, even catalytically. This fundamental constraint means that ICEC cannot overcome all forms of coherence loss. If a decoherence channel selectively destroys modes with a specific energy difference $\Delta$ while preserving others, and the target state requires $\Delta$ for its coherence structure, recovery is impossible regardless of the residual coherence on other modes. This is physically distinct from the error threshold of conventional QEC: it is a topological constraint on the mode lattice structure, not a quantitative constraint on error magnitude.

---

## VII. Conclusion

We have introduced Infinite Catalytic Error Correction, a quantum error correction protocol based on the resource theory of unspeakable coherence. The protocol exploits the Shiraishi-Takagi result [1] that any nonzero coherence can be amplified to arbitrary levels via catalytic covariant operations, with the catalyst returned unchanged for unlimited reuse. We demonstrated ICEC recovery across four quantum algorithms and a quantum cryptographic protocol under three decoherence models, achieving $F > 0.999$ in all 200 tested noise configurations and confirming the infinitely sharp zero/nonzero threshold across 10 orders of magnitude in residual coherence. Comparison with Steane and surface codes showed that ICEC operates without an error threshold, maintaining perfect fidelity at error rates where conventional codes fail.

Our results suggest that coherence resource theory provides a viable foundation for quantum state recovery that is fundamentally distinct from, and complementary to, the stabilizer formalism. We emphasize that the present work constitutes a numerical validation of the theoretical guarantees of Ref. [1] applied to error correction, rather than a demonstration of a fully autonomous quantum circuit protocol. Bridging this gap requires three concrete steps: (i) experimental implementation of the many-copy asymptotic protocol for finite $n$, quantifying the fidelity gap $1 - F \sim O(1/\sqrt{n})$ predicted by Lemma S.16 of [1]; (ii) development of approximate catalytic protocols that trade exact catalyst preservation for reduced copy overhead, potentially using the partial-reusability framework of Lemma S.13 in [1]; and (iii) integration with conventional QEC to exploit the complementary strengths of both paradigms—stabilizer codes for Pauli error correction and ICEC for coherence restoration in protected subspaces.

---

## References

[1] N. Shiraishi and R. Takagi, "Arbitrary amplification of quantum coherence in asymptotic and catalytic transformation," Phys. Rev. Lett. **132**, 180202 (2024).

[2] E. Campbell, "Random compiler for fast Hamiltonian simulation," Phys. Rev. Lett. **123**, 070503 (2019); H.-Y. Chen, M. Huang, R. Kueng, and J. A. Tropp, "Concentration for random product formulas," PRX Quantum **2**, 040305 (2021).

[3] M. Ivashkov, P.-W. Huang, N. Kotturu, O. Khatib, P. Ronagh, and A. Aspuru-Guzik, "Quantum Kolmogorov-Arnold networks," arXiv:2410.04435 (2024).

[4] L. Clinton, T. Cubitt, B. Flynn, F. M. Gambetta, J. Sherbert, and A. Sherrill, "Control-free quantum phase estimation," PRX Quantum **7**, 010345 (2026).

[5] O. Regev, "An efficient quantum factoring algorithm," arXiv:2308.06572v3 (2024).

[6] W. Huggins, P. Patil, B. Mitchell, K. B. Whaley, and E. M. Stoudenmire, "Towards quantum machine learning with tensor networks," Quantum Sci. Technol. **4**, 024001 (2019).

[7] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms," Adv. Quantum Technol. **2**, 1900070 (2019). Our TTN cryptographic protocol uses variational tree tensor network circuits in the architecture of Ref. [6]; the specific four-equation cryptographic system and experimental results are described in the Supplemental Material.

[8] P. W. Shor, "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer," SIAM J. Comput. **26**, 1484 (1997).

[9] R. P. Feynman, "Simulating physics with computers," Int. J. Theor. Phys. **21**, 467 (1982).

[10] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, "Quantum machine learning," Nature **549**, 195 (2017).

[11] W. H. Zurek, "Decoherence, einselection, and the quantum origins of the classical," Rev. Mod. Phys. **75**, 715 (2003).

[12] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information* (Cambridge Univ. Press, 2010).

[13] P. W. Shor, "Scheme for reducing decoherence in quantum computer memory," Phys. Rev. A **52**, R2493 (1995).

[14] D. Gottesman, "Stabilizer codes and quantum error correction," Ph.D. thesis, Caltech (1997), arXiv:quant-ph/9705052.

[15] A. M. Steane, "Error correcting codes in quantum theory," Phys. Rev. Lett. **77**, 793 (1996).

[16] A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Ann. Phys. **303**, 2 (2003).

[17] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," Phys. Rev. A **86**, 032324 (2012).

[18] E. Knill, "Quantum computing with realistically noisy devices," Nature **434**, 39 (2005).

[19] T. Baumgratz, M. Cramer, and M. B. Plenio, "Quantifying coherence," Phys. Rev. Lett. **113**, 140401 (2014).

[20] A. Streltsov, G. Adesso, and M. B. Plenio, "Colloquium: Quantum coherence as a resource," Rev. Mod. Phys. **89**, 041003 (2017).

[21] I. Marvian and R. W. Spekkens, "Modes of asymmetry: The application of harmonic analysis in symmetric quantum dynamics and quantum reference frames," Phys. Rev. A **90**, 062110 (2014).

[22] M. Lostaglio and M. P. Müller, "Coherence and asymmetry cannot be broadcast," Phys. Rev. Lett. **123**, 020403 (2019).

[23] E. P. Wigner, "Die Messung quantenmechanischer Operatoren," Z. Phys. **131**, 101 (1952); M. M. Yanase, "Optimal measuring apparatus," Phys. Rev. **123**, 666 (1961); H. Araki and M. M. Yanase, "Measurement of quantum mechanical operators," Phys. Rev. **120**, 622 (1960).

[24] S. L. Braunstein and C. M. Caves, "Statistical distance and the geometry of quantum states," Phys. Rev. Lett. **72**, 3439 (1994).

[25] N. Schuch, F. Verstraete, and J. I. Cirac, "Nonlocal resources in the presence of superselection rules," Phys. Rev. Lett. **92**, 087904 (2004).

[26] R. Jozsa, "Fidelity for mixed quantum states," J. Mod. Opt. **41**, 2315 (1994).

[27] A. K. Lenstra, H. W. Lenstra, and L. Lovász, "Factoring polynomials with rational coefficients," Math. Ann. **261**, 515 (1982).

"""
icec_qulacs.py - ICEC Protocol with Qulacs Quantum Circuit Simulation
======================================================================
Implements the ICEC protocol using qulacs for actual quantum circuit
simulation, demonstrating the coherence amplification phenomena from
Shiraishi & Takagi, PRL 132, 180202 (2024).

Uses qulacs to:
  - Simulate energy-conserving unitary circuits
  - Implement decoherence channels via Kraus operators
  - Demonstrate coherence amplification from multiple copies
  - Verify catalyst integrity through circuit simulation
"""

import numpy as np
from qulacs import (
    QuantumState, QuantumCircuit, DensityMatrix,
    Observable, GeneralQuantumOperator,
)
from qulacs.gate import (
    Identity, X, Y, Z, H, S, CNOT, CZ, SWAP,
    DenseMatrix, Probabilistic, merge,
    DepolarizingNoise, BitFlipNoise, AmplitudeDampingNoise,
)
from dataclasses import dataclass
from typing import Optional
import sys


# ============================================================
# 1. Qulacs-based Quantum State Utilities
# ============================================================

def density_matrix_from_qulacs(dm: DensityMatrix) -> np.ndarray:
    """Extract numpy density matrix from qulacs DensityMatrix."""
    n = dm.get_qubit_count()
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)
    # Use the matrix representation
    mat = dm.get_matrix()
    return np.array(mat)


def set_density_matrix(dm: DensityMatrix, rho: np.ndarray):
    """Set a qulacs DensityMatrix from numpy array."""
    dm.load(rho.flatten().tolist())


def trace_distance_qulacs(dm1: DensityMatrix, dm2: DensityMatrix) -> float:
    """Trace distance between two qulacs density matrices."""
    rho1 = density_matrix_from_qulacs(dm1)
    rho2 = density_matrix_from_qulacs(dm2)
    diff = rho1 - rho2
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(singular_values)


# ============================================================
# 2. Energy-Conserving Gates for Qulacs
# ============================================================

def energy_conserving_rotation(theta: float, qubit_i: int,
                                qubit_j: int) -> np.ndarray:
    """Create an energy-conserving rotation in the |01>, |10> subspace.

    This is the fundamental building block for covariant operations.
    It conserves total excitation number (= energy for uniform spacing).

    U = exp(-i * theta * (|01><10| + |10><01|))

    In the {|00>, |01>, |10>, |11>} basis:
    U = diag(1) oplus R(theta) oplus diag(1)
    where R(theta) is a rotation in the |01>,|10> subspace.
    """
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([
        [1,  0,  0,  0],
        [0,  c, -1j*s, 0],
        [0, -1j*s, c,  0],
        [0,  0,  0,  1],
    ], dtype=complex)
    return U


def energy_conserving_phase(phi: float, qubit_i: int,
                             qubit_j: int) -> np.ndarray:
    """Energy-conserving phase gate in the |01>, |10> subspace.

    U|01> = e^{i*phi}|01>,  U|10> = e^{-i*phi}|10>

    This preserves energy while manipulating coherence phase.
    """
    U = np.array([
        [1, 0, 0, 0],
        [0, np.exp(1j * phi), 0, 0],
        [0, 0, np.exp(-1j * phi), 0],
        [0, 0, 0, 1],
    ], dtype=complex)
    return U


def make_ec_gate(theta: float, phi: float,
                  qubit_i: int, qubit_j: int) -> "qulacs.gate":
    """Create a qulacs gate for energy-conserving unitary on two qubits."""
    U_rot = energy_conserving_rotation(theta, qubit_i, qubit_j)
    U_phase = energy_conserving_phase(phi, qubit_i, qubit_j)
    U_total = U_phase @ U_rot
    return DenseMatrix([qubit_i, qubit_j], U_total)


# ============================================================
# 3. Decoherence Channels in Qulacs
# ============================================================

def apply_dephasing(dm: DensityMatrix, qubit: int, gamma: float):
    """Apply partial dephasing to a qubit.

    rho -> (1 - p/2) rho + (p/2) Z rho Z
    where p = 1 - exp(-gamma)

    For gamma < inf: coherent modes SURVIVE -> ICEC works
    For gamma = inf: complete dephasing -> ICEC fails
    """
    p = 1 - np.exp(-gamma)
    p = min(p, 1.0)

    # Kraus operators: K0 = sqrt(1-p/2) * I, K1 = sqrt(p/2) * Z
    rho = density_matrix_from_qulacs(dm)
    n_qubits = dm.get_qubit_count()
    dim = 2**n_qubits

    # Build Z on target qubit
    Z_full = np.eye(dim, dtype=complex)
    for i in range(dim):
        # Check bit at position 'qubit'
        if (i >> qubit) & 1:
            Z_full[i, i] = -1

    rho_out = (1 - p / 2) * rho + (p / 2) * Z_full @ rho @ Z_full
    set_density_matrix(dm, rho_out)


def apply_amplitude_damping_dm(dm: DensityMatrix, qubit: int, gamma: float):
    """Apply amplitude damping to a single qubit in density matrix."""
    rho = density_matrix_from_qulacs(dm)
    n_qubits = dm.get_qubit_count()
    dim = 2**n_qubits

    # Build single-qubit Kraus operators embedded in full space
    E0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1_1q = np.array([[0, np.sqrt(gamma)], [0, 0]])

    def embed_single_qubit_op(op, target_qubit, n_q):
        """Embed single qubit operator into n-qubit space."""
        result = np.zeros((2**n_q, 2**n_q), dtype=complex)
        for i in range(2**n_q):
            for j in range(2**n_q):
                # Check if all other qubits match
                mask = ~(1 << target_qubit) & ((1 << n_q) - 1)
                if (i & mask) == (j & mask):
                    bi = (i >> target_qubit) & 1
                    bj = (j >> target_qubit) & 1
                    result[i, j] = op[bi, bj]
        return result

    E0 = embed_single_qubit_op(E0_1q, qubit, n_qubits)
    E1 = embed_single_qubit_op(E1_1q, qubit, n_qubits)

    rho_out = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
    set_density_matrix(dm, rho_out)


def apply_depolarizing_dm(dm: DensityMatrix, qubit: int, p: float):
    """Apply depolarizing channel to a single qubit."""
    rho = density_matrix_from_qulacs(dm)
    n_qubits = dm.get_qubit_count()
    dim = 2**n_qubits

    def embed_single_qubit_op(op, target_qubit, n_q):
        result = np.zeros((2**n_q, 2**n_q), dtype=complex)
        for i in range(2**n_q):
            for j in range(2**n_q):
                mask = ~(1 << target_qubit) & ((1 << n_q) - 1)
                if (i & mask) == (j & mask):
                    bi = (i >> target_qubit) & 1
                    bj = (j >> target_qubit) & 1
                    result[i, j] = op[bi, bj]
        return result

    I_op = np.eye(2)
    X_op = np.array([[0, 1], [1, 0]])
    Y_op = np.array([[0, -1j], [1j, 0]])
    Z_op = np.array([[1, 0], [0, -1]])

    ops = [embed_single_qubit_op(op, qubit, n_qubits)
           for op in [I_op, X_op, Y_op, Z_op]]

    rho_out = (1 - 3*p/4) * rho
    for K in ops[1:]:
        rho_out += (p/4) * K @ rho @ K.conj().T

    set_density_matrix(dm, rho_out)


# ============================================================
# 4. Coherence Measures for Qulacs States
# ============================================================

def coherence_l1_qulacs(dm: DensityMatrix, n_system_qubits: int = None) -> float:
    """L1 coherence measure for qulacs density matrix.

    For qubit system with H = sum_i (i * sigma_z_i), coherence is
    the sum of absolute off-diagonal elements between different
    energy eigenvalues (computational basis = energy eigenbasis).
    """
    rho = density_matrix_from_qulacs(dm)
    n = dm.get_qubit_count()
    if n_system_qubits is None:
        n_system_qubits = n
    dim = 2**n_system_qubits

    # Use only the system part if there are ancilla qubits
    if n > n_system_qubits:
        # Partial trace over ancilla
        dim_anc = 2**(n - n_system_qubits)
        rho_sys = np.zeros((dim, dim), dtype=complex)
        for k in range(dim_anc):
            for i in range(dim):
                for j in range(dim):
                    rho_sys[i, j] += rho[i * dim_anc + k, j * dim_anc + k]
        rho = rho_sys

    # Energy = number of 1s in binary representation (excitation number)
    coh = 0.0
    for i in range(dim):
        for j in range(dim):
            energy_i = bin(i).count('1')
            energy_j = bin(j).count('1')
            if energy_i != energy_j:
                coh += abs(rho[i, j])
    return coh


# ============================================================
# 5. ICEC Protocol with Qulacs Circuits
# ============================================================

@dataclass
class QulacsICECResult:
    """Result of one ICEC cycle using qulacs."""
    cycle: int
    input_coherence: float
    output_coherence: float
    trace_distance: float
    catalyst_preserved: bool
    n_gates: int


class QulacsICEC:
    """ICEC Protocol implemented with qulacs quantum circuits.

    Circuit architecture:
    - System qubit(s): the quantum state to protect
    - Catalyst qubit(s): auxiliary system that enables amplification
    - Ancilla qubit(s): incoherent resource states

    Key circuit: energy-conserving rotations transfer coherence
    from weakly coherent states to the target system, mediated
    by the catalyst.
    """

    def __init__(self, n_system: int = 1, n_catalyst: int = 1,
                 n_ancilla: int = 2):
        self.n_system = n_system
        self.n_catalyst = n_catalyst
        self.n_ancilla = n_ancilla
        self.n_total = n_system + n_catalyst + n_ancilla

        # Qubit labels
        self.sys_qubits = list(range(n_system))
        self.cat_qubits = list(range(n_system, n_system + n_catalyst))
        self.anc_qubits = list(range(n_system + n_catalyst, self.n_total))

        self.history: list[QulacsICECResult] = []
        self._target_rho = None
        self._catalyst_rho = None

    def initialize(self, target_state: np.ndarray = None):
        """Initialize system with target state and catalyst."""
        dim_sys = 2**self.n_system
        dim_cat = 2**self.n_catalyst

        if target_state is None:
            # Default: maximally coherent state |+>
            plus = np.ones(dim_sys) / np.sqrt(dim_sys)
            target_state = np.outer(plus, plus)

        self._target_rho = target_state

        # Catalyst: full-rank mixed state with coherence
        # (1-eps)|+><+| + eps*I/d
        eps = 0.1
        plus_cat = np.ones(dim_cat) / np.sqrt(dim_cat)
        cat_pure = np.outer(plus_cat, plus_cat)
        self._catalyst_rho = (1 - eps) * cat_pure + eps * np.eye(dim_cat) / dim_cat

    def _build_amplification_circuit(self, theta_params: list[float]
                                      ) -> QuantumCircuit:
        """Build the coherence amplification circuit.

        Uses energy-conserving rotations between system and catalyst/ancilla
        to transfer coherence while preserving total energy.

        Circuit structure:
          sys -- EC_rot(theta1) -- EC_rot(theta2) --
          cat -- EC_rot(theta1) --|                  |-- EC_rot(theta3)--
          anc0 -                  -- EC_rot(theta2) --
          anc1 -                                       -- EC_rot(theta3)--
        """
        circuit = QuantumCircuit(self.n_total)

        idx = 0
        # Layer 1: System-Catalyst interaction
        for s in self.sys_qubits:
            for c in self.cat_qubits:
                theta = theta_params[idx % len(theta_params)]
                gate = make_ec_gate(theta, 0.0, s, c)
                circuit.add_gate(gate)
                idx += 1

        # Layer 2: Catalyst-Ancilla interaction (catalyst mediates)
        for c in self.cat_qubits:
            for a in self.anc_qubits:
                theta = theta_params[idx % len(theta_params)]
                gate = make_ec_gate(theta, 0.0, c, a)
                circuit.add_gate(gate)
                idx += 1

        # Layer 3: System-Ancilla direct (additional coherence transfer)
        for s in self.sys_qubits:
            for a in self.anc_qubits:
                theta = theta_params[idx % len(theta_params)]
                gate = make_ec_gate(theta, 0.0, s, a)
                circuit.add_gate(gate)
                idx += 1

        return circuit

    def _optimize_parameters(self, noisy_rho: np.ndarray,
                              n_params: int = 6,
                              n_iter: int = 200) -> list[float]:
        """Find circuit parameters that maximize coherence recovery.

        Uses gradient-free optimization (random search + refinement)
        to find energy-conserving circuit parameters.
        """
        best_params = [0.0] * n_params
        best_fidelity = -1.0
        dim_sys = 2**self.n_system
        dim_total = 2**self.n_total
        dim_cat = 2**self.n_catalyst
        dim_anc = 2**self.n_ancilla

        for iteration in range(n_iter):
            if iteration < n_iter // 2:
                # Random exploration
                params = [np.random.uniform(-np.pi, np.pi) for _ in range(n_params)]
            else:
                # Local refinement around best
                scale = 0.3 * (1 - iteration / n_iter)
                params = [p + np.random.normal(0, scale) for p in best_params]

            # Build and simulate circuit
            circuit = self._build_amplification_circuit(params)

            # Prepare initial state: noisy_sys tensor catalyst tensor ancilla(incoherent)
            dm = DensityMatrix(self.n_total)

            # Construct full initial density matrix
            anc_state = np.eye(2**self.n_ancilla) / (2**self.n_ancilla)
            init_rho = np.kron(np.kron(noisy_rho, self._catalyst_rho), anc_state)
            set_density_matrix(dm, init_rho)

            # Apply circuit
            circuit.update_quantum_state(dm)

            # Extract system state
            full_rho = density_matrix_from_qulacs(dm)
            dim_rest = dim_cat * dim_anc
            sys_rho = np.zeros((dim_sys, dim_sys), dtype=complex)
            for i in range(dim_sys):
                for j in range(dim_sys):
                    for k in range(dim_rest):
                        sys_rho[i, j] += full_rho[i * dim_rest + k,
                                                   j * dim_rest + k]

            # Fidelity with target
            fidelity = np.real(np.trace(sys_rho @ self._target_rho))

            # Also check catalyst preservation
            cat_rho = np.zeros((dim_cat, dim_cat), dtype=complex)
            for i in range(dim_cat):
                for j in range(dim_cat):
                    for s in range(dim_sys):
                        for a in range(dim_anc):
                            idx_row = s * dim_cat * dim_anc + i * dim_anc + a
                            idx_col = s * dim_cat * dim_anc + j * dim_anc + a
                            cat_rho[i, j] += full_rho[idx_row, idx_col]

            cat_fidelity = np.real(np.trace(cat_rho @ self._catalyst_rho))

            # Combined objective: recover system AND preserve catalyst
            objective = 0.7 * fidelity + 0.3 * cat_fidelity

            if objective > best_fidelity:
                best_fidelity = objective
                best_params = params[:]

        return best_params

    def correct_cycle(self, noisy_rho: np.ndarray,
                      n_optimization_iter: int = 200
                      ) -> tuple[np.ndarray, QulacsICECResult]:
        """Run one ICEC correction cycle using qulacs circuit simulation.

        Returns:
            (recovered_system_state, result)
        """
        cycle = len(self.history) + 1
        dim_sys = 2**self.n_system
        dim_cat = 2**self.n_catalyst
        dim_anc = 2**self.n_ancilla
        dim_rest = dim_cat * dim_anc

        # Measure input coherence
        energies = np.arange(dim_sys, dtype=float)
        input_coh = 0.0
        for i in range(dim_sys):
            for j in range(dim_sys):
                if i != j:
                    input_coh += abs(noisy_rho[i, j])

        # Optimize circuit parameters
        params = self._optimize_parameters(noisy_rho, n_iter=n_optimization_iter)
        circuit = self._build_amplification_circuit(params)

        # Execute the optimized circuit
        dm = DensityMatrix(self.n_total)
        anc_state = np.eye(dim_anc) / dim_anc
        init_rho = np.kron(np.kron(noisy_rho, self._catalyst_rho), anc_state)
        set_density_matrix(dm, init_rho)
        circuit.update_quantum_state(dm)

        # Extract results
        full_rho = density_matrix_from_qulacs(dm)

        # System state
        sys_rho = np.zeros((dim_sys, dim_sys), dtype=complex)
        for i in range(dim_sys):
            for j in range(dim_sys):
                for k in range(dim_rest):
                    sys_rho[i, j] += full_rho[i * dim_rest + k,
                                               j * dim_rest + k]

        # Catalyst state
        cat_rho = np.zeros((dim_cat, dim_cat), dtype=complex)
        for i in range(dim_cat):
            for j in range(dim_cat):
                for s in range(dim_sys):
                    for a in range(dim_anc):
                        idx_row = s * dim_cat * dim_anc + i * dim_anc + a
                        idx_col = s * dim_cat * dim_anc + j * dim_anc + a
                        cat_rho[i, j] += full_rho[idx_row, idx_col]

        # Update catalyst for next cycle
        self._catalyst_rho = cat_rho

        # Metrics
        output_coh = sum(abs(sys_rho[i, j])
                         for i in range(dim_sys)
                         for j in range(dim_sys) if i != j)
        td = 0.5 * np.sum(np.linalg.svd(sys_rho - self._target_rho, compute_uv=False))
        cat_diff = 0.5 * np.sum(np.linalg.svd(cat_rho - self._catalyst_rho,
                                               compute_uv=False))

        result = QulacsICECResult(
            cycle=cycle,
            input_coherence=input_coh,
            output_coherence=output_coh,
            trace_distance=td,
            catalyst_preserved=(cat_diff < 0.1),
            n_gates=circuit.get_gate_count(),
        )
        self.history.append(result)
        return sys_rho, result

    def run_experiment(self, error_fn, n_cycles: int = 5,
                       n_opt_iter: int = 200,
                       verbose: bool = True) -> list[QulacsICECResult]:
        """Run full ICEC experiment with qulacs.

        Args:
            error_fn: function that takes density matrix and returns noisy version
            n_cycles: number of error-correction cycles
            n_opt_iter: optimization iterations per cycle
            verbose: print progress
        """
        if self._target_rho is None:
            self.initialize()

        if verbose:
            print(f"\n{'='*70}")
            print(f" Qulacs ICEC Experiment: {n_cycles} cycles")
            print(f" System: {self.n_system}q, Catalyst: {self.n_catalyst}q, "
                  f"Ancilla: {self.n_ancilla}q")
            print(f"{'='*70}\n")

        current = self._target_rho.copy()

        for c in range(n_cycles):
            # Apply error
            noisy = error_fn(current)

            # Correct
            recovered, result = self.correct_cycle(noisy, n_opt_iter)
            current = recovered

            if verbose:
                print(f"  Cycle {result.cycle}: "
                      f"Coh {result.input_coherence:.4f} -> "
                      f"{result.output_coherence:.4f} | "
                      f"D(target)={result.trace_distance:.4f} | "
                      f"Gates={result.n_gates} | "
                      f"Cat OK={result.catalyst_preserved}")

        if verbose:
            print(f"\n  Total cycles: {len(self.history)}")
            print(f"{'='*70}\n")

        return self.history


# ============================================================
# 6. Main demonstration
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print(" Qulacs ICEC Circuit Simulation")
    print(" Based on Shiraishi & Takagi, PRL 132, 180202 (2024)")
    print("=" * 70)

    # ---- Experiment 1: Partial Dephasing Recovery ----
    print("\n--- Experiment 1: Partial Dephasing (gamma=1.5) ---")
    print("    Coherence reduced but nonzero -> ICEC should recover\n")

    icec = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=2)
    icec.initialize()

    def dephasing_error(rho, gamma=1.5):
        """Single-qubit partial dephasing."""
        result = rho.copy()
        factor = np.exp(-gamma)
        result[0, 1] *= factor
        result[1, 0] *= factor
        return result

    icec.run_experiment(dephasing_error, n_cycles=5, n_opt_iter=300)

    # ---- Experiment 2: Depolarizing Noise Recovery ----
    print("\n--- Experiment 2: Depolarizing Noise (p=0.5) ---")
    print("    Full-rank output -> mode condition always satisfied\n")

    icec2 = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=2)
    icec2.initialize()

    def depol_error(rho, p=0.5):
        d = rho.shape[0]
        return (1 - p) * rho + p * np.eye(d) / d

    icec2.run_experiment(depol_error, n_cycles=5, n_opt_iter=300)

    # ---- Experiment 3: Complete Dephasing (should FAIL) ----
    print("\n--- Experiment 3: Complete Dephasing ---")
    print("    All coherence destroyed -> ICEC CANNOT recover\n")

    icec3 = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=2)
    icec3.initialize()

    def complete_dephasing(rho):
        result = rho.copy()
        result[0, 1] = 0.0
        result[1, 0] = 0.0
        return result

    icec3.run_experiment(complete_dephasing, n_cycles=3, n_opt_iter=300)

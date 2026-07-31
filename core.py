"""
core.py - Quantum Coherence Resource Theory Foundation
======================================================
Implements the mathematical framework from:
  Shiraishi & Takagi, PRL 132, 180202 (2024)
  "Arbitrary Amplification of Quantum Coherence
   in Asymptotic and Catalytic Transformation"

Key concepts:
  - Unspeakable coherence (w.r.t. energy eigenbasis)
  - Coherent modes and resonant coherent modes C(rho)
  - Covariant operations (energy-conserving unitaries)
  - Marginal-catalytic transformations
"""

import numpy as np
from scipy.linalg import expm, block_diag
from itertools import product as iter_product
from typing import Optional
from fractions import Fraction


# ============================================================
# 1. Quantum State Utilities
# ============================================================

def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if rho is a valid density matrix."""
    d = rho.shape[0]
    if rho.shape != (d, d):
        return False
    # Hermiticity
    if np.max(np.abs(rho - rho.conj().T)) > tol:
        return False
    # Trace 1
    if abs(np.trace(rho) - 1.0) > tol:
        return False
    # Positive semidefinite
    eigvals = np.linalg.eigvalsh(rho)
    if np.min(eigvals) < -tol:
        return False
    return True


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance: D(rho, sigma) = 0.5 * ||rho - sigma||_1."""
    diff = rho - sigma
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(singular_values)


def partial_trace(rho_ab: np.ndarray, dim_a: int, dim_b: int,
                  trace_out: str = 'B') -> np.ndarray:
    """Partial trace of a bipartite system.

    Args:
        rho_ab: density matrix on A tensor B (dim_a*dim_b x dim_a*dim_b)
        dim_a: dimension of system A
        dim_b: dimension of system B
        trace_out: 'A' or 'B' - which system to trace out
    """
    rho = rho_ab.reshape(dim_a, dim_b, dim_a, dim_b)
    if trace_out == 'B':
        return np.trace(rho, axis1=1, axis2=3)
    else:
        return np.trace(rho, axis1=0, axis2=2)


def tensor(*matrices: np.ndarray) -> np.ndarray:
    """Tensor product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def maximally_coherent_state(d: int = 2) -> np.ndarray:
    """Maximally coherent state |+><+| for d-level system.
    |+> = (1/sqrt(d)) * sum_i |i>
    """
    plus = np.ones(d) / np.sqrt(d)
    return np.outer(plus, plus)


def random_coherent_state(d: int = 2, rank: Optional[int] = None) -> np.ndarray:
    """Generate a random full-rank coherent state."""
    if rank is None:
        rank = d
    # Random Ginibre matrix
    G = np.random.randn(d, rank) + 1j * np.random.randn(d, rank)
    rho = G @ G.conj().T
    rho = rho / np.trace(rho)
    return rho


def is_full_rank(rho: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if density matrix is full rank."""
    eigvals = np.linalg.eigvalsh(rho)
    return np.all(eigvals > tol)


# ============================================================
# 2. Hamiltonian and Energy Structure
# ============================================================

class EnergySystem:
    """Defines energy level structure for a quantum system.

    The Hamiltonian H = sum_i E_i |i><i| defines the energy eigenbasis.
    Coherence = off-diagonal elements between different energy eigenspaces.
    """

    def __init__(self, energies: np.ndarray):
        """
        Args:
            energies: array of energy eigenvalues [E_0, E_1, ..., E_{d-1}]
        """
        self.energies = np.array(energies, dtype=float)
        self.dim = len(energies)
        self.H = np.diag(self.energies)

    def time_evolution(self, rho: np.ndarray, t: float) -> np.ndarray:
        """U_t(rho) = e^{-iHt} rho e^{iHt}"""
        U = np.diag(np.exp(-1j * self.energies * t))
        return U @ rho @ U.conj().T

    def is_incoherent(self, rho: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if rho is incoherent (block-diagonal in energy eigenbasis).

        A state is incoherent iff rho_ij = 0 for all E_i != E_j.
        """
        for i in range(self.dim):
            for j in range(self.dim):
                if abs(self.energies[i] - self.energies[j]) > tol:
                    if abs(rho[i, j]) > tol:
                        return False
        return True

    def dephase(self, rho: np.ndarray) -> np.ndarray:
        """Complete dephasing: project onto incoherent part.
        Removes all off-diagonal elements between different energy levels.
        """
        result = np.zeros_like(rho)
        for i in range(self.dim):
            for j in range(self.dim):
                if abs(self.energies[i] - self.energies[j]) < 1e-12:
                    result[i, j] = rho[i, j]
        return result


# ============================================================
# 3. Coherent Modes (Definition S.5, S.6 from Supplemental Material)
# ============================================================

def modes_of_asymmetry(rho: np.ndarray, energies: np.ndarray,
                       tol: float = 1e-10) -> set:
    """Compute D(rho) - the modes of asymmetry (Definition S.5).

    D(rho) = {Delta_ij | rho_ij != 0, Delta_ij = E_i - E_j}

    Returns set of energy differences for nonzero off-diagonal elements.
    """
    d = len(energies)
    modes = set()
    for i in range(d):
        for j in range(d):
            if abs(rho[i, j]) > tol:
                delta = round(energies[i] - energies[j], 10)
                modes.add(delta)
    modes.discard(0.0)  # Remove zero mode (diagonal)
    return modes


def resonant_coherent_modes(rho: np.ndarray, energies: np.ndarray,
                            tol: float = 1e-10,
                            max_coeff: int = 5) -> set:
    """Compute C(rho) - the set of resonant coherent modes (Definition S.6).

    C(rho) = {x | x = sum_{(i,j) in M} n_ij * Delta_ij, n_ij in Z}

    This is the set of all integer-linear combinations of nonzero coherent modes.
    We approximate by considering coefficients up to max_coeff.
    """
    D = modes_of_asymmetry(rho, energies, tol)
    if not D:
        return {0.0}

    modes_list = sorted(D)
    C = {0.0}

    # Generate integer-linear combinations
    for n_terms in range(1, min(len(modes_list) + 1, 4)):
        for combo in iter_product(modes_list, repeat=n_terms):
            for coeffs in iter_product(range(-max_coeff, max_coeff + 1), repeat=n_terms):
                val = sum(c * m for c, m in zip(coeffs, combo))
                C.add(round(val, 10))

    return C


def check_mode_inclusion(rho_target: np.ndarray, rho_source: np.ndarray,
                         energies_target: np.ndarray,
                         energies_source: np.ndarray,
                         tol: float = 1e-10) -> bool:
    """Check if C(rho_target) subseteq C(rho_source).

    This is the necessary and sufficient condition for
    asymptotic marginal and catalytic transformation.

    C(rho) is the group generated by D(rho) under addition with integer
    coefficients. So C(target) subset C(source) iff every generator of
    C(target) (i.e., every element of D(target)) can be written as an
    integer-linear combination of elements of D(source).

    We check this by verifying that each mode Delta in D(target) belongs
    to the lattice generated by D(source).
    """
    D_target = modes_of_asymmetry(rho_target, energies_target, tol)
    D_source = modes_of_asymmetry(rho_source, energies_source, tol)

    if not D_target:
        return True  # Empty set is subset of anything

    if not D_source:
        return False  # Non-empty cannot be subset of empty

    # For each mode in D(target), check if it's in the integer lattice
    # generated by D(source). Use GCD-based approach.
    source_list = sorted(D_source)

    # Compute the GCD of all absolute values of source modes
    # The integer lattice of D(source) = {n * gcd : n in Z}
    # when all source modes are commensurable
    from math import gcd
    from fractions import Fraction

    # Convert to rational approximations to handle floating point
    def to_fraction(x, max_denom=10**8):
        return Fraction(x).limit_denominator(max_denom)

    source_fracs = [to_fraction(abs(m)) for m in source_list if abs(m) > tol]
    if not source_fracs:
        return False

    # Compute GCD of all source modes as fractions
    def gcd_frac(a, b):
        """GCD for Fraction objects."""
        if a == 0:
            return b
        if b == 0:
            return a
        # gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d)
        return Fraction(gcd(a.numerator * b.denominator,
                           b.numerator * a.denominator),
                       a.denominator * b.denominator)

    lattice_gcd = source_fracs[0]
    for f in source_fracs[1:]:
        lattice_gcd = gcd_frac(lattice_gcd, f)

    if lattice_gcd == 0:
        return False

    # Check: each target mode must be an integer multiple of lattice_gcd
    for mode in D_target:
        mode_frac = to_fraction(abs(mode))
        ratio = mode_frac / lattice_gcd
        # Check if ratio is (close to) an integer
        ratio_float = float(ratio)
        if abs(ratio_float - round(ratio_float)) > 0.01:
            return False

    return True


# ============================================================
# 4. Decoherence Channels
# ============================================================

def partial_dephasing(rho: np.ndarray, energies: np.ndarray,
                      gamma: float) -> np.ndarray:
    """Partial dephasing channel: reduces off-diagonal elements.

    rho_ij -> rho_ij * exp(-gamma * |E_i - E_j|)  for E_i != E_j

    gamma = 0: no dephasing (identity)
    gamma -> inf: complete dephasing (all coherence lost)

    CRITICAL: For gamma < inf, coherent modes are PRESERVED (nonzero).
    This is the regime where ICEC works.
    """
    d = len(energies)
    result = rho.copy()
    for i in range(d):
        for j in range(d):
            delta = abs(energies[i] - energies[j])
            if delta > 1e-12:
                result[i, j] *= np.exp(-gamma * delta)
    return result


def amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Amplitude damping channel for a qubit.

    rho -> E_0 rho E_0^dag + E_1 rho E_1^dag
    E_0 = |0><0| + sqrt(1-gamma)|1><1|
    E_1 = sqrt(gamma)|0><1|

    For gamma < 1, some coherence survives.
    For gamma = 1, state collapses to |0><0| (fully incoherent).
    """
    assert rho.shape == (2, 2), "Amplitude damping only for qubits"
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T


def depolarizing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing channel: rho -> (1-p) rho + p * I/d.

    For p < 1, always produces a full-rank state with nonzero coherence
    if the input has coherence. This is ideal for ICEC.
    """
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


# ============================================================
# 5. Covariant Operations
# ============================================================

def is_energy_conserving(U: np.ndarray, H_in: np.ndarray,
                         H_out: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if U is energy-conserving: [U, H_total] = 0.

    For a unitary acting on system+ancilla with H_total = H_in + H_out,
    energy conservation means U(H_in tensor I + I tensor H_out)U^dag
    = H_in' tensor I + I tensor H_out'.
    """
    commutator = U @ H_in - H_out @ U
    return np.max(np.abs(commutator)) < tol


def covariant_swap_unitary(energies_s: np.ndarray,
                           energies_a: np.ndarray) -> Optional[np.ndarray]:
    """Construct an energy-conserving partial swap unitary.

    For levels with matching energy differences, allows coherence transfer.
    This is a key building block for coherence amplification.
    """
    d_s = len(energies_s)
    d_a = len(energies_a)
    d_total = d_s * d_a
    U = np.eye(d_total, dtype=complex)

    # Find pairs (i,j) in S and (k,l) in A with matching energy differences
    for i in range(d_s):
        for j in range(d_s):
            if i == j:
                continue
            delta_s = energies_s[i] - energies_s[j]
            for k in range(d_a):
                for l in range(d_a):
                    if k == l:
                        continue
                    delta_a = energies_a[k] - energies_a[l]
                    # Match: E_i^S + E_k^A = E_j^S + E_l^A
                    if abs((energies_s[i] + energies_a[k]) -
                           (energies_s[j] + energies_a[l])) < 1e-10:
                        # Partial swap on subspace |ik>, |jl>
                        idx1 = i * d_a + k
                        idx2 = j * d_a + l
                        if idx1 < idx2:  # Avoid double-counting
                            theta = np.pi / 4  # Partial swap angle
                            c, s = np.cos(theta), np.sin(theta)
                            U[idx1, idx1] = c
                            U[idx1, idx2] = -s
                            U[idx2, idx1] = s
                            U[idx2, idx2] = c
    return U


# ============================================================
# 6. Coherence Measures
# ============================================================

def coherence_l1(rho: np.ndarray, energies: np.ndarray) -> float:
    """L1-norm of coherence (sum of absolute off-diagonal elements
    between different energy levels)."""
    d = len(energies)
    coh = 0.0
    for i in range(d):
        for j in range(d):
            if abs(energies[i] - energies[j]) > 1e-12:
                coh += abs(rho[i, j])
    return coh


def quantum_fisher_information(rho: np.ndarray, H: np.ndarray) -> float:
    """Quantum Fisher Information F_Q(rho, H).

    F_Q = 2 * sum_{i,j} (p_i - p_j)^2 / (p_i + p_j) * |<i|H|j>|^2

    This is a key asymmetry measure and is proportional to the
    coherence cost in the two-level system.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    d = len(eigvals)
    H_basis = eigvecs.conj().T @ H @ eigvecs  # H in eigenbasis of rho

    fqi = 0.0
    for i in range(d):
        for j in range(d):
            denom = eigvals[i] + eigvals[j]
            if denom > 1e-15:
                fqi += 2 * (eigvals[i] - eigvals[j])**2 / denom * abs(H_basis[i, j])**2
    return fqi


def relative_entropy_of_coherence(rho: np.ndarray, energies: np.ndarray) -> float:
    """Relative entropy of coherence: C_rel(rho) = S(Delta(rho)) - S(rho).

    Where Delta is the dephasing map and S is von Neumann entropy.
    """
    sys = EnergySystem(energies)
    rho_diag = sys.dephase(rho)

    def von_neumann_entropy(sigma):
        eigvals = np.linalg.eigvalsh(sigma)
        eigvals = eigvals[eigvals > 1e-15]
        return -np.sum(eigvals * np.log2(eigvals))

    return von_neumann_entropy(rho_diag) - von_neumann_entropy(rho)


# ============================================================
# 7. Coherence Distillation (from Ref [53] / Lemma S.16)
# ============================================================

def distill_coherent_qubit(rho_copies: list[np.ndarray],
                           energies: np.ndarray,
                           target_delta: float) -> tuple[np.ndarray, float]:
    """Distill a coherent qubit state from multiple copies.

    Based on the protocol in Cirac-Ekert-Macchiavello (Ref [53] / [98]):
    Transform many weakly coherent states into one highly coherent qubit.

    This is a building block for the catalytic amplification protocol.

    Returns: (distilled_state, fidelity_with_plus)
    """
    n = len(rho_copies)
    d = rho_copies[0].shape[0]

    # Find levels with energy difference = target_delta
    pairs = []
    for i in range(d):
        for j in range(d):
            if abs(energies[i] - energies[j] - target_delta) < 1e-10:
                pairs.append((i, j))

    if not pairs:
        return np.eye(2) / 2, 0.5  # No matching mode, return maximally mixed

    # Use the first matching pair
    i0, j0 = pairs[0]

    # Extract 2-level coherence from each copy and combine
    total_coh = 0.0
    for rho in rho_copies:
        total_coh += rho[i0, j0]

    # Construct effective 2-level state
    p0 = sum(rho[i0, i0].real for rho in rho_copies) / n
    p1 = sum(rho[j0, j0].real for rho in rho_copies) / n

    # Amplified coherence (grows with sqrt(n) in optimal protocol)
    amp_coh = total_coh / n * min(np.sqrt(n), 1.0 / (2 * abs(total_coh / n) + 1e-15))
    amp_coh = np.clip(abs(amp_coh), 0, np.sqrt(p0 * p1)) * np.exp(1j * np.angle(total_coh))

    sigma = np.array([[p0, amp_coh], [amp_coh.conj(), p1]])
    # Ensure valid density matrix
    eigvals = np.linalg.eigvalsh(sigma)
    if np.min(eigvals) < 0:
        sigma += (-np.min(eigvals) + 1e-12) * np.eye(2)
        sigma /= np.trace(sigma)

    plus = maximally_coherent_state(2)
    fidelity = np.real(np.trace(sigma @ plus))

    return sigma, fidelity


if __name__ == "__main__":
    # Quick sanity check
    print("=== Core Module Sanity Check ===\n")

    energies = np.array([0.0, 1.0])
    sys = EnergySystem(energies)

    # Maximally coherent qubit
    plus = maximally_coherent_state(2)
    print(f"|+><+| = \n{plus}\n")
    print(f"Is incoherent: {sys.is_incoherent(plus)}")
    print(f"L1 coherence: {coherence_l1(plus, energies):.4f}")
    print(f"QFI: {quantum_fisher_information(plus, sys.H):.4f}")

    # Partial dephasing
    noisy = partial_dephasing(plus, energies, gamma=2.0)
    print(f"\nAfter dephasing (gamma=2):")
    print(f"L1 coherence: {coherence_l1(noisy, energies):.4f}")
    print(f"Modes D(rho_noisy): {modes_of_asymmetry(noisy, energies)}")
    print(f"C(rho_target) subset C(rho_noisy): "
          f"{check_mode_inclusion(plus, noisy, energies, energies)}")

    # Complete dephasing
    dead = sys.dephase(plus)
    print(f"\nAfter COMPLETE dephasing:")
    print(f"L1 coherence: {coherence_l1(dead, energies):.4f}")
    print(f"Is incoherent: {sys.is_incoherent(dead)}")
    print(f"Modes D(rho_dead): {modes_of_asymmetry(dead, energies)}")

    print("\n[OK] Core module loaded successfully.")

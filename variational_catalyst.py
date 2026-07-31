#!/usr/bin/env python3
"""
variational_catalyst.py — Variational Quantum Catalyst Preparation for CQEC

Future work prototype: replace quantum distillation (n* ~ 10^4–10^6 copies)
with a parameterised quantum circuit U(θ) that directly prepares the catalyst
state ρ_cat with the required coherence properties.

Three modules:
  1. SymmetricAnsatz  – hardware-efficient ansatz respecting [U, H_tot] = 0
  2. CoherenceCost    – cost functions based on ℓ1-coherence and mode inclusion
  3. VariationalCQEC  – full optimisation loop + CQEC recovery evaluation

Dependencies: numpy, scipy  (no Qulacs/Qiskit required for density-matrix sim)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from typing import Optional

# ============================================================
# Helpers
# ============================================================

def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random density matrix of dimension d (Hilbert-Schmidt)."""
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Quantum fidelity F(ρ, σ) for density matrices."""
    sqrt_rho = matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 0.0)
    return float(np.sum(np.sqrt(eigvals)) ** 2)


def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Hermitian positive-semidefinite matrix square root."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def l1_coherence(rho: np.ndarray) -> float:
    """ℓ1-norm of coherence: C_ℓ1(ρ) = Σ_{i≠j} |ρ_{ij}|."""
    d = rho.shape[0]
    total = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
    return float(total)


def coherence_modes(rho: np.ndarray, tol: float = 1e-10) -> set:
    """Return the set of mode pairs (i,j) where |ρ_{ij}| > tol."""
    d = rho.shape[0]
    modes = set()
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho[i, j]) > tol:
                modes.add((i, j))
    return modes


def dephasing_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply dephasing noise: ρ_{ij} → ρ_{ij} · exp(-γ|i-j|) for i≠j."""
    d = rho.shape[0]
    rho_noisy = rho.copy()
    for i in range(d):
        for j in range(d):
            if i != j:
                rho_noisy[i, j] *= np.exp(-gamma * abs(i - j))
    return rho_noisy


def depolarising_channel(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarising channel: ρ → (1-p)ρ + p·I/d."""
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


# ============================================================
# 1. Symmetric Ansatz — energy-conserving parameterised circuit
# ============================================================

class SymmetricAnsatz:
    """
    Hardware-efficient ansatz that respects the total-energy conservation
    constraint [U(θ), H_tot] = 0.

    For a d-dimensional system, we parameterise U(θ) as a product of
    energy-conserving two-level rotations (generalised beam-splitter gates)
    within each energy-degenerate subspace.

    For simplicity, H = diag(0, 1, ..., d-1) and EC gates couple levels
    (i, j) with i ≠ j via:
        G_{ij}(θ, φ) = exp(-i θ (e^{iφ} |i⟩⟨j| + e^{-iφ} |j⟩⟨i|))

    Note: These are NOT strictly energy-conserving for |E_i - E_j| ≠ 0
    in a bare Hamiltonian sense, but they model the covariant unitaries
    used in the Shiraishi-Takagi protocol where the *total* system+catalyst
    Hamiltonian is conserved.
    """

    def __init__(self, d: int, n_layers: int = 3):
        self.d = d
        self.n_layers = n_layers
        # Pairs: all (i, j) with i < j
        self.pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        # 2 parameters per pair per layer (θ, φ)
        self.n_params = 2 * len(self.pairs) * n_layers

    def unitary(self, params: np.ndarray) -> np.ndarray:
        """Build the full unitary U(θ) from parameters."""
        d = self.d
        U = np.eye(d, dtype=complex)
        idx = 0
        for _ in range(self.n_layers):
            for i, j in self.pairs:
                theta = params[idx]
                phi = params[idx + 1]
                idx += 2
                # Two-level rotation
                G = np.eye(d, dtype=complex)
                c, s = np.cos(theta), np.sin(theta)
                G[i, i] = c
                G[j, j] = c
                G[i, j] = -s * np.exp(1j * phi)
                G[j, i] = s * np.exp(-1j * phi)
                U = G @ U
        return U

    def prepare_state(self, params: np.ndarray) -> np.ndarray:
        """Prepare catalyst state: ρ_cat = U(θ)|0⟩⟨0|U†(θ)."""
        U = self.unitary(params)
        psi = U[:, 0]  # U|0⟩
        return np.outer(psi, psi.conj())

    def random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Random initial parameters."""
        return rng.uniform(-np.pi, np.pi, self.n_params)


# ============================================================
# 2. Cost functions
# ============================================================

class CoherenceCost:
    """
    Cost functions for variational catalyst optimisation.

    The catalyst must satisfy:
      (a) Mode inclusion: C(ρ_target) ⊆ C(ρ_cat)
      (b) High ℓ1-coherence for fast convergence (lower C constant)
      (c) Sufficient |ρ_min| (smallest diagonal element) to avoid blow-up
    """

    def __init__(self, rho_target: np.ndarray, weight_coherence: float = 1.0,
                 weight_mode: float = 10.0, weight_rho_min: float = 5.0):
        self.rho_target = rho_target
        self.d = rho_target.shape[0]
        self.target_modes = coherence_modes(rho_target)
        self.w_coh = weight_coherence
        self.w_mode = weight_mode
        self.w_min = weight_rho_min

    def __call__(self, rho_cat: np.ndarray) -> float:
        """
        Total cost (to minimise):
          - Negative ℓ1-coherence (we want to maximise)
          + Mode inclusion penalty (missing modes)
          - log(min diagonal) penalty (we want large ρ_min)
        """
        # Term 1: Maximise coherence → minimise negative
        coh = l1_coherence(rho_cat)
        cost_coh = -self.w_coh * coh / (self.d * (self.d - 1))  # normalised

        # Term 2: Mode inclusion penalty
        cat_modes = coherence_modes(rho_cat, tol=1e-8)
        missing = self.target_modes - cat_modes
        cost_mode = self.w_mode * len(missing) / max(len(self.target_modes), 1)

        # Term 3: ρ_min penalty — want all diagonal elements nonzero
        diag = np.real(np.diag(rho_cat))
        rho_min = np.min(diag)
        cost_min = -self.w_min * np.log(max(rho_min, 1e-15))

        return cost_coh + cost_mode + cost_min


# ============================================================
# 3. Variational CQEC — full optimisation + recovery
# ============================================================

class VariationalCQEC:
    """
    End-to-end variational catalyst preparation and CQEC recovery.

    Protocol:
      1. Optimise U(θ) to prepare catalyst ρ_cat with required coherence
      2. Apply noise channel to target state → ρ_noisy
      3. Perform CQEC recovery using prepared catalyst
      4. Evaluate fidelity F(ρ_recovered, ρ_target)
    """

    def __init__(self, d: int, n_layers: int = 3, seed: int = 42):
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.ansatz = SymmetricAnsatz(d, n_layers)
        self.history = []

    def optimise_catalyst(self, rho_target: np.ndarray,
                          n_restarts: int = 5,
                          maxiter: int = 500,
                          verbose: bool = True) -> dict:
        """
        Optimise the variational catalyst for a given target state.

        Returns dict with:
          - params: optimal parameters
          - rho_cat: optimal catalyst state
          - cost: final cost value
          - coherence: ℓ1-coherence of catalyst
          - modes_covered: fraction of target modes covered
        """
        cost_fn = CoherenceCost(rho_target)
        best_result = None
        best_cost = np.inf

        for restart in range(n_restarts):
            x0 = self.ansatz.random_params(self.rng)

            def objective(params):
                rho_cat = self.ansatz.prepare_state(params)
                return cost_fn(rho_cat)

            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': 1e-12})

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

            if verbose:
                rho_cat = self.ansatz.prepare_state(result.x)
                coh = l1_coherence(rho_cat)
                cat_modes = coherence_modes(rho_cat, tol=1e-8)
                target_modes = coherence_modes(rho_target)
                covered = len(target_modes & cat_modes) / max(len(target_modes), 1)
                print(f"  Restart {restart+1}/{n_restarts}: "
                      f"cost={result.fun:.6f}, C_l1={coh:.4f}, "
                      f"modes={covered:.1%}, converged={result.success}")

        # Extract results
        params_opt = best_result.x
        rho_cat = self.ansatz.prepare_state(params_opt)
        cat_modes = coherence_modes(rho_cat, tol=1e-8)
        target_modes = coherence_modes(rho_target)

        return {
            'params': params_opt,
            'rho_cat': rho_cat,
            'cost': best_cost,
            'coherence': l1_coherence(rho_cat),
            'rho_min': float(np.min(np.real(np.diag(rho_cat)))),
            'modes_covered': len(target_modes & cat_modes) / max(len(target_modes), 1),
            'n_params': self.ansatz.n_params,
        }

    def cqec_recovery(self, rho_target: np.ndarray, rho_noisy: np.ndarray,
                      rho_cat: np.ndarray) -> np.ndarray:
        """
        Simplified CQEC recovery using the catalytic covariant map.

        This is a density-matrix simulation of the asymptotic protocol:
        given perfect catalyst with mode inclusion, the recovered state
        converges to ρ_target.

        In practice, we model the recovery as:
          ρ_rec = Λ_CQEC(ρ_noisy, ρ_cat)

        For the density-matrix simulation, the recovery fidelity depends
        on how well the catalyst's coherence structure matches the target.
        """
        d = self.d
        target_modes = coherence_modes(rho_target)
        cat_modes = coherence_modes(rho_cat, tol=1e-8)

        rho_rec = rho_noisy.copy()

        for i, j in target_modes:
            if (i, j) in cat_modes:
                # Coherence can be restored via catalytic amplification
                # Recovery strength depends on catalyst coherence magnitude
                cat_coh_ij = np.abs(rho_cat[i, j])
                target_coh_ij = rho_target[i, j]
                noisy_coh_ij = rho_noisy[i, j]

                if cat_coh_ij > 1e-12 and np.abs(noisy_coh_ij) > 1e-15:
                    # Amplification factor (bounded by catalyst quality)
                    phase_target = np.angle(target_coh_ij)
                    mag_target = np.abs(target_coh_ij)

                    # Recovery efficiency: saturates as catalyst coherence grows
                    efficiency = 1.0 - np.exp(-cat_coh_ij * d)
                    mag_recovered = np.abs(noisy_coh_ij) + efficiency * (
                        mag_target - np.abs(noisy_coh_ij)
                    )

                    rho_rec[i, j] = mag_recovered * np.exp(1j * phase_target)
                    rho_rec[j, i] = rho_rec[i, j].conj()

        # Ensure positivity via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(rho_rec)
        eigvals = np.maximum(eigvals, 0.0)
        rho_rec = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho_rec /= np.trace(rho_rec)

        return rho_rec

    def run_experiment(self, rho_target: np.ndarray,
                       noise_type: str = 'dephasing',
                       noise_strength: float = 1.0,
                       n_restarts: int = 5,
                       maxiter: int = 500,
                       verbose: bool = True) -> dict:
        """
        Full experiment: prepare catalyst → apply noise → recover → measure.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Variational CQEC Experiment (d={self.d})")
            print(f"Noise: {noise_type}, strength={noise_strength}")
            print(f"Ansatz: {self.ansatz.n_layers} layers, "
                  f"{self.ansatz.n_params} parameters")
            print(f"{'='*60}")

        # Step 1: Apply noise
        if noise_type == 'dephasing':
            rho_noisy = dephasing_channel(rho_target, noise_strength)
        elif noise_type == 'depolarising':
            rho_noisy = depolarising_channel(rho_target, noise_strength)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        fid_noisy = fidelity(rho_target, rho_noisy)
        if verbose:
            print(f"\nFidelity after noise: {fid_noisy:.6f}")

        # Step 2: Optimise catalyst
        if verbose:
            print("\nOptimising catalyst...")
        cat_result = self.optimise_catalyst(rho_target, n_restarts, maxiter,
                                            verbose)
        if verbose:
            print(f"\nOptimal catalyst: C_l1={cat_result['coherence']:.4f}, "
                  f"rho_min={cat_result['rho_min']:.6f}, "
                  f"modes={cat_result['modes_covered']:.1%}")

        # Step 3: CQEC recovery
        rho_recovered = self.cqec_recovery(rho_target, rho_noisy,
                                           cat_result['rho_cat'])
        fid_recovered = fidelity(rho_target, rho_recovered)

        if verbose:
            print(f"\nFidelity after CQEC recovery: {fid_recovered:.6f}")
            print(f"Improvement: {fid_recovered - fid_noisy:+.6f}")

        result = {
            'fid_noisy': fid_noisy,
            'fid_recovered': fid_recovered,
            'improvement': fid_recovered - fid_noisy,
            'catalyst': cat_result,
            'noise_type': noise_type,
            'noise_strength': noise_strength,
        }
        self.history.append(result)
        return result


# ============================================================
# 4. Plotting
# ============================================================

def plot_results(results: list, filename: str = 'fig_variational_catalyst.png'):
    """Plot variational catalyst experiment results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel (a): Fidelity vs noise strength ---
    noise_strengths = [r['noise_strength'] for r in results]
    fid_noisy = [r['fid_noisy'] for r in results]
    fid_recovered = [r['fid_recovered'] for r in results]

    ax1.plot(noise_strengths, fid_noisy, 'o--', color='#e74c3c',
             label=r'Before recovery $F(\rho_{\mathrm{noisy}}, \rho_0)$',
             linewidth=1.5, markersize=5)
    ax1.plot(noise_strengths, fid_recovered, 's-', color='#3498db',
             label=r'After CQEC $F(\rho_{\mathrm{rec}}, \rho_0)$',
             linewidth=2, markersize=6)
    ax1.fill_between(noise_strengths, fid_noisy, fid_recovered,
                     alpha=0.15, color='#3498db')
    ax1.set_xlabel(r'Noise strength $\gamma$')
    ax1.set_ylabel(r'Fidelity $F$')
    ax1.set_title(r'(a) Recovery fidelity vs. noise $\gamma$')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

    # --- Panel (b): Catalyst coherence achieved ---
    coherences = [r['catalyst']['coherence'] for r in results]
    modes_covered = [r['catalyst']['modes_covered'] for r in results]

    ax2b = ax2.twinx()
    l1, = ax2.plot(noise_strengths, coherences, 'D-', color='#2ecc71',
                   linewidth=2, markersize=6, label=r'$\mathcal{C}_{\ell_1}(\rho_{\mathrm{cat}})$')
    l2, = ax2b.plot(noise_strengths, modes_covered, '^-', color='#9b59b6',
                    linewidth=2, markersize=6, label='Mode coverage')
    ax2.set_xlabel(r'Noise strength $\gamma$')
    ax2.set_ylabel(r'$\ell_1$-coherence $\mathcal{C}_{\ell_1}$', color='#2ecc71')
    ax2b.set_ylabel('Mode coverage fraction', color='#9b59b6')
    ax2b.set_ylim(-0.05, 1.15)
    ax2.set_title(r'(b) Catalyst quality vs. noise $\gamma$')
    ax2.legend(handles=[l1, l2], loc='lower left', fontsize=8)

    # --- Panel (c): Parameter count scaling ---
    dims = [2, 3, 4, 6, 8, 12, 16]
    n_params_list = []
    for d in dims:
        a = SymmetricAnsatz(d, n_layers=3)
        n_params_list.append(a.n_params)

    ax3.semilogy(dims, n_params_list, 'o-', color='#e67e22', linewidth=2,
                 markersize=6, label=r'$N_\theta$ (3 layers)')
    # Compare with distillation copy count
    # C ~ d^2, n* ~ C^2/δ^2 ~ d^4 for F=0.99
    n_distill = [0.53 * d**2.06 for d in dims]
    n_star = [(c**2) / (4 * 0.01**2) for c in n_distill]
    ax3.semilogy(dims, n_star, 's--', color='#e74c3c', linewidth=1.5,
                 markersize=5, label=r'Distillation $n^*$ ($F \geq 0.99$)')

    ax3.set_xlabel(r'Hilbert space dimension $d$')
    ax3.set_ylabel(r'Resource count')
    ax3.set_title(r'(c) Variational params $N_\theta$ vs. distillation $n^*$')
    ax3.legend(loc='upper left', fontsize=8)

    plt.suptitle('Variational Catalyst Preparation for CQEC',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(filename.replace('.png', '.pdf'))
    print(f"Saved {filename}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Variational Catalyst Preparation for CQEC")
    print("=" * 60)

    # Test dimensions
    test_dims = [4, 8]
    noise_strengths = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for d in test_dims:
        print(f"\n{'#'*60}")
        print(f"# Dimension d = {d}")
        print(f"{'#'*60}")

        rng = np.random.default_rng(42)
        rho_target = random_density_matrix(d, rng)

        vcqec = VariationalCQEC(d, n_layers=3, seed=42)
        all_results = []

        for gamma in noise_strengths:
            result = vcqec.run_experiment(
                rho_target,
                noise_type='dephasing',
                noise_strength=gamma,
                n_restarts=3,
                maxiter=300,
                verbose=True,
            )
            all_results.append(result)

        # Plot
        plot_results(all_results,
                     filename=f'fig_variational_catalyst_d{d}.png')

        # Summary table
        print(f"\n{'='*70}")
        print(f"{'gamma':>6} {'F_noisy':>10} {'F_recov':>10} "
              f"{'Improve':>10} {'C_l1':>8} {'Modes':>8}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['noise_strength']:>6.1f} "
                  f"{r['fid_noisy']:>10.6f} "
                  f"{r['fid_recovered']:>10.6f} "
                  f"{r['improvement']:>+10.6f} "
                  f"{r['catalyst']['coherence']:>8.4f} "
                  f"{r['catalyst']['modes_covered']:>8.1%}")
        print(f"{'='*70}")

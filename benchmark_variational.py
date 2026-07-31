#!/usr/bin/env python3
"""
benchmark_variational.py — Run VariationalCQEC on the same benchmarks as paper_prxq.

Compares variational catalyst preparation + CQEC recovery against the
asymptotic (oracle) results from the paper for:
  - QKAN (d=4), qDRIFT (d=8), CF-QPE (d=16), Regev (d=64)
  - Noise: dephasing (γ=2), depolarizing (p=0.3), combined
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variational_catalyst import (
    VariationalCQEC, random_density_matrix,
    dephasing_channel, depolarising_channel, fidelity,
    l1_coherence, coherence_modes, SymmetricAnsatz,
)

# ============================================================
# Generate representative algorithmic states
# ============================================================

def make_qkan_state(rng):
    """QKAN: d=4, Chebyshev amplitudes T_n(0.5)."""
    amps = np.array([1.0, 0.5, -0.5, -1.0])  # T_0..T_3
    # Algorithmic output truncates T_3
    amps_alg = np.array([1.0, 0.5, -0.5, 0.0])
    amps_alg = amps_alg / np.linalg.norm(amps_alg)
    psi = amps_alg.astype(complex)
    return np.outer(psi, psi.conj())


def make_qdrift_state(rng):
    """qDRIFT: d=8, Heisenberg model e^{-iHt}|000⟩."""
    d = 8
    # Heisenberg Hamiltonian for 3 qubits (simplified)
    # Generate a structured state with realistic coherence
    H = np.zeros((d, d), dtype=complex)
    J, h = 1.0, 0.5
    # XX + YY + ZZ terms for nearest-neighbor pairs
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    for pair in [(0, 1), (1, 2)]:
        for pauli in [sx, sy, sz]:
            ops = [I2, I2, I2]
            ops[pair[0]] = pauli
            ops[pair[1]] = pauli
            term = np.kron(np.kron(ops[0], ops[1]), ops[2])
            H += J * term

    for q in range(3):
        ops = [I2, I2, I2]
        ops[q] = sz
        term = np.kron(np.kron(ops[0], ops[1]), ops[2])
        H += h * term

    t = 1.0
    U = np.linalg.matrix_power(
        np.eye(d) - 1j * H * t / 1000, 1000
    )  # Simple Trotter
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    psi = U @ psi0
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def make_cfqpe_state(rng):
    """CF-QPE: d=16, phase estimation spectrum state."""
    d = 16
    # Simulate a 4-qubit state with rich coherence structure
    phases = rng.uniform(0, 2 * np.pi, d)
    amps = np.exp(1j * phases) / np.sqrt(d)
    # Give non-uniform amplitudes (Fermi-Hubbard-like)
    weights = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25,
                        0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05])
    amps = amps * weights
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj())


def make_regev_state(rng):
    """Regev (reduced): d=16, discrete Gaussian lattice state.
    Full d=64 has ~20k ansatz params; d=32 has ~3k. Both too slow.
    We use d=16 to match CF-QPE dimension but with a qualitatively
    different coherence structure (Gaussian vs uniform)."""
    d = 16
    grid = np.arange(d)
    sigma = d / 4
    amps = np.exp(-grid**2 / (2 * sigma**2))
    phases = np.exp(2j * np.pi * grid * 9 / d)
    amps = amps * phases
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj())


def amplitude_damping_channel(rho, gamma_ad):
    """Single-qubit amplitude damping applied to each qubit."""
    n_qubits = int(np.log2(rho.shape[0]))
    d = rho.shape[0]
    result = rho.copy()

    E0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma_ad)]], dtype=complex)
    E1_1q = np.array([[0, np.sqrt(gamma_ad)], [0, 0]], dtype=complex)

    for q in range(n_qubits):
        I_pre = np.eye(2**q, dtype=complex)
        I_post = np.eye(2**(n_qubits - q - 1), dtype=complex)
        E0 = np.kron(np.kron(I_pre, E0_1q), I_post)
        E1 = np.kron(np.kron(I_pre, E1_1q), I_post)
        result = E0 @ result @ E0.conj().T + E1 @ result @ E1.conj().T

    return result


def combined_noise(rho, gamma_deph=1.0, p_depol=0.15, gamma_ad=0.1):
    """Combined noise: dephasing + depolarizing + amplitude damping."""
    rho_out = dephasing_channel(rho, gamma_deph)
    rho_out = depolarising_channel(rho_out, p_depol)
    rho_out = amplitude_damping_channel(rho_out, gamma_ad)
    return rho_out


# ============================================================
# Benchmark configurations
# ============================================================

benchmarks = {
    'QKAN': {'d': 4, 'n_qubits': 2, 'make_state': make_qkan_state,
             'n_layers': 3, 'n_restarts': 5, 'maxiter': 500},
    'qDRIFT': {'d': 8, 'n_qubits': 3, 'make_state': make_qdrift_state,
               'n_layers': 3, 'n_restarts': 5, 'maxiter': 500},
    'CF-QPE': {'d': 16, 'n_qubits': 4, 'make_state': make_cfqpe_state,
               'n_layers': 4, 'n_restarts': 5, 'maxiter': 800},
    'Regev': {'d': 16, 'n_qubits': 4, 'make_state': make_regev_state,
              'n_layers': 3, 'n_restarts': 3, 'maxiter': 500},
}

noise_models = {
    'Dephasing': lambda rho: dephasing_channel(rho, 2.0),
    'Depolarizing': lambda rho: depolarising_channel(rho, 0.3),
    'Combined': lambda rho: combined_noise(rho),
}


# ============================================================
# Run benchmarks
# ============================================================

def run_benchmarks():
    rng = np.random.default_rng(42)
    all_results = {}

    print("=" * 90)
    print("Variational CQEC Benchmark")
    print("=" * 90)

    for alg_name, alg_cfg in benchmarks.items():
        d = alg_cfg['d']
        print(f"\n{'#' * 90}")
        print(f"# {alg_name} (d={d}, {alg_cfg['n_qubits']} qubits)")
        print(f"# Ansatz: {alg_cfg['n_layers']} layers, "
              f"{SymmetricAnsatz(d, alg_cfg['n_layers']).n_params} params")
        print(f"{'#' * 90}")

        rho_target = alg_cfg['make_state'](rng)
        coh = l1_coherence(rho_target)
        modes = coherence_modes(rho_target)
        print(f"Target state: C_l1 = {coh:.4f}, |D| = {len(modes)} modes")

        vcqec = VariationalCQEC(d, n_layers=alg_cfg['n_layers'], seed=42)
        alg_results = {}

        for noise_name, noise_fn in noise_models.items():
            t0 = time.time()
            print(f"\n--- {noise_name} ---")

            rho_noisy = noise_fn(rho_target)
            fid_noisy = fidelity(rho_target, rho_noisy)
            coh_noisy = l1_coherence(rho_noisy)
            modes_noisy = coherence_modes(rho_noisy)
            mode_incl = modes.issubset(modes_noisy)

            print(f"  F_noisy = {fid_noisy:.6f}, C_l1(noisy) = {coh_noisy:.4f}")
            print(f"  Mode inclusion: {mode_incl} "
                  f"({len(modes & modes_noisy)}/{len(modes)} modes)")

            # Optimise catalyst
            cat_result = vcqec.optimise_catalyst(
                rho_target,
                n_restarts=alg_cfg['n_restarts'],
                maxiter=alg_cfg['maxiter'],
                verbose=False,
            )

            # Recovery
            rho_rec = vcqec.cqec_recovery(rho_target, rho_noisy,
                                          cat_result['rho_cat'])
            fid_rec = fidelity(rho_target, rho_rec)
            dt = time.time() - t0

            result = {
                'fid_noisy': fid_noisy,
                'fid_recovered': fid_rec,
                'improvement': fid_rec - fid_noisy,
                'catalyst_coherence': cat_result['coherence'],
                'catalyst_rho_min': cat_result['rho_min'],
                'modes_covered': cat_result['modes_covered'],
                'n_params': cat_result['n_params'],
                'time_s': dt,
            }
            alg_results[noise_name] = result

            print(f"  F_recovered = {fid_rec:.6f} "
                  f"(+{fid_rec - fid_noisy:.6f})")
            print(f"  Catalyst: C_l1 = {cat_result['coherence']:.4f}, "
                  f"rho_min = {cat_result['rho_min']:.6f}, "
                  f"modes = {cat_result['modes_covered']:.1%}")
            print(f"  Time: {dt:.1f}s")

        all_results[alg_name] = alg_results

    return all_results


def print_summary_table(results):
    """Print formatted summary table."""
    print(f"\n\n{'=' * 100}")
    print(f"{'Algorithm':<12} {'Noise':<15} {'F_noisy':>10} {'F_recov':>10} "
          f"{'Improve':>10} {'C_l1(cat)':>10} {'Modes':>8} {'N_θ':>6} {'Time':>6}")
    print(f"{'-' * 100}")

    for alg_name, alg_res in results.items():
        for noise_name, r in alg_res.items():
            print(f"{alg_name:<12} {noise_name:<15} "
                  f"{r['fid_noisy']:>10.6f} "
                  f"{r['fid_recovered']:>10.6f} "
                  f"{r['improvement']:>+10.6f} "
                  f"{r['catalyst_coherence']:>10.4f} "
                  f"{r['modes_covered']:>8.1%} "
                  f"{r['n_params']:>6d} "
                  f"{r['time_s']:>5.1f}s")
    print(f"{'=' * 100}")


def plot_comparison(results):
    """Plot comparison: variational vs paper asymptotic results."""
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    alg_names = list(results.keys())
    noise_names = list(results[alg_names[0]].keys())

    # Paper asymptotic results (from Table I)
    paper_results = {
        'QKAN': {'Dephasing': 1.0000, 'Depolarizing': 1.0000, 'Combined': 1.0000},
        'qDRIFT': {'Dephasing': 0.9999, 'Depolarizing': 0.9997, 'Combined': 0.9999},
        'CF-QPE': {'Dephasing': 1.0000, 'Depolarizing': 1.0000, 'Combined': 1.0000},
        'Regev': {'Dephasing': 0.9998, 'Depolarizing': 0.9996, 'Combined': 0.9997},
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    colors_alg = {'QKAN': '#3498db', 'qDRIFT': '#e74c3c',
                  'CF-QPE': '#2ecc71', 'Regev': '#9b59b6'}

    for ni, noise_name in enumerate(noise_names):
        ax = axes[ni]
        x = np.arange(len(alg_names))
        width = 0.25

        # Before recovery
        fid_before = [results[a][noise_name]['fid_noisy'] for a in alg_names]
        # Variational recovery
        fid_var = [results[a][noise_name]['fid_recovered'] for a in alg_names]
        # Paper asymptotic
        fid_paper = [paper_results[a][noise_name] for a in alg_names]

        bars1 = ax.bar(x - width, fid_before, width,
                       label=r'Before recovery',
                       color='#e74c3c', alpha=0.6)
        bars2 = ax.bar(x, fid_var, width,
                       label=r'Variational CQEC',
                       color='#3498db', alpha=0.8)
        bars3 = ax.bar(x + width, fid_paper, width,
                       label=r'Asymptotic CQEC (paper)',
                       color='#2ecc71', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(alg_names, rotation=15, fontsize=9)
        ax.set_ylabel(r'Fidelity $F(\rho_{\mathrm{rec}},\,\rho_0)$')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

        noise_display = {
            'Dephasing': r'Dephasing ($\gamma=2.0$)',
            'Depolarizing': r'Depolarizing ($p=0.3$)',
            'Combined': 'Combined',
        }
        ax.set_title(noise_display[noise_name], fontsize=11)
        ax.legend(fontsize=7, loc='lower left')

        # Add value labels on bars
        for bar_group in [bars2]:
            for bar in bar_group:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle(r'Variational Catalyst CQEC vs. Asymptotic CQEC',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_benchmark_variational.png')
    plt.savefig('fig_benchmark_variational.pdf')
    print("\nSaved fig_benchmark_variational.png/pdf")
    plt.close()

    # --- Panel 2: Parameter scaling ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dims = [4, 8, 16, 16]
    n_params = [results[a][noise_names[0]]['n_params'] for a in alg_names]
    times = [max(results[a][n]['time_s'] for n in noise_names) for a in alg_names]

    ax1.semilogy(dims, n_params, 'o-', color='#3498db', linewidth=2,
                 markersize=8, label=r'$N_\theta$ (variational)')
    # Distillation copies for comparison
    Cs = [8.5, 42, 170, 170]  # Regev at d=16
    n_star = [c**2 / (4 * 0.01**2) for c in Cs]
    ax1.semilogy(dims, n_star, 's--', color='#e74c3c', linewidth=1.5,
                 markersize=6, label=r'Distillation $n^*$ ($F \geq 0.99$)')

    ax1.set_xlabel(r'Hilbert space dimension $d$')
    ax1.set_ylabel(r'Resource count')
    ax1.set_title(r'(a) Variational params $N_\theta$ vs. distillation $n^*$')
    ax1.legend(fontsize=9)

    # Recovery fidelity vs dimension (dephasing)
    fid_var_deph = [results[a]['Dephasing']['fid_recovered'] for a in alg_names]
    fid_noisy_deph = [results[a]['Dephasing']['fid_noisy'] for a in alg_names]

    ax2.plot(dims, fid_noisy_deph, 's--', color='#e74c3c', linewidth=1.5,
             markersize=6, label=r'Before recovery')
    ax2.plot(dims, fid_var_deph, 'o-', color='#3498db', linewidth=2,
             markersize=8, label=r'Variational CQEC')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax2.set_xlabel(r'Hilbert space dimension $d$')
    ax2.set_ylabel(r'Fidelity $F$')
    ax2.set_title(r'(b) Recovery fidelity vs. dimension (dephasing $\gamma=2.0$)')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)

    plt.suptitle('Variational Catalyst Preparation: Scaling Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_benchmark_scaling.png')
    plt.savefig('fig_benchmark_scaling.pdf')
    print("Saved fig_benchmark_scaling.png/pdf")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    results = run_benchmarks()
    print_summary_table(results)
    plot_comparison(results)

#!/usr/bin/env python3
"""
benchmark_blind_cqec.py - Blind Catalytic QEC: Target-Unknown Fidelity Analysis
================================================================================

When the target state is UNKNOWN, ICEC must estimate it from the noisy state
before correction. This benchmark quantifies the fidelity penalty of blind
estimation under various strategies and noise regimes.

Estimation strategies:
  1. Naive: use noisy state directly as target (no estimation)
  2. Noise-inversion: invert known noise channel analytically
  3. Coherence-maximization: maximize coherence within mode structure
  4. Iterative refinement: run ICEC multiple rounds, refining estimate
  5. Multi-copy tomography: estimate from multiple noisy copies

For each strategy, we sweep noise strength and measure:
  - Fidelity F(recovered, true_target)
  - Trace distance D(recovered, true_target)
  - Coherence recovery ratio
"""

import numpy as np
from scipy.linalg import expm, sqrtm, logm
from typing import Callable, Optional
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, '/Users/deeptell01/Documents/alterego/personal/cqec')
from core import (
    EnergySystem, partial_trace, tensor, trace_distance,
    coherence_l1, quantum_fisher_information,
    modes_of_asymmetry, check_mode_inclusion,
    partial_dephasing, amplitude_damping, depolarizing_channel,
    maximally_coherent_state, random_coherent_state,
    is_valid_density_matrix, is_full_rank,
)
from icec import CatalystState, CoherenceAmplifier, ICECProtocol


# ============================================================
# Fidelity measure
# ============================================================

def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    sqrt_rho = sqrtm(rho)
    sqrt_rho = np.array(sqrt_rho, dtype=complex)
    product = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(product)
    eigvals = np.maximum(eigvals.real, 0)
    fidelity = (np.sum(np.sqrt(eigvals)))**2
    return min(float(fidelity.real), 1.0)


def ensure_valid_dm(rho: np.ndarray) -> np.ndarray:
    """Project onto valid density matrix."""
    rho = 0.5 * (rho + rho.conj().T)  # Hermiticity
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    rho = rho / np.trace(rho)
    return rho


# ============================================================
# Target estimation strategies (blind - no knowledge of target)
# ============================================================

def estimate_naive(noisy: np.ndarray, energies: np.ndarray,
                   **kwargs) -> np.ndarray:
    """Strategy 1: Use noisy state directly as target estimate.
    No correction attempted on the estimate itself."""
    return noisy.copy()


def estimate_noise_inversion(noisy: np.ndarray, energies: np.ndarray,
                             noise_channel: Optional[Callable] = None,
                             noise_param: float = 0.0,
                             noise_type: str = 'dephasing',
                             **kwargs) -> np.ndarray:
    """Strategy 2: Invert the known noise channel analytically.

    Requires knowledge of the noise model (but NOT the target state).
    - Dephasing: rho_ij * exp(-gamma*|Ei-Ej|) -> invert by dividing
    - Depolarizing: (1-p)*rho + p*I/d -> invert by (rho - p*I/d)/(1-p)
    - Amplitude damping: approximate inversion via pseudoinverse
    """
    d = noisy.shape[0]

    if noise_type == 'dephasing':
        gamma = noise_param
        est = noisy.copy()
        for i in range(d):
            for j in range(d):
                delta = abs(energies[i] - energies[j])
                if delta > 1e-12:
                    decay = np.exp(-gamma * delta)
                    if decay > 1e-10:
                        est[i, j] /= decay
                    else:
                        # Mode too decayed - cannot invert reliably
                        est[i, j] = 0
        return ensure_valid_dm(est)

    elif noise_type == 'depolarizing':
        p = noise_param
        if p >= 1.0:
            return noisy.copy()
        est = (noisy - p * np.eye(d) / d) / (1 - p)
        return ensure_valid_dm(est)

    elif noise_type == 'amplitude_damping':
        gamma = noise_param
        # Approximate inversion for qubit
        est = noisy.copy()
        est[1, 1] = noisy[1, 1] / (1 - gamma) if gamma < 1 else 0
        est[0, 0] = 1 - est[1, 1]
        sqrt_factor = np.sqrt(1 - gamma) if gamma < 1 else 0
        if sqrt_factor > 1e-10:
            est[0, 1] = noisy[0, 1] / sqrt_factor
            est[1, 0] = noisy[1, 0] / sqrt_factor
        return ensure_valid_dm(est)

    return noisy.copy()


def estimate_coherence_max(noisy: np.ndarray, energies: np.ndarray,
                           **kwargs) -> np.ndarray:
    """Strategy 3: Maximize coherence within the mode structure.

    Keeps the population (diagonal) of the noisy state but maximizes
    off-diagonal coherence to the physicality bound sqrt(pi*pj).
    This is the 'most coherent state consistent with the diagonal'.
    """
    d = noisy.shape[0]
    est = np.array(noisy, dtype=complex)
    for i in range(d):
        for j in range(d):
            if i != j:
                # Maximize coherence: |rho_ij| <= sqrt(rho_ii * rho_jj)
                max_coh = np.sqrt(max(noisy[i, i].real, 0) *
                                  max(noisy[j, j].real, 0))
                if abs(noisy[i, j]) > 1e-15:
                    phase = np.angle(noisy[i, j])
                    est[i, j] = max_coh * np.exp(1j * phase)
                else:
                    # Mode lost: assume positive real coherence (best guess)
                    est[i, j] = complex(max_coh * 0.5)
    return ensure_valid_dm(est)


def estimate_iterative(noisy: np.ndarray, energies: np.ndarray,
                       n_iterations: int = 5, n_copies: int = 50,
                       **kwargs) -> np.ndarray:
    """Strategy 4: Iterative refinement.

    1. Start with coherence-maximized estimate
    2. Run ICEC with current estimate as target
    3. Use recovered state as new estimate
    4. Repeat

    Each iteration should improve the estimate if modes are preserved.
    """
    system = EnergySystem(energies)
    current_est = estimate_coherence_max(noisy, energies)

    for it in range(n_iterations):
        # Run ICEC with current estimate as target
        protocol = ICECProtocol(energies, current_est)
        protocol.initialize(n_copies)
        recovered, result = protocol.correct(noisy, n_copies)

        if not result.mode_condition_satisfied:
            break

        # Use recovered state to refine: average with current estimate
        # (damped update to avoid oscillation)
        alpha = 0.5
        new_est = alpha * recovered + (1 - alpha) * current_est
        current_est = ensure_valid_dm(new_est)

    return current_est


def estimate_multicopy(noisy: np.ndarray, energies: np.ndarray,
                       noise_channel: Optional[Callable] = None,
                       n_samples: int = 20,
                       **kwargs) -> np.ndarray:
    """Strategy 5: Multi-copy tomographic estimation.

    Simulate having n_samples independent noisy copies.
    Average them (reduces noise variance), then boost coherence.
    """
    d = noisy.shape[0]
    if noise_channel is None:
        return noisy.copy()

    # Generate a 'ground truth'-ish estimate by averaging many noisy copies
    # In practice: start from the noisy state and generate more samples
    # by applying the noise channel to a random initial guess
    # Here we use the noisy state and add statistical fluctuations
    avg = np.zeros_like(noisy, dtype=complex)
    for _ in range(n_samples):
        # Simulate measurement noise: small perturbation of the noisy state
        perturbation = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) * 0.01
        perturbation = 0.5 * (perturbation + perturbation.conj().T)
        sample = ensure_valid_dm(noisy + perturbation)
        avg += sample
    avg /= n_samples

    # Boost coherence (undo some of the noise averaging)
    est = estimate_coherence_max(avg, energies)
    return est


# ============================================================
# Run blind CQEC benchmark
# ============================================================

def run_blind_benchmark(
    target_state: np.ndarray,
    energies: np.ndarray,
    noise_type: str,
    noise_params: np.ndarray,
    n_copies: int = 100,
    label: str = ''
) -> dict:
    """Run blind CQEC benchmark for one noise type across noise strengths.

    Returns dict with fidelities for each strategy.
    """
    strategies = {
        'Naive (no estimation)': estimate_naive,
        'Noise-channel inversion': estimate_noise_inversion,
        'Coherence maximization': estimate_coherence_max,
        'Iterative refinement': estimate_iterative,
        'Multi-copy tomography': estimate_multicopy,
        'Oracle (known target)': None,  # reference: perfect knowledge
    }

    results = {name: {'fidelity': [], 'trace_dist': [], 'coh_ratio': []}
               for name in strategies}
    results['No correction'] = {'fidelity': [], 'trace_dist': [], 'coh_ratio': []}

    target_coh = coherence_l1(target_state, energies)

    for param in noise_params:
        # Apply noise
        if noise_type == 'dephasing':
            noise_fn = lambda rho, g=param: partial_dephasing(rho, energies, g)
        elif noise_type == 'depolarizing':
            noise_fn = lambda rho, p=param: depolarizing_channel(rho, p)
        elif noise_type == 'amplitude_damping':
            noise_fn = lambda rho, g=param: amplitude_damping(rho, g)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        noisy = noise_fn(target_state)

        # No correction baseline
        f_noisy = state_fidelity(noisy, target_state)
        d_noisy = trace_distance(noisy, target_state)
        c_noisy = coherence_l1(noisy, energies) / max(target_coh, 1e-15)
        results['No correction']['fidelity'].append(f_noisy)
        results['No correction']['trace_dist'].append(d_noisy)
        results['No correction']['coh_ratio'].append(c_noisy)

        for name, estimator in strategies.items():
            if name == 'Oracle (known target)':
                # Perfect knowledge
                estimated_target = target_state.copy()
            else:
                estimated_target = estimator(
                    noisy, energies,
                    noise_channel=noise_fn,
                    noise_param=param,
                    noise_type=noise_type,
                    n_copies=n_copies,
                )

            # Run ICEC with estimated target
            protocol = ICECProtocol(energies, estimated_target)
            protocol.initialize(n_copies)
            recovered, result = protocol.correct(noisy, n_copies)

            # Measure against TRUE target
            f = state_fidelity(recovered, target_state)
            d = trace_distance(recovered, target_state)
            c = coherence_l1(recovered, energies) / max(target_coh, 1e-15)

            results[name]['fidelity'].append(f)
            results[name]['trace_dist'].append(d)
            results[name]['coh_ratio'].append(c)

    return results


# ============================================================
# Higher-dimensional test
# ============================================================

def run_qutrit_benchmark(noise_params: np.ndarray, n_copies: int = 100):
    """Run on a 3-level system (qutrit) for generality."""
    energies = np.array([0.0, 1.0, 2.0])
    # Non-trivial qutrit target state
    psi = np.array([1, np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 3)]) / np.sqrt(3)
    target = np.outer(psi, psi.conj())

    return run_blind_benchmark(
        target, energies, 'dephasing', noise_params,
        n_copies=n_copies, label='Qutrit'
    )


# ============================================================
# Plotting
# ============================================================

def plot_results(results: dict, noise_params: np.ndarray,
                 noise_label: str, title: str, filename: str):
    """Plot fidelity vs noise strength for all strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {
        'No correction': '#999999',
        'Naive (no estimation)': '#e74c3c',
        'Noise-channel inversion': '#3498db',
        'Coherence maximization': '#2ecc71',
        'Iterative refinement': '#9b59b6',
        'Multi-copy tomography': '#e67e22',
        'Oracle (known target)': '#1a1a1a',
    }
    linestyles = {
        'No correction': ':',
        'Oracle (known target)': '--',
    }

    # Panel 1: Fidelity
    ax = axes[0]
    for name, data in results.items():
        ls = linestyles.get(name, '-')
        lw = 2.5 if name in ('Oracle (known target)', 'No correction') else 1.8
        ax.plot(noise_params, data['fidelity'], label=name,
                color=colors.get(name, '#333'), linestyle=ls, linewidth=lw,
                marker='o', markersize=3)
    ax.set_xlabel(noise_label, fontsize=12)
    ax.set_ylabel('Fidelity with true target', fontsize=12)
    ax.set_title('Recovery Fidelity', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    # Panel 2: Trace distance
    ax = axes[1]
    for name, data in results.items():
        ls = linestyles.get(name, '-')
        lw = 2.5 if name in ('Oracle (known target)', 'No correction') else 1.8
        ax.plot(noise_params, data['trace_dist'], label=name,
                color=colors.get(name, '#333'), linestyle=ls, linewidth=lw,
                marker='o', markersize=3)
    ax.set_xlabel(noise_label, fontsize=12)
    ax.set_ylabel('Trace distance to true target', fontsize=12)
    ax.set_title('Trace Distance (lower is better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    # Panel 3: Coherence recovery ratio
    ax = axes[2]
    for name, data in results.items():
        ls = linestyles.get(name, '-')
        lw = 2.5 if name in ('Oracle (known target)', 'No correction') else 1.8
        ax.plot(noise_params, data['coh_ratio'], label=name,
                color=colors.get(name, '#333'), linestyle=ls, linewidth=lw,
                marker='o', markersize=3)
    ax.set_xlabel(noise_label, fontsize=12)
    ax.set_ylabel('Coherence ratio (recovered / target)', fontsize=12)
    ax.set_title('Coherence Recovery', fontsize=13, fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.set_ylim(-0.02, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_fidelity_gap(all_results: dict, noise_params: np.ndarray,
                      filename: str):
    """Plot the fidelity GAP between oracle and each blind strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Naive (no estimation)': '#e74c3c',
        'Noise-channel inversion': '#3498db',
        'Coherence maximization': '#2ecc71',
        'Iterative refinement': '#9b59b6',
        'Multi-copy tomography': '#e67e22',
    }

    for noise_type, results in all_results.items():
        oracle_f = np.array(results['Oracle (known target)']['fidelity'])
        for name in colors:
            if name in results:
                blind_f = np.array(results[name]['fidelity'])
                gap = oracle_f - blind_f
                ax.plot(noise_params, gap, label=f'{name} ({noise_type})',
                        color=colors[name],
                        linestyle='-' if noise_type == 'dephasing' else '--',
                        linewidth=1.5, marker='s', markersize=3)

    ax.set_xlabel('Noise strength', fontsize=12)
    ax.set_ylabel('Fidelity gap (Oracle − Blind)', fontsize=12)
    ax.set_title('Fidelity Penalty from Unknown Target State',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    out_dir = '/Users/deeptell01/Documents/alterego/personal/cqec'

    print("=" * 70)
    print(" Blind CQEC Benchmark: Target-Unknown Fidelity Analysis")
    print("=" * 70)

    # --- Qubit tests ---
    energies_qubit = np.array([0.0, 1.0])
    target_qubit = maximally_coherent_state(2)

    n_copies = 100
    all_results = {}

    # 1. Dephasing sweep
    print("\n[1/4] Dephasing noise sweep (qubit)...")
    deph_params = np.linspace(0.1, 5.0, 20)
    res_deph = run_blind_benchmark(
        target_qubit, energies_qubit, 'dephasing', deph_params, n_copies)
    all_results['dephasing'] = res_deph
    plot_results(res_deph, deph_params, 'Dephasing γ',
                 'Blind CQEC: Dephasing Noise (Qubit)',
                 os.path.join(out_dir, 'fig_blind_dephasing.png'))

    # 2. Depolarizing sweep
    print("\n[2/4] Depolarizing noise sweep (qubit)...")
    depol_params = np.linspace(0.05, 0.95, 20)
    res_depol = run_blind_benchmark(
        target_qubit, energies_qubit, 'depolarizing', depol_params, n_copies)
    all_results['depolarizing'] = res_depol
    plot_results(res_depol, depol_params, 'Depolarizing p',
                 'Blind CQEC: Depolarizing Noise (Qubit)',
                 os.path.join(out_dir, 'fig_blind_depolarizing.png'))

    # 3. Amplitude damping sweep
    print("\n[3/4] Amplitude damping sweep (qubit)...")
    amp_params = np.linspace(0.05, 0.95, 20)
    res_amp = run_blind_benchmark(
        target_qubit, energies_qubit, 'amplitude_damping', amp_params, n_copies)
    all_results['amplitude_damping'] = res_amp
    plot_results(res_amp, amp_params, 'Damping γ',
                 'Blind CQEC: Amplitude Damping (Qubit)',
                 os.path.join(out_dir, 'fig_blind_amplitude_damping.png'))

    # 4. Qutrit dephasing
    print("\n[4/4] Dephasing noise sweep (qutrit)...")
    qutrit_params = np.linspace(0.1, 4.0, 16)
    res_qutrit = run_qutrit_benchmark(qutrit_params, n_copies)
    plot_results(res_qutrit, qutrit_params, 'Dephasing γ',
                 'Blind CQEC: Dephasing Noise (Qutrit, d=3)',
                 os.path.join(out_dir, 'fig_blind_qutrit.png'))

    # Fidelity gap summary
    print("\n[Summary] Fidelity gap plot...")
    plot_fidelity_gap(all_results, deph_params,
                      os.path.join(out_dir, 'fig_blind_fidelity_gap.png'))

    # --- Print numerical summary ---
    print("\n" + "=" * 70)
    print(" Numerical Summary (Dephasing, γ=2.0)")
    print("=" * 70)
    idx_gamma2 = np.argmin(np.abs(deph_params - 2.0))
    print(f"  {'Strategy':<30s}  {'Fidelity':>8s}  {'Trace Dist':>10s}  {'Coh Ratio':>10s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*10}")
    for name, data in res_deph.items():
        f = data['fidelity'][idx_gamma2]
        d = data['trace_dist'][idx_gamma2]
        c = data['coh_ratio'][idx_gamma2]
        print(f"  {name:<30s}  {f:8.4f}  {d:10.4f}  {c:10.4f}")

    print(f"\n  (Depolarizing, p=0.5)")
    idx_p05 = np.argmin(np.abs(depol_params - 0.5))
    for name, data in res_depol.items():
        f = data['fidelity'][idx_p05]
        d = data['trace_dist'][idx_p05]
        c = data['coh_ratio'][idx_p05]
        print(f"  {name:<30s}  {f:8.4f}  {d:10.4f}  {c:10.4f}")

    print(f"\n  (Amplitude Damping, γ=0.5)")
    idx_ad05 = np.argmin(np.abs(amp_params - 0.5))
    for name, data in res_amp.items():
        f = data['fidelity'][idx_ad05]
        d = data['trace_dist'][idx_ad05]
        c = data['coh_ratio'][idx_ad05]
        print(f"  {name:<30s}  {f:8.4f}  {d:10.4f}  {c:10.4f}")

    print("\n[Done] All blind CQEC benchmarks complete.")

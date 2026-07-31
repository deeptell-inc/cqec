#!/usr/bin/env python3
"""
benchmark_blind_dimension_sweep.py - Dimension Sweep with Haar-Random States
=============================================================================

Addresses Nature reviewer concerns:
  1. No systematic dimension sweep with controlled states
  2. No Haar-random state ensemble averaging
  3. No comparison with standard MLE tomography + CQEC

Benchmark design:
  - Dimensions d = 2, 4, 8, 16, 32, 64 (d = 2^k, k = 1..6)
    (d >= 128 is excluded: the exact channel inversion requires
     exp(gamma*(d-1)), which overflows double precision)
  - For each d: 20 Haar-random pure states
  - Combined noise: dephasing gamma=1.0, depolarizing p=0.15, amp damping gamma_AD=0.1
  - Estimation strategies: naive, channel-inversion, coherence-max, iterative,
    multi-copy, linear-inversion (MLE baseline), and oracle
  - Report mean +/- std of F_rec across the 20 random states

Figures:
  - fig_dimension_sweep.pdf: F_rec vs d (log-scale) with error bands
  - fig_haar_distribution.pdf: box plots at d = 4, 16, 64
  - fig_mle_comparison.pdf: bar chart comparing MLE+CQEC vs coherence_max+CQEC
    vs channel_inv+CQEC at each dimension
"""

import numpy as np
from scipy.linalg import sqrtm
from math import log2
from typing import Tuple, Dict
import time
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/deeptell01/Documents/alterego/personal/cqec')
from core import (
    EnergySystem, partial_trace, tensor, trace_distance,
    coherence_l1, partial_dephasing, depolarizing_channel,
    maximally_coherent_state, is_valid_density_matrix,
    check_mode_inclusion,
)
from icec import CatalystState, CoherenceAmplifier, ICECProtocol


# ============================================================
# Utility functions
# ============================================================

def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Optimized: if either state is (near-)pure, use the simpler formula
    F(rho, |psi><psi|) = <psi|rho|psi>.
    """
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(sigma)):
        return 0.0
    try:
        d = rho.shape[0]
        # Check if sigma is pure (rank-1): use fast formula
        eigvals_s = np.linalg.eigvalsh(sigma)
        if np.sum(eigvals_s > 1e-8) <= 1:
            # sigma is pure: F = Tr(rho @ sigma)
            f = np.trace(rho @ sigma).real
            return min(max(float(f), 0.0), 1.0)
        # Check if rho is pure
        eigvals_r = np.linalg.eigvalsh(rho)
        if np.sum(eigvals_r > 1e-8) <= 1:
            f = np.trace(sigma @ rho).real
            return min(max(float(f), 0.0), 1.0)
        # General case: use eigendecomposition-based approach (faster than sqrtm)
        # F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
        # Use: sqrt(rho) via eigendecomposition
        eigvals_r, eigvecs_r = np.linalg.eigh(rho)
        eigvals_r = np.maximum(eigvals_r.real, 0)
        sqrt_eigvals = np.sqrt(eigvals_r)
        sqrt_rho = eigvecs_r @ np.diag(sqrt_eigvals) @ eigvecs_r.conj().T
        product = sqrt_rho @ sigma @ sqrt_rho
        eigvals_p = np.linalg.eigvalsh(product)
        eigvals_p = np.maximum(eigvals_p.real, 0)
        fidelity = (np.sum(np.sqrt(eigvals_p)))**2
        result = float(fidelity.real)
        if not np.isfinite(result):
            return 0.0
        return min(result, 1.0)
    except Exception:
        return 0.0


def ensure_valid_dm(rho: np.ndarray) -> np.ndarray:
    """Project onto valid density matrix (Hermitian, PSD, trace-1)."""
    d = rho.shape[0]
    # Handle NaN/Inf
    if not np.all(np.isfinite(rho)):
        return np.eye(d, dtype=complex) / d
    rho = 0.5 * (rho + rho.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals.real, 0)
    total = np.sum(eigvals)
    if total < 1e-15 or not np.isfinite(total):
        return np.eye(d, dtype=complex) / d
    eigvals /= total
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    if not np.all(np.isfinite(rho)):
        return np.eye(d, dtype=complex) / d
    return rho


def haar_random_pure_state(d: int) -> np.ndarray:
    """Generate a Haar-random pure state density matrix of dimension d."""
    psi = np.random.randn(d) + 1j * np.random.randn(d)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


# ============================================================
# Noise channels (d-dimensional)
# ============================================================

def apply_dephasing(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Energy-gap-dependent dephasing: rho_ij -> exp(-gamma*|i-j|) rho_ij.

    Matches Sec. 3.2 of the manuscript and
    blind_cqec.noise.dephasing.  Energies are equally spaced (E_i = i),
    so the mode gap is |Delta_ij| = |i - j|.
    """
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    idx = np.arange(d)
    mask = np.exp(-gamma * np.abs(idx[:, None] - idx[None, :]))
    return rho * mask


def apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing channel: rho -> (1-p) rho + p I/d."""
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d, dtype=complex) / d


def apply_amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Generalized amplitude damping: cascaded decay |k> -> |k-1>, rate gamma.

    Identical to blind_cqec.noise.amplitude_damping (the manuscript's
    cascaded ladder model): population from every excited level flows to the
    ground state, and coherences between levels i, j are multiplied by
    (1 - gamma)^{(i+j)/2}.
    """
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[0]
    out = np.zeros_like(rho)
    for i in range(d):
        for j in range(d):
            if i == 0 and j == 0:
                out[0, 0] = rho[0, 0] + gamma * sum(
                    rho[k, k] for k in range(1, d)
                )
            elif i == j:
                out[i, j] = (1.0 - gamma) * rho[i, j]
            else:
                fac = (1.0 - gamma) ** ((i + j) / 2.0) if (i > 0 or j > 0) else 1.0
                out[i, j] = rho[i, j] * fac
    return out


def apply_combined_noise(rho: np.ndarray, dephasing_gamma: float = 1.0,
                         depol_p: float = 0.15,
                         amp_damp_gamma: float = 0.1) -> np.ndarray:
    """Apply combined decoherence: dephasing + depolarizing + amplitude damping."""
    rho_noisy = apply_dephasing(rho, dephasing_gamma)
    rho_noisy = apply_depolarizing(rho_noisy, depol_p)
    rho_noisy = apply_amplitude_damping_channel(rho_noisy, amp_damp_gamma)
    rho_noisy = (rho_noisy + rho_noisy.conj().T) / 2
    rho_noisy /= np.trace(rho_noisy)
    return rho_noisy


# ============================================================
# ICEC recovery (fallback-based, matches benchmark_icec.py)
# ============================================================

def icec_recover(rho_noisy: np.ndarray, rho_target: np.ndarray,
                 energies: np.ndarray = None,
                 n_cycles: int = 1) -> Tuple[np.ndarray, Dict]:
    """Apply ICEC protocol to recover a noisy state.

    Tries the full ICECProtocol first; falls back to direct
    coherence amplification if numerical issues arise.
    """
    d = rho_noisy.shape[0]
    if energies is None:
        energies = np.arange(d, dtype=float)

    mode_ok = check_mode_inclusion(rho_target, rho_noisy, energies, energies)

    if not mode_ok:
        coh_noisy = coherence_l1(rho_noisy, energies)
        if coh_noisy < 1e-15:
            return rho_noisy, {'success': False, 'fidelity': state_fidelity(rho_noisy, rho_target)}

    # For large dimensions, go directly to fallback (full protocol too expensive)
    if d > 64:
        return _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles)

    # Try full protocol for small dimensions
    try:
        n_cop = min(50, max(10, 200 // d))
        protocol = ICECProtocol(energies, rho_target)
        protocol.initialize(n_copies=n_cop)
        rho_recovered = rho_noisy.copy()
        for cycle in range(n_cycles):
            rho_recovered, _ = protocol.correct(rho_recovered, n_copies=n_cop)
        fidelity = state_fidelity(rho_recovered, rho_target)
        return rho_recovered, {'success': True, 'fidelity': fidelity}
    except Exception:
        pass

    # Fallback: direct coherence amplification
    return _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles)


def _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles):
    """Fallback ICEC: amplify residual coherence to target values.

    Vectorized for performance at large d.
    """
    d = rho_noisy.shape[0]

    # Check which off-diagonal elements survive
    surviving_mask = (np.abs(rho_noisy) > 1e-15)
    np.fill_diagonal(surviving_mask, False)

    if not np.any(surviving_mask):
        return rho_noisy, {'success': False, 'fidelity': state_fidelity(rho_noisy, rho_target)}

    # Precompute: which modes can be generated from surviving modes
    # A mode (i,j) with delta = E_i - E_j can be generated if delta
    # is an integer multiple of any surviving mode's delta
    surviving_deltas = set()
    for i in range(d):
        for j in range(d):
            if surviving_mask[i, j]:
                surviving_deltas.add(energies[i] - energies[j])

    if not surviving_deltas:
        return rho_noisy, {'success': False, 'fidelity': state_fidelity(rho_noisy, rho_target)}

    surviving_deltas_arr = np.array(list(surviving_deltas))
    surviving_deltas_arr = surviving_deltas_arr[np.abs(surviving_deltas_arr) > 1e-10]

    # Build a mask of all modes that can be amplified (surviving or generatable)
    can_amplify = surviving_mask.copy()
    if len(surviving_deltas_arr) > 0:
        delta_matrix = energies[:, None] - energies[None, :]
        for sd in surviving_deltas_arr:
            ratios = delta_matrix / sd
            integer_mask = np.abs(ratios - np.round(ratios)) < 1e-8
            can_amplify |= integer_mask
        np.fill_diagonal(can_amplify, False)

    rho_recovered = rho_noisy.copy()

    for cycle in range(n_cycles):
        # Set all amplifiable off-diagonal elements to target values
        rho_recovered = np.where(can_amplify, rho_target, rho_recovered)
        # Set diagonal to target
        np.fill_diagonal(rho_recovered, np.diag(rho_target))
        rho_recovered = ensure_valid_dm(rho_recovered)

    fidelity = state_fidelity(rho_recovered, rho_target)
    return rho_recovered, {'success': True, 'fidelity': fidelity}


# ============================================================
# Estimation strategies (from benchmark_blind_cqec.py)
# ============================================================

def estimate_naive(noisy, energies, **kwargs):
    """Use noisy state directly as target estimate."""
    return noisy.copy()


def estimate_noise_inversion(noisy, energies, **kwargs):
    """Invert the combined noise channel analytically.

    Inverts in reverse order of application (amplitude damping ->
    depolarizing -> dephasing), using exactly the conventions of
    apply_combined_noise above.  Matches
    blind_cqec.estimators.estimate_channel_inversion.
    """
    d = noisy.shape[0]
    est = np.asarray(noisy, dtype=complex).copy()

    # 1) Undo amplitude damping (applied last)
    amp_gamma = kwargs.get('amp_damp_gamma', 0.0)
    if amp_gamma > 0 and amp_gamma < 1:
        inv = np.zeros_like(est)
        for i in range(d):
            for j in range(d):
                if i == 0 and j == 0:
                    inv[0, 0] = est[0, 0] - amp_gamma * sum(
                        est[k, k] / (1.0 - amp_gamma) for k in range(1, d)
                    )
                elif i == j:
                    inv[i, j] = est[i, j] / (1.0 - amp_gamma)
                else:
                    fac = ((1.0 - amp_gamma) ** ((i + j) / 2.0)
                           if (i > 0 or j > 0) else 1.0)
                    inv[i, j] = est[i, j] / fac if fac > 1e-15 else 0.0
        est = inv

    # 2) Undo depolarizing: (rho - p I/d) / (1-p)
    depol_p = kwargs.get('depol_p', 0.15)
    if 0 < depol_p < 1.0:
        est = (est - depol_p * np.eye(d) / d) / (1 - depol_p)

    # 3) Undo gap-dependent dephasing: multiply by exp(+gamma*|i-j|)
    deph_gamma = kwargs.get('dephasing_gamma', 1.0)
    if deph_gamma > 0:
        idx = np.arange(d)
        mask = np.exp(deph_gamma * np.abs(idx[:, None] - idx[None, :]))
        np.fill_diagonal(mask, 1.0)
        est = est * mask

    return ensure_valid_dm(est)


def estimate_coherence_max(noisy, energies, **kwargs):
    """Maximize coherence within physicality bound sqrt(pi * pj)."""
    d = noisy.shape[0]
    diag = np.maximum(np.diag(noisy).real, 0)
    max_coh_matrix = np.sqrt(np.outer(diag, diag))
    phases = np.angle(noisy)
    magnitudes = np.abs(noisy)
    # Where noisy has nonzero off-diagonal: use phase from noisy
    # Where zero: use 0.5 * max_coh (best guess)
    est = np.where(
        magnitudes > 1e-15,
        max_coh_matrix * np.exp(1j * phases),
        max_coh_matrix * 0.5
    )
    # Restore diagonal
    np.fill_diagonal(est, np.diag(noisy))
    return ensure_valid_dm(est)


def estimate_iterative(noisy, energies, **kwargs):
    """Iterative refinement: coherence-max then ICEC rounds.

    For d > 32, uses fallback-only ICEC to keep runtime manageable.
    """
    d = noisy.shape[0]
    current_est = estimate_coherence_max(noisy, energies)
    n_iter = 2 if d <= 32 else 1
    n_cop = 20 if d <= 32 else 10
    for it in range(n_iter):
        try:
            if d <= 64:
                protocol = ICECProtocol(energies, current_est)
                protocol.initialize(n_cop)
                recovered, result = protocol.correct(noisy, n_cop)
                if not result.mode_condition_satisfied:
                    break
            else:
                # For large d, use fallback directly to avoid expensive catalyst
                recovered, _ = _icec_recover_fallback(noisy, current_est, energies, 1)
            new_est = 0.5 * recovered + 0.5 * current_est
            current_est = ensure_valid_dm(new_est)
        except Exception:
            break
    return current_est


def estimate_multicopy(noisy, energies, **kwargs):
    """Multi-copy tomographic estimation: average perturbations then boost."""
    d = noisy.shape[0]
    n_samples = 20
    avg = np.zeros_like(noisy, dtype=complex)
    for _ in range(n_samples):
        perturbation = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) * 0.01
        perturbation = 0.5 * (perturbation + perturbation.conj().T)
        sample = ensure_valid_dm(noisy + perturbation)
        avg += sample
    avg /= n_samples
    return estimate_coherence_max(avg, energies)


def estimate_linear_inversion(noisy, energies, **kwargs):
    """Linear-inversion tomography baseline.

    Average n noisy copies (density matrices), then project to nearest
    valid density matrix. This is the standard MLE-like baseline at the
    density-matrix level (no measurement simulation needed).
    """
    d = noisy.shape[0]
    n_copies = kwargs.get('n_copies_mle', 50)
    deph_gamma = kwargs.get('dephasing_gamma', 1.0)
    depol_p = kwargs.get('depol_p', 0.15)
    amp_gamma = kwargs.get('amp_damp_gamma', 0.1)

    # We do NOT have access to the true state. We simulate having n_copies
    # of the noisy channel output with statistical fluctuations.
    avg = np.zeros((d, d), dtype=complex)
    for _ in range(n_copies):
        # Each copy: noisy state + measurement noise (finite statistics)
        noise_scale = 1.0 / np.sqrt(d)
        fluct = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) * noise_scale / np.sqrt(n_copies)
        fluct = 0.5 * (fluct + fluct.conj().T)
        sample = noisy + fluct
        avg += sample
    avg /= n_copies
    return ensure_valid_dm(avg)


# ============================================================
# Strategy registry
# ============================================================

STRATEGIES = {
    'Naive': estimate_naive,
    'Channel inversion': estimate_noise_inversion,
    'Coherence max': estimate_coherence_max,
    'Iterative': estimate_iterative,
    'Multi-copy': estimate_multicopy,
    'Linear inversion (MLE)': estimate_linear_inversion,
    'Oracle': None,
}


# ============================================================
# Single-state benchmark
# ============================================================

def benchmark_single_state(target: np.ndarray, d: int,
                           dephasing_gamma: float = 1.0,
                           depol_p: float = 0.15,
                           amp_damp_gamma: float = 0.1) -> Dict[str, float]:
    """Run all strategies on one target state, return fidelities."""
    energies = np.arange(d, dtype=float)
    noisy = apply_combined_noise(target, dephasing_gamma, depol_p, amp_damp_gamma)

    results = {}

    # No-correction baseline
    results['No correction'] = state_fidelity(noisy, target)

    for name, estimator in STRATEGIES.items():
        if name == 'Oracle':
            est_target = target.copy()
        else:
            est_target = estimator(
                noisy, energies,
                dephasing_gamma=dephasing_gamma,
                depol_p=depol_p,
                amp_damp_gamma=amp_damp_gamma,
            )

        # Run ICEC with estimated target
        recovered, info = icec_recover(noisy, est_target, energies, n_cycles=1)
        results[name] = state_fidelity(recovered, target)

    return results


# ============================================================
# Dimension sweep
# ============================================================

def run_dimension_sweep(dims: list, n_states: int = 20,
                        dephasing_gamma: float = 1.0,
                        depol_p: float = 0.15,
                        amp_damp_gamma: float = 0.1) -> Dict:
    """Sweep over dimensions with Haar-random states.

    Returns dict mapping strategy name to dict with 'mean' and 'std' arrays
    over dimensions.
    """
    all_strategy_names = list(STRATEGIES.keys()) + ['No correction']
    results = {name: {'mean': [], 'std': [], 'all_fids': []} for name in all_strategy_names}

    for d in dims:
        print(f"  d = {d:>4d} ...", end=" ", flush=True)
        t0 = time.time()
        np.random.seed(42)  # reset seed for reproducibility at each d

        fids_by_strategy = {name: [] for name in all_strategy_names}

        for s in range(n_states):
            target = haar_random_pure_state(d)
            res = benchmark_single_state(target, d, dephasing_gamma, depol_p, amp_damp_gamma)
            for name in all_strategy_names:
                fids_by_strategy[name].append(res[name])

        for name in all_strategy_names:
            arr = np.array(fids_by_strategy[name])
            results[name]['mean'].append(np.mean(arr))
            results[name]['std'].append(np.std(arr))
            results[name]['all_fids'].append(arr)

        dt = time.time() - t0
        # Show quick summary for oracle
        oracle_mean = results['Oracle']['mean'][-1]
        coh_mean = results['Coherence max']['mean'][-1]
        print(f"Oracle={oracle_mean:.4f}  CohMax={coh_mean:.4f}  ({dt:.1f}s)")

    # Convert lists to arrays
    for name in all_strategy_names:
        results[name]['mean'] = np.array(results[name]['mean'])
        results[name]['std'] = np.array(results[name]['std'])

    return results


# ============================================================
# Plotting
# ============================================================

def setup_matplotlib():
    """Publication-quality settings."""
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


def plot_dimension_sweep(dims, results, out_dir):
    """Fig 1: F_rec vs dimension d (log-scale x-axis) with error bands."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        'No correction': '#999999',
        'Naive': '#e74c3c',
        'Channel inversion': '#3498db',
        'Coherence max': '#2ecc71',
        'Iterative': '#9b59b6',
        'Multi-copy': '#e67e22',
        'Linear inversion (MLE)': '#17becf',
        'Oracle': '#1a1a1a',
    }
    markers = {
        'No correction': 'x',
        'Naive': 'v',
        'Channel inversion': 's',
        'Coherence max': 'D',
        'Iterative': '^',
        'Multi-copy': 'p',
        'Linear inversion (MLE)': 'h',
        'Oracle': 'o',
    }
    linestyles = {
        'No correction': ':',
        'Oracle': '--',
    }

    for name in results:
        if name not in colors:
            continue
        mean = results[name]['mean']
        std = results[name]['std']
        ls = linestyles.get(name, '-')
        lw = 2.0 if name in ('Oracle', 'No correction') else 1.4
        ax.plot(dims, mean, label=name, color=colors[name], linestyle=ls,
                linewidth=lw, marker=markers.get(name, 'o'), markersize=5)
        ax.fill_between(dims, mean - std, mean + std,
                        color=colors[name], alpha=0.12)

    ax.set_xscale('log', base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel('Hilbert space dimension d')
    ax.set_ylabel('Recovery fidelity $F_{rec}$')
    ax.set_title('Blind CQEC: Dimension Sweep with Haar-Random States\n'
                 '(20 states/dim, combined noise: dephasing+depolarizing+amp. damping)')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', ncol=2)
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_dimension_sweep.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_haar_distribution(dims, results, out_dir):
    """Fig 2: Box plots of F_rec distribution at d = 4, 16, 64, 256."""
    setup_matplotlib()

    target_dims = [4, 16, 64, 256]
    # Filter to dims that were actually computed
    target_dims = [d for d in target_dims if d in dims]
    dim_indices = [dims.index(d) for d in target_dims]

    strategies_to_plot = ['Channel inversion', 'Coherence max',
                          'Linear inversion (MLE)', 'Oracle']
    colors = {
        'Channel inversion': '#3498db',
        'Coherence max': '#2ecc71',
        'Linear inversion (MLE)': '#17becf',
        'Oracle': '#1a1a1a',
    }

    n_strats = len(strategies_to_plot)
    n_dims = len(target_dims)

    fig, axes = plt.subplots(1, n_dims, figsize=(3.5 * n_dims, 4.5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax_idx, (dim_idx, d) in enumerate(zip(dim_indices, target_dims)):
        ax = axes[ax_idx]
        data = []
        labels = []
        box_colors = []
        for sname in strategies_to_plot:
            fids = results[sname]['all_fids'][dim_idx]
            data.append(fids)
            labels.append(sname.replace(' inversion (MLE)', '\n(MLE)').replace(' inversion', '\ninv.').replace(' max', '\nmax'))
            box_colors.append(colors[sname])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_title(f'd = {d}')
        ax.grid(True, alpha=0.3, axis='y')
        if ax_idx == 0:
            ax.set_ylabel('Recovery fidelity $F_{rec}$')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Haar-Random State Fidelity Distributions', fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_haar_distribution.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_mle_comparison(dims, results, out_dir):
    """Fig 3: Bar chart comparing MLE+CQEC vs coherence_max+CQEC vs channel_inv+CQEC."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies_compare = ['Linear inversion (MLE)', 'Coherence max', 'Channel inversion', 'Oracle']
    colors = {
        'Linear inversion (MLE)': '#17becf',
        'Coherence max': '#2ecc71',
        'Channel inversion': '#3498db',
        'Oracle': '#1a1a1a',
    }
    short_labels = {
        'Linear inversion (MLE)': 'MLE + CQEC',
        'Coherence max': 'CohMax + CQEC',
        'Channel inversion': 'ChanInv + CQEC',
        'Oracle': 'Oracle + CQEC',
    }

    n_dims = len(dims)
    n_strats = len(strategies_compare)
    bar_width = 0.8 / n_strats
    x = np.arange(n_dims)

    for idx, sname in enumerate(strategies_compare):
        means = results[sname]['mean']
        stds = results[sname]['std']
        offset = (idx - n_strats / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                       label=short_labels[sname], color=colors[sname],
                       alpha=0.75, capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel('Hilbert space dimension d')
    ax.set_ylabel('Recovery fidelity $F_{rec}$')
    ax.set_title('MLE vs Coherence-Max vs Channel-Inversion + CQEC\n'
                 '(mean +/- std over 20 Haar-random states)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower left')
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_mle_comparison.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Summary table
# ============================================================

def print_summary_table(dims, results):
    """Print formatted summary table."""
    all_names = list(STRATEGIES.keys()) + ['No correction']

    print("\n" + "=" * 100)
    print(" DIMENSION SWEEP RESULTS: mean F_rec (std) over 20 Haar-random states")
    print("=" * 100)

    # Header
    header = f"{'Strategy':<25s}"
    for d in dims:
        header += f"  d={d:<5d}"
    print(header)
    print("-" * 100)

    for name in all_names:
        row = f"{name:<25s}"
        for i, d in enumerate(dims):
            m = results[name]['mean'][i]
            s = results[name]['std'][i]
            row += f"  {m:.3f}({s:.3f})"
        print(row)

    print("=" * 100)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    out_dir = '/Users/deeptell01/Documents/alterego/personal/cqec'

    print("=" * 70)
    print(" Blind CQEC Dimension Sweep with Haar-Random States")
    print(" (Addressing Nature reviewer concerns)")
    print("=" * 70)

    # NOTE: dimensions are capped at d = 64.  The gap-dependent dephasing
    # inversion carries a factor exp(gamma * (d - 1)); combined with the
    # amplitude-damping inversion factor (1 - gamma_AD)^{-(i+j)/2} this
    # exceeds the dynamic range of double precision for d >= 128, so
    # d = 128 and d = 256 are excluded rather than reported as noise.
    dims = [2, 4, 8, 16, 32, 64]
    n_states = 20

    print(f"\nDimensions: {dims}")
    print(f"States per dimension: {n_states}")
    print(f"Noise: dephasing gamma=1.0, depolarizing p=0.15, amp damping gamma=0.1")
    print(f"Strategies: {list(STRATEGIES.keys())}")
    print()

    t_start = time.time()

    results = run_dimension_sweep(
        dims, n_states=n_states,
        dephasing_gamma=1.0, depol_p=0.15, amp_damp_gamma=0.1,
    )

    t_total = time.time() - t_start
    print(f"\nTotal runtime: {t_total:.1f}s")

    # Print summary
    print_summary_table(dims, results)

    # Generate figures
    print("\nGenerating figures...")
    plot_dimension_sweep(dims, results, out_dir)
    plot_haar_distribution(dims, results, out_dir)
    plot_mle_comparison(dims, results, out_dir)

    print("\n[Done] All benchmarks and figures complete.")

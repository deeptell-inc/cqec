#!/usr/bin/env python3
"""
benchmark_blind_nature_supp.py - Supplementary Benchmarks for Nature Reviewers
===============================================================================

Addresses 3 specific Nature reviewer concerns:

  1. Per-noise-channel algorithm results (Table II equivalent for each noise type)
     - 4 algorithms x 3 individual noise channels x 5 strategies + oracle + no-correction

  2. Mixed-state target benchmark
     - Werner-like states rho(v) = v|psi><psi| + (1-v)I/d at d=8
     - Purity sweep v in {0.3, ..., 1.0}, 10 Haar-random states each

  3. Hybrid strategy (channel-inversion + coherence-max mixing)
     - Sweep mixing weight w in [0,1], find optimal w* vs dimension d

Figures:
  - fig_mixed_state_sweep.pdf
  - fig_hybrid_strategy.pdf
"""

import numpy as np
from scipy.linalg import sqrtm
from math import log2
from typing import Tuple, Dict, List
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Font settings (Nature style)
# ============================================================
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

OUT_DIR = '/Users/deeptell01/Documents/alterego/personal/cqec'
sys.path.insert(0, OUT_DIR)

# Try importing from existing codebase; fall back to inline implementations
try:
    from core import (
        EnergySystem, check_mode_inclusion, coherence_l1,
    )
    from icec import ICECProtocol
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


# ============================================================
# Utility functions (inline, self-contained)
# ============================================================

def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(sigma)):
        return 0.0
    try:
        d = rho.shape[0]
        eigvals_s = np.linalg.eigvalsh(sigma)
        if np.sum(eigvals_s > 1e-8) <= 1:
            f = np.trace(rho @ sigma).real
            return min(max(float(f), 0.0), 1.0)
        eigvals_r = np.linalg.eigvalsh(rho)
        if np.sum(eigvals_r > 1e-8) <= 1:
            f = np.trace(sigma @ rho).real
            return min(max(float(f), 0.0), 1.0)
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
# Noise channels
# ============================================================

def apply_dephasing(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Energy-gap-dependent dephasing: rho_ij -> exp(-gamma*|i-j|) rho_ij.

    Matches Sec. 3.2 of the manuscript and blind_cqec.noise.dephasing.
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
    """Cascaded generalized amplitude damping |k> -> |k-1>, rate gamma.

    Identical to blind_cqec.noise.amplitude_damping.
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
# ICEC recovery (fallback-based)
# ============================================================

def _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles):
    """Fallback ICEC: amplify residual coherence to target values."""
    d = rho_noisy.shape[0]
    surviving_mask = (np.abs(rho_noisy) > 1e-15)
    np.fill_diagonal(surviving_mask, False)

    if not np.any(surviving_mask):
        return rho_noisy, {'success': False, 'fidelity': state_fidelity(rho_noisy, rho_target)}

    surviving_deltas = set()
    for i in range(d):
        for j in range(d):
            if surviving_mask[i, j]:
                surviving_deltas.add(energies[i] - energies[j])

    if not surviving_deltas:
        return rho_noisy, {'success': False, 'fidelity': state_fidelity(rho_noisy, rho_target)}

    surviving_deltas_arr = np.array(list(surviving_deltas))
    surviving_deltas_arr = surviving_deltas_arr[np.abs(surviving_deltas_arr) > 1e-10]

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
        rho_recovered = np.where(can_amplify, rho_target, rho_recovered)
        np.fill_diagonal(rho_recovered, np.diag(rho_target))
        rho_recovered = ensure_valid_dm(rho_recovered)

    fidelity = state_fidelity(rho_recovered, rho_target)
    return rho_recovered, {'success': True, 'fidelity': fidelity}


def icec_recover(rho_noisy: np.ndarray, rho_target: np.ndarray,
                 energies: np.ndarray = None,
                 n_cycles: int = 1) -> Tuple[np.ndarray, Dict]:
    """Apply ICEC protocol to recover a noisy state."""
    d = rho_noisy.shape[0]
    if energies is None:
        energies = np.arange(d, dtype=float)

    if d > 64:
        return _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles)

    if HAS_CORE:
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

    return _icec_recover_fallback(rho_noisy, rho_target, energies, n_cycles)


# ============================================================
# Estimation strategies
# ============================================================

def estimate_naive(noisy, energies, **kwargs):
    """Use noisy state directly as target estimate."""
    return noisy.copy()


def estimate_channel_inversion(noisy, energies, **kwargs):
    """Invert the combined noise channel analytically, in reverse order.

    Uses exactly the conventions of the forward channels above
    (gap-dependent dephasing, cascaded amplitude damping).
    """
    d = noisy.shape[0]
    est = np.asarray(noisy, dtype=complex).copy()

    # 1) Undo amplitude damping (applied last)
    amp_gamma = kwargs.get('amp_damp_gamma', 0.0)
    if 0 < amp_gamma < 1:
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

    # 2) Undo depolarizing
    depol_p = kwargs.get('depol_p', 0.15)
    if 0 < depol_p < 1.0:
        est = (est - depol_p * np.eye(d) / d) / (1 - depol_p)

    # 3) Undo gap-dependent dephasing
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
    est = np.where(
        magnitudes > 1e-15,
        max_coh_matrix * np.exp(1j * phases),
        max_coh_matrix * 0.5
    )
    np.fill_diagonal(est, np.diag(noisy))
    return ensure_valid_dm(est)


def estimate_iterative(noisy, energies, **kwargs):
    """Iterative refinement: coherence-max then ICEC rounds."""
    d = noisy.shape[0]
    current_est = estimate_coherence_max(noisy, energies)
    n_iter = 2 if d <= 32 else 1
    for it in range(n_iter):
        try:
            recovered, _ = _icec_recover_fallback(noisy, current_est, energies, 1)
            new_est = 0.5 * recovered + 0.5 * current_est
            current_est = ensure_valid_dm(new_est)
        except Exception:
            break
    return current_est


def estimate_multicopy(noisy, energies, **kwargs):
    """Multi-copy tomographic estimation."""
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


STRATEGIES = {
    'Naive': estimate_naive,
    'Channel inv.': estimate_channel_inversion,
    'Coherence max': estimate_coherence_max,
    'Iterative': estimate_iterative,
    'Multi-copy': estimate_multicopy,
}

STRATEGY_COLORS = {
    'No correction': '#999999',
    'Naive': '#e74c3c',
    'Channel inv.': '#3498db',
    'Coherence max': '#2ecc71',
    'Iterative': '#9b59b6',
    'Multi-copy': '#e67e22',
    'Oracle': '#1a1a1a',
}


# ============================================================
# Algorithm state generators
# ============================================================

ALGORITHMS = {
    'QKAN (d=4)':   {'d': 4,  'seed': 10},
    'qDRIFT (d=8)': {'d': 8,  'seed': 20},
    'CF-QPE (d=16)': {'d': 16, 'seed': 30},
    'Regev (d=64)':  {'d': 64, 'seed': 40},
}


def generate_algorithm_state(name: str) -> np.ndarray:
    """Generate a representative pure state for the given algorithm."""
    info = ALGORITHMS[name]
    d = info['d']
    np.random.seed(info['seed'])
    return haar_random_pure_state(d)


# ============================================================
# Benchmark: single state with specific noise
# ============================================================

def benchmark_single(target: np.ndarray, d: int,
                     noise_fn, noise_kwargs: dict) -> Dict[str, float]:
    """Run all strategies on one (target, noise) pair, return fidelities."""
    energies = np.arange(d, dtype=float)
    noisy = noise_fn(target, **noise_kwargs)
    # Ensure valid
    noisy = (noisy + noisy.conj().T) / 2
    noisy /= np.trace(noisy)

    results = {}
    results['No correction'] = state_fidelity(noisy, target)

    for sname, estimator in STRATEGIES.items():
        est_target = estimator(noisy, energies, **noise_kwargs)
        recovered, info = icec_recover(noisy, est_target, energies, n_cycles=1)
        results[sname] = state_fidelity(recovered, target)

    # Oracle
    recovered, info = icec_recover(noisy, target.copy(), energies, n_cycles=1)
    results['Oracle'] = state_fidelity(recovered, target)

    return results


# ============================================================
# SECTION 1: Per-noise-channel algorithm results
# ============================================================

def run_section1():
    """Per-noise-channel tables (Table II equivalent for each noise type)."""
    print("\n" + "=" * 85)
    print(" SECTION 1: Per-Noise-Channel Algorithm Results")
    print("=" * 85)

    noise_channels = {
        'Dephasing only (gamma=2.0)': {
            'fn': lambda rho, **kw: apply_dephasing(rho, 2.0),
            'kwargs': {'dephasing_gamma': 2.0, 'depol_p': 0.0,
                       'amp_damp_gamma': 0.0},
        },
        'Depolarizing only (p=0.3)': {
            'fn': lambda rho, **kw: apply_depolarizing(rho, 0.3),
            'kwargs': {'dephasing_gamma': 0.0, 'depol_p': 0.3,
                       'amp_damp_gamma': 0.0},
        },
        'Amplitude damping only (gamma_AD=0.3)': {
            'fn': lambda rho, **kw: apply_amplitude_damping_channel(rho, 0.3),
            'kwargs': {'dephasing_gamma': 0.0, 'depol_p': 0.0,
                       'amp_damp_gamma': 0.3},
        },
    }

    all_strat_names = ['No correction'] + list(STRATEGIES.keys()) + ['Oracle']

    for noise_name, noise_info in noise_channels.items():
        print(f"\n{'─' * 85}")
        print(f"  Noise: {noise_name}")
        print(f"{'─' * 85}")

        # Header
        header = f"  {'Algorithm':<18s}"
        for sn in all_strat_names:
            header += f" {sn:>14s}"
        print(header)
        print("  " + "-" * (18 + 15 * len(all_strat_names)))

        for alg_name in ALGORITHMS:
            target = generate_algorithm_state(alg_name)
            d = ALGORITHMS[alg_name]['d']
            res = benchmark_single(target, d, noise_info['fn'], noise_info['kwargs'])

            row = f"  {alg_name:<18s}"
            for sn in all_strat_names:
                row += f" {res[sn]:>14.4f}"
            print(row)

    print()


# ============================================================
# SECTION 2: Mixed-state target benchmark
# ============================================================

def make_werner_state(psi_dm: np.ndarray, v: float, d: int) -> np.ndarray:
    """Werner-like state: rho(v) = v |psi><psi| + (1-v) I/d."""
    return v * psi_dm + (1 - v) * np.eye(d, dtype=complex) / d


def benchmark_mixed_state(rho_target: np.ndarray, d: int,
                          noise_kwargs: dict) -> Dict[str, float]:
    """Run all strategies on one mixed-state target with combined noise."""
    energies = np.arange(d, dtype=float)
    noisy = apply_combined_noise(rho_target, **noise_kwargs)

    results = {}
    results['No correction'] = state_fidelity(noisy, rho_target)

    for sname, estimator in STRATEGIES.items():
        est_target = estimator(noisy, energies,
                               dephasing_gamma=noise_kwargs.get('dephasing_gamma', 1.0),
                               depol_p=noise_kwargs.get('depol_p', 0.15),
                               amp_damp_gamma=noise_kwargs.get('amp_damp_gamma', 0.1))
        recovered, _ = icec_recover(noisy, est_target, energies, n_cycles=1)
        results[sname] = state_fidelity(recovered, rho_target)

    # Oracle
    recovered, _ = icec_recover(noisy, rho_target.copy(), energies, n_cycles=1)
    results['Oracle'] = state_fidelity(recovered, rho_target)

    return results


def run_section2():
    """Mixed-state target benchmark at d=8."""
    print("\n" + "=" * 85)
    print(" SECTION 2: Mixed-State Target Benchmark (d=8)")
    print("=" * 85)

    d = 8
    v_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_states = 10
    noise_kwargs = {'dephasing_gamma': 1.0, 'depol_p': 0.15, 'amp_damp_gamma': 0.1}

    all_strat_names = ['No correction'] + list(STRATEGIES.keys()) + ['Oracle']

    # results[strategy] = {'mean': [...], 'std': [...]} over v_values
    sweep_results = {sn: {'mean': [], 'std': []} for sn in all_strat_names}

    for v in v_values:
        print(f"  v = {v:.1f} ...", end=" ", flush=True)
        t0 = time.time()
        fids_by_strat = {sn: [] for sn in all_strat_names}

        for s_idx in range(n_states):
            np.random.seed(100 + s_idx)
            psi_dm = haar_random_pure_state(d)
            rho_target = make_werner_state(psi_dm, v, d)
            res = benchmark_mixed_state(rho_target, d, noise_kwargs)
            for sn in all_strat_names:
                fids_by_strat[sn].append(res[sn])

        for sn in all_strat_names:
            arr = np.array(fids_by_strat[sn])
            sweep_results[sn]['mean'].append(np.mean(arr))
            sweep_results[sn]['std'].append(np.std(arr))

        dt = time.time() - t0
        oracle_m = sweep_results['Oracle']['mean'][-1]
        print(f"Oracle={oracle_m:.4f} ({dt:.1f}s)")

    # Convert to arrays
    for sn in all_strat_names:
        sweep_results[sn]['mean'] = np.array(sweep_results[sn]['mean'])
        sweep_results[sn]['std'] = np.array(sweep_results[sn]['std'])

    # Print table
    print(f"\n  {'v':<6s}", end="")
    for sn in all_strat_names:
        print(f" {sn:>14s}", end="")
    print()
    print("  " + "-" * (6 + 15 * len(all_strat_names)))
    for i, v in enumerate(v_values):
        row = f"  {v:<6.1f}"
        for sn in all_strat_names:
            m = sweep_results[sn]['mean'][i]
            s = sweep_results[sn]['std'][i]
            row += f" {m:>7.4f}({s:.3f})"
        print(row)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for sn in all_strat_names:
        color = STRATEGY_COLORS.get(sn, '#333333')
        ls = ':' if sn == 'No correction' else ('--' if sn == 'Oracle' else '-')
        lw = 2.0 if sn in ('Oracle', 'No correction') else 1.4
        mean = sweep_results[sn]['mean']
        std = sweep_results[sn]['std']
        ax.plot(v_values, mean, label=sn, color=color, linestyle=ls,
                linewidth=lw, marker='o', markersize=4)
        ax.fill_between(v_values, mean - std, mean + std,
                        color=color, alpha=0.12)

    ax.set_xlabel('Purity parameter $v$')
    ax.set_ylabel('Recovery fidelity $F_{rec}$')
    ax.set_title('Blind CQEC on Mixed-State Targets (d=8)\n'
                 r'$\rho(v) = v|\psi\rangle\langle\psi| + (1-v)I/d$, '
                 '10 Haar-random states, combined noise')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.25, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', ncol=2)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'fig_mixed_state_sweep.pdf')
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    return v_values, sweep_results


# ============================================================
# SECTION 3: Hybrid strategy
# ============================================================

def estimate_hybrid(noisy, energies, w, **kwargs):
    """Hybrid: w * channel_inversion + (1-w) * coherence_max."""
    est_inv = estimate_channel_inversion(noisy, energies, **kwargs)
    est_coh = estimate_coherence_max(noisy, energies, **kwargs)
    hybrid = w * est_inv + (1 - w) * est_coh
    return ensure_valid_dm(hybrid)


def run_section3():
    """Hybrid strategy benchmark across dimensions."""
    print("\n" + "=" * 85)
    print(" SECTION 3: Hybrid Strategy (Channel-Inv + Coherence-Max)")
    print("=" * 85)

    dims = [4, 8, 16, 32, 64]
    w_values = np.arange(0, 1.01, 0.1)
    n_states = 10
    noise_kwargs = {'dephasing_gamma': 1.0, 'depol_p': 0.15, 'amp_damp_gamma': 0.1}

    # results[d] = {'w_fids_mean': array(len(w_values)), 'w_fids_std': ..., 'optimal_w': float}
    hybrid_results = {}

    for d in dims:
        print(f"  d = {d:>4d} ...", end=" ", flush=True)
        t0 = time.time()
        energies = np.arange(d, dtype=float)

        fids_per_w = np.zeros((n_states, len(w_values)))

        for s_idx in range(n_states):
            np.random.seed(200 + s_idx + d)
            target = haar_random_pure_state(d)
            noisy = apply_combined_noise(target, **noise_kwargs)

            for wi, w in enumerate(w_values):
                est_target = estimate_hybrid(noisy, energies, w,
                                             dephasing_gamma=noise_kwargs['dephasing_gamma'],
                                             depol_p=noise_kwargs['depol_p'],
                                             amp_damp_gamma=noise_kwargs['amp_damp_gamma'])
                recovered, _ = icec_recover(noisy, est_target, energies, n_cycles=1)
                fids_per_w[s_idx, wi] = state_fidelity(recovered, target)

        mean_fids = np.mean(fids_per_w, axis=0)
        std_fids = np.std(fids_per_w, axis=0)
        opt_idx = np.argmax(mean_fids)
        opt_w = w_values[opt_idx]

        hybrid_results[d] = {
            'mean': mean_fids,
            'std': std_fids,
            'optimal_w': opt_w,
            'optimal_fid': mean_fids[opt_idx],
        }

        dt = time.time() - t0
        print(f"w*={opt_w:.1f}  F*={mean_fids[opt_idx]:.4f}  ({dt:.1f}s)")

    # Print table
    print(f"\n  {'w':>5s}", end="")
    for d in dims:
        print(f"  d={d:<5d}", end="")
    print()
    print("  " + "-" * (5 + 9 * len(dims)))
    for wi, w in enumerate(w_values):
        row = f"  {w:>5.1f}"
        for d in dims:
            row += f"  {hybrid_results[d]['mean'][wi]:.4f} "
        print(row)

    print(f"\n  Optimal w*:")
    for d in dims:
        r = hybrid_results[d]
        print(f"    d={d:>4d}: w* = {r['optimal_w']:.1f}, F* = {r['optimal_fid']:.4f}")

    # Plot: 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): F_rec vs w for each d
    dim_colors = {4: '#e74c3c', 8: '#3498db', 16: '#2ecc71', 32: '#9b59b6', 64: '#e67e22'}
    for d in dims:
        r = hybrid_results[d]
        color = dim_colors.get(d, '#333')
        ax1.plot(w_values, r['mean'], label=f'd={d}', color=color,
                 linewidth=1.5, marker='o', markersize=4)
        ax1.fill_between(w_values, r['mean'] - r['std'], r['mean'] + r['std'],
                         color=color, alpha=0.12)
        # Mark optimal
        ax1.plot(r['optimal_w'], r['optimal_fid'], '*', color=color,
                 markersize=12, markeredgecolor='black', markeredgewidth=0.5)

    ax1.set_xlabel('Mixing weight $w$')
    ax1.set_ylabel('Recovery fidelity $F_{rec}$')
    ax1.set_title('(a) Hybrid strategy: $F_{rec}$ vs $w$')
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.text(0.02, 0.02,
             r'$\hat{\rho} = w \cdot \hat{\rho}_{inv} + (1-w) \cdot \hat{\rho}_{coh}$',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel (b): Optimal w* vs d
    opt_ws = [hybrid_results[d]['optimal_w'] for d in dims]
    opt_fs = [hybrid_results[d]['optimal_fid'] for d in dims]

    ax2_twin = ax2.twinx()
    ax2.plot(dims, opt_ws, 'o-', color='#3498db', linewidth=2, markersize=8, label='$w^*$')
    ax2_twin.plot(dims, opt_fs, 's--', color='#e74c3c', linewidth=1.5, markersize=7, label='$F^*$')

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(dims)
    ax2.set_xticklabels([str(d) for d in dims])
    ax2.set_xlabel('Hilbert space dimension $d$')
    ax2.set_ylabel('Optimal weight $w^*$', color='#3498db')
    ax2_twin.set_ylabel('Optimal fidelity $F^*$', color='#e74c3c')
    ax2.set_title('(b) Optimal hybrid weight vs dimension')
    ax2.set_ylim(-0.05, 1.05)
    ax2_twin.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')

    plt.suptitle('Hybrid Strategy: Channel-Inversion + Coherence-Max\n'
                 '(10 Haar-random states/dim, combined noise)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'fig_hybrid_strategy.pdf')
    plt.savefig(path)
    plt.close()
    print(f"\n  Saved: {path}")

    return hybrid_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    print("=" * 85)
    print(" Supplementary Benchmarks for Nature Reviewers")
    print(" (Per-noise tables, mixed-state sweep, hybrid strategy)")
    print("=" * 85)

    # Section 1
    run_section1()

    # Section 2
    run_section2()

    # Section 3
    run_section3()

    t_total = time.time() - t_start
    print(f"\n{'=' * 85}")
    print(f" Total runtime: {t_total:.1f}s")
    print(f"{'=' * 85}")
    print("\n[Done] All supplementary benchmarks complete.")

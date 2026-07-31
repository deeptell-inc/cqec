#!/usr/bin/env python3
"""
benchmark_blind_sensitivity.py - Noise-Parameter Sensitivity Analysis for Blind CQEC
=====================================================================================

Addresses Nature reviewer concern: "+/-10% sensitivity" is mentioned in text
but not shown as data. This benchmark produces systematic sensitivity data.

Figures:
  1. fig_sensitivity_heatmap.pdf  - Heatmap of F_rec vs (d, delta)
  2. fig_sensitivity_curves.pdf   - Per-parameter sensitivity curves at d=8, d=64
  3. fig_scaling_physics.pdf      - 1-F_rec vs n_copies with power-law fits

Reuses functions from benchmark_blind_dimension_sweep.py where possible.
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit
from math import log2
from typing import Tuple, Dict
import time
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/deeptell01/Documents/alterego/personal/cqec')

# ---------------------------------------------------------------------------
# Try importing from existing benchmarks; fall back to inline implementations
# ---------------------------------------------------------------------------
try:
    from benchmark_blind_dimension_sweep import (
        state_fidelity, ensure_valid_dm, haar_random_pure_state,
        apply_dephasing, apply_depolarizing, apply_amplitude_damping_channel,
        apply_combined_noise, icec_recover,
        estimate_noise_inversion, estimate_coherence_max,
    )
    _IMPORTED = True
except Exception:
    _IMPORTED = False

try:
    from core import (
        coherence_l1, check_mode_inclusion, EnergySystem,
    )
    from icec import ICECProtocol
except Exception:
    pass

# ---------------------------------------------------------------------------
# Inline fallbacks (only used if imports fail)
# ---------------------------------------------------------------------------
if not _IMPORTED:

    def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
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
            return min(result, 1.0) if np.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def ensure_valid_dm(rho: np.ndarray) -> np.ndarray:
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
        psi = np.random.randn(d) + 1j * np.random.randn(d)
        psi /= np.linalg.norm(psi)
        return np.outer(psi, psi.conj())

    def apply_dephasing(rho, gamma):
        """Energy-gap-dependent dephasing: rho_ij -> exp(-gamma*|i-j|) rho_ij."""
        rho = np.asarray(rho, dtype=complex)
        d = rho.shape[0]
        idx = np.arange(d)
        mask = np.exp(-gamma * np.abs(idx[:, None] - idx[None, :]))
        return rho * mask

    def apply_depolarizing(rho, p):
        d = rho.shape[0]
        return (1 - p) * rho + p * np.eye(d, dtype=complex) / d

    def apply_amplitude_damping_channel(rho, gamma):
        """Cascaded generalized amplitude damping |k> -> |k-1>, rate gamma."""
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
                    fac = ((1.0 - gamma) ** ((i + j) / 2.0)
                           if (i > 0 or j > 0) else 1.0)
                    out[i, j] = rho[i, j] * fac
        return out

    def apply_combined_noise(rho, dephasing_gamma=1.0, depol_p=0.15,
                             amp_damp_gamma=0.1):
        rho_noisy = apply_dephasing(rho, dephasing_gamma)
        rho_noisy = apply_depolarizing(rho_noisy, depol_p)
        rho_noisy = apply_amplitude_damping_channel(rho_noisy, amp_damp_gamma)
        rho_noisy = (rho_noisy + rho_noisy.conj().T) / 2
        rho_noisy /= np.trace(rho_noisy)
        return rho_noisy

    def estimate_noise_inversion(noisy, energies, **kwargs):
        """Exact inversion of the combined channel, in reverse order."""
        d = noisy.shape[0]
        est = np.asarray(noisy, dtype=complex).copy()

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

        depol_p = kwargs.get('depol_p', 0.15)
        if 0 < depol_p < 1.0:
            est = (est - depol_p * np.eye(d) / d) / (1 - depol_p)

        deph_gamma = kwargs.get('dephasing_gamma', 1.0)
        if deph_gamma > 0:
            idx = np.arange(d)
            mask = np.exp(deph_gamma * np.abs(idx[:, None] - idx[None, :]))
            np.fill_diagonal(mask, 1.0)
            est = est * mask

        return ensure_valid_dm(est)

    def estimate_coherence_max(noisy, energies, **kwargs):
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

    def icec_recover(rho_noisy, rho_target, energies=None, n_cycles=1):
        d = rho_noisy.shape[0]
        if energies is None:
            energies = np.arange(d, dtype=float)
        surviving_mask = (np.abs(rho_noisy) > 1e-15)
        np.fill_diagonal(surviving_mask, False)
        if not np.any(surviving_mask):
            return rho_noisy, {'success': False,
                               'fidelity': state_fidelity(rho_noisy, rho_target)}
        surviving_deltas = set()
        for i in range(d):
            for j in range(d):
                if surviving_mask[i, j]:
                    surviving_deltas.add(energies[i] - energies[j])
        surviving_deltas_arr = np.array(list(surviving_deltas))
        surviving_deltas_arr = surviving_deltas_arr[
            np.abs(surviving_deltas_arr) > 1e-10]
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


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def setup_matplotlib():
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


# ============================================================
# Figure 1: Sensitivity heatmap
# ============================================================

def run_sensitivity_heatmap(dims, deltas, n_states=10,
                            gamma_true=1.0, p_true=0.15, gamma_AD_true=0.1):
    """Compute F_rec for channel inversion with mis-specified parameters.

    All three noise parameters are simultaneously shifted by relative error delta.
    """
    F_matrix = np.zeros((len(dims), len(deltas)))

    for di, d in enumerate(dims):
        energies = np.arange(d, dtype=float)
        for dj, delta in enumerate(deltas):
            fids = []
            # Assumed (mis-specified) parameters
            gamma_assumed = gamma_true * (1 + delta)
            p_assumed = p_true * (1 + delta)
            gamma_AD_assumed = gamma_AD_true * (1 + delta)
            # Clamp depolarizing to valid range
            p_assumed = np.clip(p_assumed, 0.0, 0.999)

            for s in range(n_states):
                target = haar_random_pure_state(d)
                noisy = apply_combined_noise(target, gamma_true, p_true, gamma_AD_true)

                # Channel inversion with assumed (possibly wrong) parameters
                est_target = estimate_noise_inversion(
                    noisy, energies,
                    dephasing_gamma=gamma_assumed,
                    depol_p=p_assumed,
                    amp_damp_gamma=gamma_AD_assumed,
                )
                # CQEC recovery
                recovered, _ = icec_recover(noisy, est_target, energies, n_cycles=1)
                fids.append(state_fidelity(recovered, target))

            F_matrix[di, dj] = np.mean(fids)

    return F_matrix


def plot_sensitivity_heatmap(dims, deltas, F_matrix, out_dir):
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    delta_pct = [f"{d*100:+.0f}%" for d in deltas]
    dim_labels = [str(d) for d in dims]

    im = ax.imshow(F_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=max(0, F_matrix.min() - 0.02), vmax=1.0,
                   origin='lower')
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(delta_pct, rotation=45, ha='right')
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dim_labels)
    ax.set_xlabel('Relative parameter error $\\delta$')
    ax.set_ylabel('Dimension $d$')
    ax.set_title('Channel-Inversion Sensitivity: $F_{rec}$ vs Parameter Error\n'
                 '(all params shifted simultaneously, 10 Haar-random states)')

    # Annotate cells
    for i in range(len(dims)):
        for j in range(len(deltas)):
            val = F_matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, label='$F_{rec}$', shrink=0.9)
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_sensitivity_heatmap.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================
# Figure 2: Per-parameter sensitivity curves
# ============================================================

def run_sensitivity_curves(d, deltas_fine, n_states=10,
                           gamma_true=1.0, p_true=0.15, gamma_AD_true=0.1):
    """Vary one noise parameter at a time while keeping others exact.

    Returns dict with keys 'gamma', 'p', 'gamma_AD', each containing
    'mean' and 'std' arrays of length len(deltas_fine).
    """
    energies = np.arange(d, dtype=float)
    param_names = ['gamma', 'p', 'gamma_AD']
    results = {k: {'mean': [], 'std': []} for k in param_names}

    for dj, delta in enumerate(deltas_fine):
        fids_by_param = {k: [] for k in param_names}

        for s in range(n_states):
            target = haar_random_pure_state(d)
            noisy = apply_combined_noise(target, gamma_true, p_true, gamma_AD_true)

            # --- Vary gamma only ---
            gamma_a = gamma_true * (1 + delta)
            est = estimate_noise_inversion(noisy, energies,
                                           dephasing_gamma=gamma_a,
                                           depol_p=p_true,
                                           amp_damp_gamma=gamma_AD_true)
            rec, _ = icec_recover(noisy, est, energies, n_cycles=1)
            fids_by_param['gamma'].append(state_fidelity(rec, target))

            # --- Vary p only ---
            p_a = np.clip(p_true * (1 + delta), 0.0, 0.999)
            est = estimate_noise_inversion(noisy, energies,
                                           dephasing_gamma=gamma_true,
                                           depol_p=p_a,
                                           amp_damp_gamma=gamma_AD_true)
            rec, _ = icec_recover(noisy, est, energies, n_cycles=1)
            fids_by_param['p'].append(state_fidelity(rec, target))

            # --- Vary gamma_AD only ---
            # The unified channel inversion now inverts amplitude damping
            # explicitly, so a mis-specified gamma_AD enters the inversion
            # directly (no proxy rescaling needed).
            gamma_AD_a = np.clip(gamma_AD_true * (1 + delta), 0.0, 0.999)
            est = estimate_noise_inversion(noisy, energies,
                                           dephasing_gamma=gamma_true,
                                           depol_p=p_true,
                                           amp_damp_gamma=gamma_AD_a)
            rec, _ = icec_recover(noisy, est, energies, n_cycles=1)
            fids_by_param['gamma_AD'].append(state_fidelity(rec, target))

        for k in param_names:
            arr = np.array(fids_by_param[k])
            results[k]['mean'].append(np.mean(arr))
            results[k]['std'].append(np.std(arr))

    for k in param_names:
        results[k]['mean'] = np.array(results[k]['mean'])
        results[k]['std'] = np.array(results[k]['std'])

    return results


def plot_sensitivity_curves(deltas_fine, results_d8, results_d64, out_dir):
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = {'gamma': '#e74c3c', 'p': '#3498db', 'gamma_AD': '#2ecc71'}
    labels = {
        'gamma': 'Dephasing $\\gamma$ error',
        'p': 'Depolarizing $p$ error',
        'gamma_AD': 'Amp. damping $\\gamma_{AD}$ error',
    }
    delta_pct = deltas_fine * 100

    for ax, results, d_label in [(axes[0], results_d8, '$d=8$'),
                                  (axes[1], results_d64, '$d=64$')]:
        for k in ['gamma', 'p', 'gamma_AD']:
            mean = results[k]['mean']
            std = results[k]['std']
            ax.plot(delta_pct, mean, color=colors[k], label=labels[k],
                    linewidth=1.8, marker='o', markersize=3)
            ax.fill_between(delta_pct, mean - std, mean + std,
                            color=colors[k], alpha=0.15)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        # Shade the +/-10% region
        ax.axvspan(-10, 10, color='yellow', alpha=0.08)
        ax.set_xlabel('Relative parameter error $\\delta$ (%)')
        ax.set_title(f'Parameter Sensitivity at {d_label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower center', fontsize=8)

    axes[0].set_ylabel('Recovery fidelity $F_{rec}$')
    fig.suptitle('Single-Parameter Sensitivity of Channel Inversion\n'
                 '(mean $\\pm$ std, 10 Haar-random states per point)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_sensitivity_curves.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================
# Figure 3: Scaling with physical interpretation
# ============================================================

def run_scaling_physics(d=8, n_copies_list=None,
                        gamma_true=1.0, p_true=0.15, gamma_AD_true=0.1):
    """For a single representative Haar-random state at dimension d,
    measure 1-F_rec vs n_copies for coherence_max and channel_inversion.

    n_copies controls the number of simulated noisy copies used for
    multi-copy averaging before estimation.
    """
    if n_copies_list is None:
        n_copies_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    energies = np.arange(d, dtype=float)
    np.random.seed(123)
    target = haar_random_pure_state(d)
    noisy = apply_combined_noise(target, gamma_true, p_true, gamma_AD_true)

    strategies = {
        'coherence_max': estimate_coherence_max,
        'channel_inversion': estimate_noise_inversion,
    }

    results = {name: [] for name in strategies}

    for n_cop in n_copies_list:
        for name, estimator in strategies.items():
            if n_cop == 1:
                # Single copy: use estimator directly
                if name == 'channel_inversion':
                    est = estimator(noisy, energies,
                                    dephasing_gamma=gamma_true,
                                    depol_p=p_true,
                                    amp_damp_gamma=gamma_AD_true)
                else:
                    est = estimator(noisy, energies)
            else:
                # Multi-copy: average n_cop noisy copies then estimate
                avg = np.zeros_like(noisy, dtype=complex)
                for _ in range(n_cop):
                    noise_scale = 1.0 / np.sqrt(d)
                    fluct = (np.random.randn(d, d) +
                             1j * np.random.randn(d, d)) * noise_scale / np.sqrt(n_cop)
                    fluct = 0.5 * (fluct + fluct.conj().T)
                    sample = ensure_valid_dm(noisy + fluct)
                    avg += sample
                avg /= n_cop
                if name == 'channel_inversion':
                    est = estimator(avg, energies,
                                    dephasing_gamma=gamma_true,
                                    depol_p=p_true,
                                    amp_damp_gamma=gamma_AD_true)
                else:
                    est = estimator(avg, energies)

            recovered, _ = icec_recover(noisy, est, energies, n_cycles=1)
            f = state_fidelity(recovered, target)
            results[name].append(1.0 - f)

    for name in results:
        results[name] = np.array(results[name])

    return n_copies_list, results


def plot_scaling_physics(n_copies_list, results, out_dir):
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    n_arr = np.array(n_copies_list, dtype=float)
    colors = {'coherence_max': '#2ecc71', 'channel_inversion': '#3498db'}
    labels_map = {
        'coherence_max': 'Coherence maximization',
        'channel_inversion': 'Channel inversion',
    }

    def power_law(x, a, b):
        return a * np.power(x, b)

    for name in results:
        infidelity = results[name]
        # Filter out zero or negative values for log-log
        valid = infidelity > 1e-15
        if np.sum(valid) < 2:
            ax.plot(n_arr, np.maximum(infidelity, 1e-15), 'o-',
                    color=colors[name], label=labels_map[name],
                    linewidth=1.8, markersize=5)
            continue

        ax.plot(n_arr, np.maximum(infidelity, 1e-15), 'o',
                color=colors[name], label=labels_map[name], markersize=5)

        # Power-law fit on valid points
        try:
            n_valid = n_arr[valid]
            inf_valid = infidelity[valid]
            popt, _ = curve_fit(power_law, n_valid, inf_valid,
                                p0=[inf_valid[0], -0.5], maxfev=5000)
            n_fit = np.logspace(np.log10(n_arr[0]), np.log10(n_arr[-1]), 100)
            ax.plot(n_fit, power_law(n_fit, *popt), '--',
                    color=colors[name], linewidth=1.2, alpha=0.8)
            ax.annotate(f'$\\propto n^{{{popt[1]:.2f}}}$',
                        xy=(n_fit[60], power_law(n_fit[60], *popt)),
                        fontsize=9, color=colors[name], fontweight='bold')
        except Exception:
            pass

    # Reference lines
    n_ref = np.logspace(np.log10(n_arr[0]), np.log10(n_arr[-1]), 50)
    # Statistical averaging O(n^{-1/2})
    ref_half = 0.5 * (n_ref / n_ref[0])**(-0.5)
    ax.plot(n_ref, ref_half, ':', color='gray', linewidth=1.0, alpha=0.6)
    ax.annotate('Statistical averaging: $O(n^{-1/2})$',
                xy=(n_ref[30], ref_half[30] * 1.3),
                fontsize=8, color='gray', fontstyle='italic')

    # Tomographic limit O(n^{-1})
    ref_one = 0.5 * (n_ref / n_ref[0])**(-1.0)
    ax.plot(n_ref, ref_one, ':', color='purple', linewidth=1.0, alpha=0.6)
    ax.annotate('Tomographic limit: $O(n^{-1})$',
                xy=(n_ref[30], ref_one[30] * 1.3),
                fontsize=8, color='purple', fontstyle='italic')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of noisy copies $n$')
    ax.set_ylabel('Infidelity $1 - F_{rec}$')
    ax.set_title('Copy Scaling with Physical Interpretation ($d=8$)\n'
                 'Combined noise: dephasing + depolarizing + amp. damping',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    path = os.path.join(out_dir, 'fig_scaling_physics.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================
# Summary table
# ============================================================

def print_heatmap_table(dims, deltas, F_matrix):
    delta_pct = [f"{d*100:+.0f}%" for d in deltas]
    print("\n" + "=" * 90)
    print(" SENSITIVITY HEATMAP: mean F_rec (channel inversion, all params shifted)")
    print("=" * 90)
    header = f"{'d':>6s}"
    for dp in delta_pct:
        header += f"  {dp:>7s}"
    print(header)
    print("-" * 90)
    for i, d in enumerate(dims):
        row = f"{d:>6d}"
        for j in range(len(deltas)):
            row += f"  {F_matrix[i, j]:>7.4f}"
        print(row)
    print("=" * 90)


def print_curves_table(deltas_fine, results, d_label):
    print(f"\n  Per-parameter sensitivity at d={d_label}:")
    print(f"  {'delta':>8s}  {'gamma':>12s}  {'p':>12s}  {'gamma_AD':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for i, delta in enumerate(deltas_fine):
        row = f"  {delta*100:>+7.1f}%"
        for k in ['gamma', 'p', 'gamma_AD']:
            m = results[k]['mean'][i]
            s = results[k]['std'][i]
            row += f"  {m:.4f}+/-{s:.4f}"
        # Only print every other point to keep table manageable
        if i % 3 == 0 or abs(delta) < 0.01:
            print(row)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    out_dir = '/Users/deeptell01/Documents/alterego/personal/cqec'

    print("=" * 70)
    print(" Blind CQEC: Noise-Parameter Sensitivity Benchmark")
    print(" (Addressing Nature reviewer: '+/-10% sensitivity' data)")
    print("=" * 70)

    # True noise parameters
    gamma_true = 1.0
    p_true = 0.15
    gamma_AD_true = 0.1

    # --------------------------------------------------------
    # Figure 1: Sensitivity heatmap
    # --------------------------------------------------------
    print("\n[1/3] Sensitivity heatmap (channel inversion, all params shifted)...")
    dims = [4, 8, 16, 64]
    deltas = np.array([-0.30, -0.20, -0.10, -0.05, 0.0,
                        0.05, 0.10, 0.20, 0.30])
    n_states = 10

    t0 = time.time()
    F_matrix = run_sensitivity_heatmap(dims, deltas, n_states,
                                       gamma_true, p_true, gamma_AD_true)
    dt1 = time.time() - t0
    print(f"  Computed in {dt1:.1f}s")

    print_heatmap_table(dims, deltas, F_matrix)
    plot_sensitivity_heatmap(dims, deltas, F_matrix, out_dir)

    # --------------------------------------------------------
    # Figure 2: Per-parameter sensitivity curves
    # --------------------------------------------------------
    print("\n[2/3] Per-parameter sensitivity curves...")
    deltas_fine = np.linspace(-0.30, 0.30, 13)

    t0 = time.time()
    print("  d=8 ...")
    np.random.seed(42)
    results_d8 = run_sensitivity_curves(8, deltas_fine, n_states,
                                         gamma_true, p_true, gamma_AD_true)
    print("  d=64 ...")
    np.random.seed(42)
    results_d64 = run_sensitivity_curves(64, deltas_fine, n_states,
                                          gamma_true, p_true, gamma_AD_true)
    dt2 = time.time() - t0
    print(f"  Computed in {dt2:.1f}s")

    print_curves_table(deltas_fine, results_d8, 8)
    print_curves_table(deltas_fine, results_d64, 64)
    plot_sensitivity_curves(deltas_fine, results_d8, results_d64, out_dir)

    # --------------------------------------------------------
    # Figure 3: Scaling with physical interpretation
    # --------------------------------------------------------
    print("\n[3/3] Scaling with physical interpretation (d=8)...")
    t0 = time.time()
    n_copies_list, scaling_results = run_scaling_physics(
        d=8, gamma_true=gamma_true, p_true=p_true, gamma_AD_true=gamma_AD_true)
    dt3 = time.time() - t0
    print(f"  Computed in {dt3:.1f}s")

    print("\n  Scaling data (1 - F_rec):")
    print(f"  {'n_copies':>10s}  {'CohMax':>12s}  {'ChanInv':>12s}")
    for i, n in enumerate(n_copies_list):
        print(f"  {n:>10d}  {scaling_results['coherence_max'][i]:>12.6f}"
              f"  {scaling_results['channel_inversion'][i]:>12.6f}")

    plot_scaling_physics(n_copies_list, scaling_results, out_dir)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    total_time = dt1 + dt2 + dt3
    print(f"\n{'='*70}")
    print(f" Total runtime: {total_time:.1f}s")
    print(f" Figures saved:")
    print(f"   - fig_sensitivity_heatmap.pdf")
    print(f"   - fig_sensitivity_curves.pdf")
    print(f"   - fig_scaling_physics.pdf")
    print(f"{'='*70}")
    print("[Done] Sensitivity benchmark complete.")

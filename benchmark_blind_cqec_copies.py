#!/usr/bin/env python3
"""
benchmark_blind_cqec_copies.py - Copy-Economized Blind CQEC
=============================================================

How does blind CQEC fidelity degrade when we reduce the number of copies
available for catalytic amplification?

Sweeps n_copies from 1 to 200, at fixed noise strengths, for each
estimation strategy. Also computes:
  - Minimum copies needed to reach F >= 0.95 / 0.99
  - Fidelity-per-copy efficiency: F(n) / n
  - Copy scaling exponent: fit F(n) ~ 1 - A * n^{-alpha}

Three noise channels x {qubit, qutrit} x 6 strategies = 36 curves.
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit
from typing import Callable, Optional
import sys
import os
import warnings
warnings.filterwarnings('ignore')

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
    EnergySystem, trace_distance,
    coherence_l1, partial_dephasing, amplitude_damping,
    depolarizing_channel, maximally_coherent_state,
)
from icec import ICECProtocol

# Import estimation strategies from the previous benchmark
from benchmark_blind_cqec import (
    state_fidelity, ensure_valid_dm,
    estimate_naive, estimate_noise_inversion,
    estimate_coherence_max, estimate_iterative,
    estimate_multicopy,
)


# ============================================================
# Copy-sweep benchmark
# ============================================================

STRATEGIES = {
    'Naive': estimate_naive,
    'Channel inversion': estimate_noise_inversion,
    'Coherence max': estimate_coherence_max,
    'Iterative (5 rounds)': estimate_iterative,
    'Multi-copy tomo': estimate_multicopy,
    'Oracle': None,
}

COLORS = {
    'Naive': '#e74c3c',
    'Channel inversion': '#3498db',
    'Coherence max': '#2ecc71',
    'Iterative (5 rounds)': '#9b59b6',
    'Multi-copy tomo': '#e67e22',
    'Oracle': '#1a1a1a',
    'No correction': '#999999',
}


def sweep_copies(
    target: np.ndarray,
    energies: np.ndarray,
    noise_fn: Callable,
    noise_param: float,
    noise_type: str,
    copy_values: np.ndarray,
) -> dict:
    """Sweep n_copies and measure fidelity for each estimation strategy."""

    noisy = noise_fn(target)
    target_coh = coherence_l1(target, energies)

    # Baselines (independent of copies)
    f_noisy = state_fidelity(noisy, target)
    d_noisy = trace_distance(noisy, target)

    results = {name: {'fidelity': [], 'trace_dist': [], 'coh_ratio': []}
               for name in STRATEGIES}
    results['No correction'] = {
        'fidelity': [f_noisy] * len(copy_values),
        'trace_dist': [d_noisy] * len(copy_values),
        'coh_ratio': [coherence_l1(noisy, energies) / max(target_coh, 1e-15)] * len(copy_values),
    }

    for n_c in copy_values:
        n_c = int(n_c)
        for name, estimator in STRATEGIES.items():
            if name == 'Oracle':
                est_target = target.copy()
            else:
                est_target = estimator(
                    noisy, energies,
                    noise_channel=noise_fn,
                    noise_param=noise_param,
                    noise_type=noise_type,
                    n_copies=n_c,
                    n_iterations=min(5, n_c),  # cap iterations by copies
                    n_samples=min(20, n_c),
                )

            protocol = ICECProtocol(energies, est_target)
            protocol.initialize(n_c)
            recovered, _ = protocol.correct(noisy, n_c)

            f = state_fidelity(recovered, target)
            d = trace_distance(recovered, target)
            c = coherence_l1(recovered, energies) / max(target_coh, 1e-15)

            results[name]['fidelity'].append(f)
            results[name]['trace_dist'].append(d)
            results[name]['coh_ratio'].append(c)

    return results


def find_min_copies(fidelities: list, copy_values: np.ndarray,
                    threshold: float) -> Optional[int]:
    """Find minimum copies to reach a fidelity threshold."""
    for i, f in enumerate(fidelities):
        if f >= threshold:
            return int(copy_values[i])
    return None


def fit_scaling(fidelities: list, copy_values: np.ndarray):
    """Fit F(n) ~ 1 - A * n^{-alpha}. Returns (A, alpha) or None."""
    try:
        def model(n, A, alpha):
            return 1.0 - A * np.power(n, -alpha)
        fid = np.array(fidelities)
        # Only fit where F < 0.999 (avoid flat region)
        mask = fid < 0.999
        if np.sum(mask) < 3:
            return None
        popt, _ = curve_fit(model, copy_values[mask], fid[mask],
                            p0=[0.5, 0.5], bounds=([0, 0], [10, 5]),
                            maxfev=5000)
        return popt  # (A, alpha)
    except Exception:
        return None


# ============================================================
# Plotting
# ============================================================

def plot_copy_sweep(results: dict, copy_values: np.ndarray,
                    title: str, filename: str):
    """Plot fidelity vs n_copies for all strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Fidelity vs copies
    ax = axes[0]
    for name, data in results.items():
        ls = ':' if name == 'No correction' else ('--' if name == 'Oracle' else '-')
        lw = 2.5 if name in ('Oracle', 'No correction') else 1.8
        ax.plot(copy_values, data['fidelity'], label=name,
                color=COLORS.get(name, '#333'), linestyle=ls, linewidth=lw,
                marker='o', markersize=2.5)
    ax.set_xlabel('Number of copies n', fontsize=12)
    ax.set_ylabel('Fidelity with true target', fontsize=12)
    ax.set_title('Fidelity vs Copy Budget', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(copy_values[-1] * 0.85, 0.955, 'F=0.95', fontsize=8, color='gray')
    ax.text(copy_values[-1] * 0.85, 0.995, 'F=0.99', fontsize=8, color='gray')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    # Panel 2: Fidelity vs copies (log-scale for copies)
    ax = axes[1]
    for name, data in results.items():
        ls = ':' if name == 'No correction' else ('--' if name == 'Oracle' else '-')
        lw = 2.5 if name in ('Oracle', 'No correction') else 1.8
        ax.semilogx(copy_values, data['fidelity'], label=name,
                     color=COLORS.get(name, '#333'), linestyle=ls, linewidth=lw,
                     marker='o', markersize=2.5)
    ax.set_xlabel('Number of copies n (log scale)', fontsize=12)
    ax.set_ylabel('Fidelity with true target', fontsize=12)
    ax.set_title('Fidelity vs Copy Budget (log)', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    # Panel 3: Fidelity efficiency F(n)/n
    ax = axes[2]
    for name, data in results.items():
        if name == 'No correction':
            continue
        fid_arr = np.array(data['fidelity'])
        efficiency = fid_arr / copy_values
        ax.loglog(copy_values, efficiency, label=name,
                  color=COLORS.get(name, '#333'),
                  linestyle='--' if name == 'Oracle' else '-',
                  linewidth=1.8, marker='s', markersize=2.5)
    ax.set_xlabel('Number of copies n', fontsize=12)
    ax.set_ylabel('Fidelity per copy F(n)/n', fontsize=12)
    ax.set_title('Copy Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_min_copies_heatmap(min_copies_data: dict, filename: str):
    """Heatmap of minimum copies to reach F>=0.95 and F>=0.99."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (threshold, thresh_label) in enumerate([(0.95, 'F ≥ 0.95'), (0.99, 'F ≥ 0.99')]):
        ax = axes[idx]
        strategies = [s for s in STRATEGIES if s != 'Oracle']
        scenarios = list(min_copies_data.keys())

        matrix = np.full((len(strategies), len(scenarios)), np.nan)
        for j, scenario in enumerate(scenarios):
            for i, strat in enumerate(strategies):
                val = min_copies_data[scenario].get(strat, {}).get(threshold)
                if val is not None:
                    matrix[i, j] = val

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                        vmin=1, vmax=200)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=10)
        ax.set_title(f'Min copies for {thresh_label}', fontsize=13, fontweight='bold')

        # Annotate cells
        for i in range(len(strategies)):
            for j in range(len(scenarios)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, '>200', ha='center', va='center',
                            fontsize=9, color='red', fontweight='bold')
                else:
                    ax.text(j, i, f'{int(val)}', ha='center', va='center',
                            fontsize=9, color='white' if val > 100 else 'black')

        fig.colorbar(im, ax=ax, label='n_copies', shrink=0.8)

    plt.suptitle('Copy Budget Required for Target Fidelity (Blind CQEC)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_scaling_exponents(scaling_data: dict, filename: str):
    """Bar chart of scaling exponents alpha for F(n) ~ 1 - A*n^{-alpha}."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = list(scaling_data.keys())
    strategies = [s for s in STRATEGIES if s != 'Oracle']
    x = np.arange(len(scenarios))
    width = 0.15

    for i, strat in enumerate(strategies):
        alphas = []
        for scenario in scenarios:
            params = scaling_data[scenario].get(strat)
            if params is not None:
                alphas.append(params[1])  # alpha
            else:
                alphas.append(0)
        bars = ax.bar(x + i * width, alphas, width, label=strat,
                      color=COLORS.get(strat, '#333'), alpha=0.85)

    ax.set_xlabel('Noise scenario', fontsize=12)
    ax.set_ylabel('Scaling exponent α', fontsize=12)
    ax.set_title('Copy Scaling: F(n) ~ 1 − A·n^{−α}  (higher α = faster convergence)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
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
    print(" Copy-Economized Blind CQEC Benchmark")
    print("=" * 70)

    # Copy values to sweep: 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200
    copy_values = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200])

    # ---- Define noise scenarios ----
    energies_q = np.array([0.0, 1.0])
    target_q = maximally_coherent_state(2)

    energies_qt = np.array([0.0, 1.0, 2.0])
    psi_qt = np.array([1, np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 3)]) / np.sqrt(3)
    target_qt = np.outer(psi_qt, psi_qt.conj())

    scenarios = {
        'Deph γ=1 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: partial_dephasing(rho, energies_q, 1.0),
            'noise_param': 1.0, 'noise_type': 'dephasing',
        },
        'Deph γ=3 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: partial_dephasing(rho, energies_q, 3.0),
            'noise_param': 3.0, 'noise_type': 'dephasing',
        },
        'Depol p=0.5 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: depolarizing_channel(rho, 0.5),
            'noise_param': 0.5, 'noise_type': 'depolarizing',
        },
        'Depol p=0.8 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: depolarizing_channel(rho, 0.8),
            'noise_param': 0.8, 'noise_type': 'depolarizing',
        },
        'Amp γ=0.3 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: amplitude_damping(rho, 0.3),
            'noise_param': 0.3, 'noise_type': 'amplitude_damping',
        },
        'Amp γ=0.7 (qubit)': {
            'target': target_q, 'energies': energies_q,
            'noise_fn': lambda rho: amplitude_damping(rho, 0.7),
            'noise_param': 0.7, 'noise_type': 'amplitude_damping',
        },
        'Deph γ=1 (qutrit)': {
            'target': target_qt, 'energies': energies_qt,
            'noise_fn': lambda rho: partial_dephasing(rho, energies_qt, 1.0),
            'noise_param': 1.0, 'noise_type': 'dephasing',
        },
        'Deph γ=3 (qutrit)': {
            'target': target_qt, 'energies': energies_qt,
            'noise_fn': lambda rho: partial_dephasing(rho, energies_qt, 3.0),
            'noise_param': 3.0, 'noise_type': 'dephasing',
        },
    }

    all_results = {}
    min_copies_data = {}
    scaling_data = {}

    for sname, cfg in scenarios.items():
        print(f"\n[Running] {sname} ...")
        res = sweep_copies(
            cfg['target'], cfg['energies'], cfg['noise_fn'],
            cfg['noise_param'], cfg['noise_type'], copy_values,
        )
        all_results[sname] = res

        # Plot individual sweep
        safe_name = sname.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        plot_copy_sweep(res, copy_values, f'Copy Sweep: {sname}',
                        os.path.join(out_dir, f'fig_copies_{safe_name}.png'))

        # Compute min copies and scaling
        min_copies_data[sname] = {}
        scaling_data[sname] = {}
        for strat_name in STRATEGIES:
            if strat_name == 'Oracle':
                continue
            fids = res[strat_name]['fidelity']
            min_copies_data[sname][strat_name] = {
                0.95: find_min_copies(fids, copy_values, 0.95),
                0.99: find_min_copies(fids, copy_values, 0.99),
            }
            scaling_data[sname][strat_name] = fit_scaling(fids, copy_values)

    # ---- Summary plots ----
    print("\n[Summary] Heatmap of minimum copies...")
    plot_min_copies_heatmap(min_copies_data,
                            os.path.join(out_dir, 'fig_copies_min_heatmap.png'))

    print("[Summary] Scaling exponents...")
    plot_scaling_exponents(scaling_data,
                           os.path.join(out_dir, 'fig_copies_scaling_exponents.png'))

    # ---- Numerical table ----
    print("\n" + "=" * 90)
    print(" Minimum copies for F >= threshold (blind CQEC)")
    print("=" * 90)
    strategies_list = [s for s in STRATEGIES if s != 'Oracle']
    header = f"  {'Scenario':<25s}"
    for s in strategies_list:
        header += f"  {s:<18s}"
    print(header)
    print(f"  {'-'*25}" + f"  {'-'*18}" * len(strategies_list))

    for threshold in [0.95, 0.99]:
        print(f"\n  --- F >= {threshold} ---")
        for sname in scenarios:
            row = f"  {sname:<25s}"
            for strat in strategies_list:
                val = min_copies_data[sname][strat][threshold]
                row += f"  {str(val) if val else '>200':>18s}"
            print(row)

    # Scaling exponents
    print(f"\n{'='*90}")
    print(" Scaling exponents: F(n) ~ 1 - A * n^(-alpha)")
    print("=" * 90)
    header = f"  {'Scenario':<25s}"
    for s in strategies_list:
        header += f"  {s:<18s}"
    print(header)
    print(f"  {'-'*25}" + f"  {'-'*18}" * len(strategies_list))
    for sname in scenarios:
        row = f"  {sname:<25s}"
        for strat in strategies_list:
            params = scaling_data[sname].get(strat)
            if params is not None:
                row += f"  α={params[1]:5.2f} A={params[0]:5.2f}"
            else:
                row += f"  {'(converged)':>18s}"
        print(row)

    print("\n[Done] Copy-economized blind CQEC benchmark complete.")

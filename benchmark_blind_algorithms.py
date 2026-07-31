#!/usr/bin/env python3
"""
benchmark_blind_algorithms.py - 4 Quantum Algorithms + Decoherence + Blind CQEC
=================================================================================

Runs the 4 quantum algorithms from the attached papers under decoherence,
then applies BLIND catalytic QEC (target state unknown) using each
estimation strategy. Measures fidelity against the true ideal state.

Algorithms:
  1. qDRIFT (Chen et al., PRX Quantum 2, 040305, 2021)
  2. QKAN (Ivashkov et al., arXiv:2410.04435, 2024)
  3. Control-Free QPE (Clinton et al., PRX Quantum 7, 010345, 2026)
  4. Regev Factoring (Regev, arXiv:2308.06572, 2024)

Estimation strategies (from benchmark_blind_cqec.py):
  - Naive, Noise-channel inversion, Coherence maximization,
    Iterative refinement, Multi-copy tomography, Oracle (known target)

Noise channels:
  - Dephasing (γ=2.0)
  - Depolarizing (p=0.3)
  - Combined (γ=1.0, p=0.15, γ_AD=0.1)
"""

import numpy as np
from scipy.linalg import expm, sqrtm
from math import pi, ceil, log2, sqrt, exp
from typing import Tuple, List, Dict
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
sys.path.insert(0, '/Users/deeptell01/Documents/alterego/personal/cqec/blind_cqec_pkg')
from core import (
    EnergySystem, trace_distance, coherence_l1,
    check_mode_inclusion, partial_dephasing,
    maximally_coherent_state,
)
from icec import ICECProtocol
from benchmark_blind_cqec import (
    state_fidelity, ensure_valid_dm,
    estimate_naive,
    estimate_coherence_max, estimate_iterative,
    estimate_multicopy,
)

# Import algorithm simulators from existing benchmark
from benchmark_icec import (
    QDRIFTSimulator, QKANSimulator, ControlFreeQPE, RegevFactoringSimulator,
    pure_state_to_density,
)

# Unified noise module (manuscript Sec. 3.2 conventions) and
# mode-inclusion-restricted recovery (Theorem 1)
from blind_cqec.noise import (
    dephasing as _pkg_dephasing,
    depolarizing as _pkg_depolarizing,
    amplitude_damping as _pkg_amplitude_damping,
)
from blind_cqec.recovery import icec_recover as pkg_icec_recover


# ============================================================
# Unified noise channels (wrappers around blind_cqec.noise)
# ============================================================

def apply_dephasing(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Energy-gap-dependent dephasing: rho_ij -> exp(-gamma*|i-j|) rho_ij."""
    return _pkg_dephasing(rho, gamma)


def apply_depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
    """Depolarizing channel: rho -> (1-p) rho + p I/d."""
    return _pkg_depolarizing(rho, p)


def apply_amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Cascaded generalized amplitude damping |k> -> |k-1>, rate gamma."""
    return _pkg_amplitude_damping(rho, gamma)


def estimate_noise_inversion(noisy: np.ndarray, energies: np.ndarray,
                             noise_config: dict = None, **kwargs) -> np.ndarray:
    """Exact analytical inversion of the applied noise, in reverse order.

    Inverts amplitude damping, then depolarizing, then gap-dependent
    dephasing -- matching the application order in apply_noise and the
    conventions of blind_cqec.estimators.estimate_channel_inversion.
    """
    d = noisy.shape[0]
    est = np.asarray(noisy, dtype=complex).copy()
    cfg = noise_config or {}
    amp_gamma = cfg.get('amp_damp', 0.0)
    depol_p = cfg.get('depol', 0.0)
    deph_gamma = cfg.get('dephasing', 0.0)

    # 1) Undo amplitude damping (applied last)
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
    if 0 < depol_p < 1:
        est = (est - depol_p * np.eye(d) / d) / (1 - depol_p)

    # 3) Undo gap-dependent dephasing
    if deph_gamma > 0:
        idx = np.arange(d)
        mask = np.exp(deph_gamma * np.abs(idx[:, None] - idx[None, :]))
        np.fill_diagonal(mask, 1.0)
        est = est * mask

    return ensure_valid_dm(est)


# ============================================================
# Blind ICEC recovery engine (generalized for arbitrary dim)
# ============================================================

def blind_icec_recover(rho_noisy: np.ndarray, rho_true_target: np.ndarray,
                       energies: np.ndarray, estimator_name: str,
                       estimator_fn, noise_config: dict,
                       noise_fn=None, n_copies: int = 100
                       ) -> Dict:
    """Apply blind ICEC: estimate target, then correct.

    The estimate is first passed through the mode-inclusion-restricted
    recovery map (blind_cqec.recovery.icec_recover, Theorem 1): coherence
    modes absent from the noisy state are zeroed, since the catalytic
    transformation cannot regenerate them.  The finite-copy ICEC protocol
    then converges toward this restricted target as n_copies grows.

    Returns dict with fidelities measured against the TRUE target.
    """
    d = rho_noisy.shape[0]

    if estimator_name == 'Oracle':
        estimated_target = rho_true_target.copy()
    else:
        estimated_target = estimator_fn(
            rho_noisy, energies,
            noise_channel=noise_fn,
            noise_config=noise_config,
            n_copies=n_copies,
            n_iterations=5,
            n_samples=20,
        )

    # Estimation fidelity
    est_fidelity = state_fidelity(estimated_target, rho_true_target)

    # Theorem 1: restrict the recovery target to modes surviving in the
    # noisy state (mode-inclusion constraint C(rho_est) within C(rho_noisy)).
    # mode_tol matches core.check_mode_inclusion's tol (1e-10) so the
    # restricted target never claims a mode the protocol considers absent.
    restricted_target = pkg_icec_recover(rho_noisy, estimated_target,
                                         mode_tol=1e-10)

    # Run ICEC toward the mode-restricted target
    protocol = ICECProtocol(energies, restricted_target)
    protocol.initialize(n_copies)
    recovered, result = protocol.correct(rho_noisy, n_copies)

    # Measure against TRUE target
    rec_fidelity = state_fidelity(recovered, rho_true_target)
    rec_trace_dist = trace_distance(recovered, rho_true_target)
    target_coh = coherence_l1(rho_true_target, energies)
    rec_coh = coherence_l1(recovered, energies)
    coh_ratio = rec_coh / max(target_coh, 1e-15)

    return {
        'estimation_fidelity': est_fidelity,
        'recovery_fidelity': rec_fidelity,
        'trace_distance': rec_trace_dist,
        'coherence_ratio': coh_ratio,
        'mode_satisfied': result.mode_condition_satisfied,
    }


# ============================================================
# Noise application (with type tracking for inversion)
# ============================================================

def apply_noise(rho, noise_config, d):
    """Apply unified noise (dephasing -> depolarizing -> amp damping).

    Returns (noisy_rho, noise_config, noise_fn).  The full config is
    passed to the channel-inversion estimator so every applied channel is
    inverted exactly, in reverse order.
    """
    energies = np.arange(d, dtype=float)
    rho_noisy = rho.copy()

    if noise_config['dephasing'] > 0:
        rho_noisy = apply_dephasing(rho_noisy, noise_config['dephasing'])

    if noise_config['depol'] > 0:
        rho_noisy = apply_depolarizing(rho_noisy, noise_config['depol'])

    if noise_config['amp_damp'] > 0:
        rho_noisy = apply_amplitude_damping_channel(rho_noisy, noise_config['amp_damp'])

    # Ensure valid
    rho_noisy = (rho_noisy + rho_noisy.conj().T) / 2
    tr = np.trace(rho_noisy).real
    if tr > 0:
        rho_noisy /= tr

    # Build noise_fn for multi-copy estimator
    nc = noise_config
    def noise_fn(rho_in, nc=nc):
        r = rho_in.copy()
        if nc['dephasing'] > 0:
            r = apply_dephasing(r, nc['dephasing'])
        if nc['depol'] > 0:
            r = apply_depolarizing(r, nc['depol'])
        if nc['amp_damp'] > 0:
            r = apply_amplitude_damping_channel(r, nc['amp_damp'])
        r = (r + r.conj().T) / 2
        t = np.trace(r).real
        if t > 0:
            r /= t
        return r

    return rho_noisy, noise_config, noise_fn


# ============================================================
# Main benchmark
# ============================================================

STRATEGIES = {
    'Naive': estimate_naive,
    'Channel inversion': estimate_noise_inversion,
    'Coherence max': estimate_coherence_max,
    'Iterative': estimate_iterative,
    'Multi-copy tomo': estimate_multicopy,
    'Oracle': None,
}

NOISE_CONFIGS = [
    {'name': 'Dephasing (γ=2.0)',
     'dephasing': 2.0, 'depol': 0.0, 'amp_damp': 0.0},
    {'name': 'Depolarizing (p=0.3)',
     'dephasing': 0.0, 'depol': 0.3, 'amp_damp': 0.0},
    {'name': 'Combined (γ=1,p=0.15,γ_AD=0.1)',
     'dephasing': 1.0, 'depol': 0.15, 'amp_damp': 0.1},
]

COLORS = {
    'No correction': '#999999',
    'Naive': '#e74c3c',
    'Channel inversion': '#3498db',
    'Coherence max': '#2ecc71',
    'Iterative': '#9b59b6',
    'Multi-copy tomo': '#e67e22',
    'Oracle': '#1a1a1a',
}


def run_all_algorithms():
    """Run all 4 quantum algorithms and return their ideal states."""
    algorithms = []

    # 1. qDRIFT
    print("  [1/4] qDRIFT (Heisenberg 3-qubit, t=1.0, N=80)...")
    try:
        qdrift = QDRIFTSimulator(n_qubits=3)
        r = qdrift.run_simulation(t=1.0, N_gates=80)
        print(f"        dim={r['dim']}, algo F={r['algorithm_fidelity']:.4f}")
        algorithms.append(r)
    except Exception as e:
        print(f"        ERROR: {e}")

    # 2. QKAN
    print("  [2/4] QKAN (CHEB-d3, sin, 4→4)...")
    try:
        qkan = QKANSimulator(n_input=4, n_output=4, cheb_degree=3)
        r = qkan.run_simulation(target_func='sin')
        print(f"        dim={r['dim']}, algo F={r['algorithm_fidelity']:.4f}")
        algorithms.append(r)
    except Exception as e:
        print(f"        ERROR: {e}")

    # 3. Control-Free QPE
    print("  [3/4] Control-Free QPE (Fermi-Hubbard 3-site, VPR)...")
    try:
        qpe = ControlFreeQPE(n_sites=3, tau=1.0, U_int=4.0)
        r = qpe.run_simulation(N_times=16, dt=0.2)
        print(f"        dim={r['dim']}, algo F={r['algorithm_fidelity']:.4f}")
        algorithms.append(r)
    except Exception as e:
        print(f"        ERROR: {e}")

    # 4. Regev Factoring
    print("  [4/4] Regev Factoring (N=15)...")
    try:
        regev = RegevFactoringSimulator(N=15)
        r = regev.run_simulation(R=4.0)
        print(f"        dim={r['dim']}, algo F={r['algorithm_fidelity']:.4f}")
        algorithms.append(r)
    except Exception as e:
        print(f"        ERROR: {e}")

    return algorithms


def run_blind_benchmark(algorithms, n_copies=100):
    """Run blind CQEC on all algorithms × noise × strategies."""

    # results[alg_name][noise_name][strategy_name] = {...}
    all_results = {}

    for alg in algorithms:
        alg_name = alg['name']
        rho_ideal = alg['rho_ideal']
        d = alg['dim']
        energies = np.arange(d, dtype=float)

        all_results[alg_name] = {}

        for noise_cfg in NOISE_CONFIGS:
            noise_name = noise_cfg['name']
            rho_noisy, ncfg, nfn = apply_noise(rho_ideal, noise_cfg, d)

            # Baseline: no correction
            f_noisy = state_fidelity(rho_noisy, rho_ideal)
            d_noisy = trace_distance(rho_noisy, rho_ideal)
            target_coh = coherence_l1(rho_ideal, energies)
            noisy_coh = coherence_l1(rho_noisy, energies)

            results_noise = {
                'No correction': {
                    'estimation_fidelity': f_noisy,
                    'recovery_fidelity': f_noisy,
                    'trace_distance': d_noisy,
                    'coherence_ratio': noisy_coh / max(target_coh, 1e-15),
                    'mode_satisfied': True,
                }
            }

            for strat_name, strat_fn in STRATEGIES.items():
                res = blind_icec_recover(
                    rho_noisy, rho_ideal, energies,
                    strat_name, strat_fn,
                    ncfg, nfn, n_copies,
                )
                results_noise[strat_name] = res

            all_results[alg_name][noise_name] = results_noise

    return all_results


# ============================================================
# Copy sweep per algorithm
# ============================================================

def run_copy_sweep(algorithms, copy_values):
    """Sweep n_copies for each algorithm at fixed combined noise."""
    noise_cfg = {'dephasing': 1.0, 'depol': 0.15, 'amp_damp': 0.1}

    # results[alg_name][strategy] = [fidelity at each n_copies]
    sweep_results = {}

    for alg in algorithms:
        alg_name = alg['name']
        rho_ideal = alg['rho_ideal']
        d = alg['dim']
        energies = np.arange(d, dtype=float)
        rho_noisy, ncfg, nfn = apply_noise(rho_ideal, noise_cfg, d)

        sweep_results[alg_name] = {s: [] for s in list(STRATEGIES.keys()) + ['No correction']}
        f_noisy = state_fidelity(rho_noisy, rho_ideal)
        sweep_results[alg_name]['No correction'] = [f_noisy] * len(copy_values)

        for nc in copy_values:
            nc = int(nc)
            for strat_name, strat_fn in STRATEGIES.items():
                res = blind_icec_recover(
                    rho_noisy, rho_ideal, energies,
                    strat_name, strat_fn,
                    ncfg, nfn, nc,
                )
                sweep_results[alg_name][strat_name].append(res['recovery_fidelity'])

    return sweep_results


# ============================================================
# Plotting
# ============================================================

def plot_main_results(all_results, filename):
    """Big summary heatmap: algorithms × noise × strategies."""
    alg_names = list(all_results.keys())
    noise_names = [n['name'] for n in NOISE_CONFIGS]
    strat_names = ['No correction'] + list(STRATEGIES.keys())

    n_alg = len(alg_names)
    n_noise = len(noise_names)
    n_strat = len(strat_names)

    fig, axes = plt.subplots(1, n_noise, figsize=(7 * n_noise, 0.8 * n_alg * n_strat / n_noise + 3))
    if n_noise == 1:
        axes = [axes]

    for ni, noise_name in enumerate(noise_names):
        ax = axes[ni]
        matrix = np.zeros((n_alg, n_strat))
        for ai, alg_name in enumerate(alg_names):
            for si, strat_name in enumerate(strat_names):
                f = all_results[alg_name][noise_name][strat_name]['recovery_fidelity']
                matrix[ai, si] = f

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(n_strat))
        ax.set_xticklabels(strat_names, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(n_alg))
        ax.set_yticklabels([a.split('(')[0].strip() for a in alg_names], fontsize=10)
        ax.set_title(noise_name, fontsize=12, fontweight='bold')

        for ai in range(n_alg):
            for si in range(n_strat):
                val = matrix[ai, si]
                color = 'white' if val < 0.5 else 'black'
                ax.text(si, ai, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        fig.colorbar(im, ax=ax, label='Fidelity', shrink=0.7)

    plt.suptitle('Blind CQEC Recovery Fidelity: 4 Algorithms × 3 Noise Models × 6 Strategies',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_bar_comparison(all_results, filename):
    """Grouped bar chart: fidelity for each algorithm, grouped by strategy."""
    alg_names = list(all_results.keys())
    strat_names = list(STRATEGIES.keys())
    noise_name = NOISE_CONFIGS[2]['name']  # Combined noise

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(alg_names))
    width = 0.12

    # No correction baseline
    baseline = [all_results[a][noise_name]['No correction']['recovery_fidelity']
                for a in alg_names]
    ax.bar(x - 3 * width, baseline, width, label='No correction',
           color=COLORS['No correction'], alpha=0.7, edgecolor='black', linewidth=0.5)

    for i, strat in enumerate(strat_names):
        vals = [all_results[a][noise_name][strat]['recovery_fidelity']
                for a in alg_names]
        ax.bar(x + (i - 2) * width, vals, width, label=strat,
               color=COLORS.get(strat, '#333'), alpha=0.85,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Recovery Fidelity (vs true ideal)', fontsize=12)
    ax.set_title(f'Blind CQEC: Combined Noise ({noise_name})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.split('(')[0].strip() for a in alg_names],
                       fontsize=10, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_copy_sweep_results(sweep_results, copy_values, filename):
    """One panel per algorithm: fidelity vs n_copies."""
    alg_names = list(sweep_results.keys())
    n_alg = len(alg_names)

    fig, axes = plt.subplots(1, n_alg, figsize=(5 * n_alg, 5))
    if n_alg == 1:
        axes = [axes]

    for i, alg_name in enumerate(alg_names):
        ax = axes[i]
        data = sweep_results[alg_name]
        for strat_name in ['No correction'] + list(STRATEGIES.keys()):
            fids = data[strat_name]
            ls = ':' if strat_name == 'No correction' else ('--' if strat_name == 'Oracle' else '-')
            lw = 2.5 if strat_name in ('Oracle', 'No correction') else 1.5
            ax.semilogx(copy_values, fids, label=strat_name,
                        color=COLORS.get(strat_name, '#333'),
                        linestyle=ls, linewidth=lw, marker='o', markersize=3)

        ax.set_xlabel('Number of copies', fontsize=11)
        ax.set_ylabel('Fidelity', fontsize=11)
        short_name = alg_name.split('(')[0].strip()
        ax.set_title(short_name, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='lower right')

    plt.suptitle('Blind CQEC Copy Scaling per Algorithm (Combined Noise)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight', dpi=150)
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_estimation_vs_recovery(all_results, filename):
    """Scatter: estimation fidelity vs recovery fidelity for all combos."""
    fig, ax = plt.subplots(figsize=(8, 7))

    markers = {'Naive': 'o', 'Channel inversion': 's', 'Coherence max': '^',
               'Iterative': 'D', 'Multi-copy tomo': 'v', 'Oracle': '*'}

    for alg_name in all_results:
        for noise_name in all_results[alg_name]:
            for strat_name in STRATEGIES:
                res = all_results[alg_name][noise_name][strat_name]
                est_f = res['estimation_fidelity']
                rec_f = res['recovery_fidelity']
                short_alg = alg_name.split('(')[0].strip()[:8]
                ax.scatter(est_f, rec_f,
                          color=COLORS.get(strat_name, '#333'),
                          marker=markers.get(strat_name, 'o'),
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, linewidth=1)

    # Legend for strategies
    for strat_name in STRATEGIES:
        ax.scatter([], [], color=COLORS.get(strat_name, '#333'),
                  marker=markers.get(strat_name, 'o'), s=60,
                  label=strat_name, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Estimation Fidelity F(estimated, true target)', fontsize=12)
    ax.set_ylabel('Recovery Fidelity F(recovered, true target)', fontsize=12)
    ax.set_title('Estimation Quality vs Recovery Quality',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
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

    print("=" * 80)
    print(" 4 Quantum Algorithms × Decoherence × Blind CQEC")
    print("=" * 80)

    # --- Step 1: Run algorithms ---
    print("\n[Step 1] Running quantum algorithms...")
    algorithms = run_all_algorithms()

    if not algorithms:
        print("ERROR: No algorithms ran successfully.")
        sys.exit(1)

    # --- Step 2: Blind CQEC across all combos ---
    print(f"\n[Step 2] Blind CQEC: {len(algorithms)} algs × "
          f"{len(NOISE_CONFIGS)} noise × {len(STRATEGIES)+1} strategies...")
    all_results = run_blind_benchmark(algorithms, n_copies=100)

    # --- Step 3: Copy sweep ---
    print("\n[Step 3] Copy sweep (combined noise)...")
    copy_values = np.array([2, 5, 10, 20, 50, 100, 200])
    sweep_results = run_copy_sweep(algorithms, copy_values)

    # --- Step 4: Plots ---
    print("\n[Step 4] Generating plots...")
    plot_main_results(all_results,
                      os.path.join(out_dir, 'fig_blind_alg_heatmap.png'))
    plot_bar_comparison(all_results,
                        os.path.join(out_dir, 'fig_blind_alg_bars.png'))
    plot_copy_sweep_results(sweep_results, copy_values,
                            os.path.join(out_dir, 'fig_blind_alg_copies.png'))
    plot_estimation_vs_recovery(all_results,
                                os.path.join(out_dir, 'fig_blind_alg_scatter.png'))

    # --- Step 5: Numerical summary ---
    print("\n" + "=" * 80)
    print(" NUMERICAL RESULTS")
    print("=" * 80)

    for alg_name in all_results:
        print(f"\n{'━' * 75}")
        print(f"  {alg_name}")
        print(f"{'━' * 75}")
        print(f"  {'Noise':<35s} {'Strategy':<20s} {'Est F':>7s} {'Rec F':>7s} "
              f"{'T.Dist':>7s} {'Coh%':>6s}")
        print(f"  {'-'*35} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

        for noise_name in all_results[alg_name]:
            results = all_results[alg_name][noise_name]
            for strat_name in ['No correction'] + list(STRATEGIES.keys()):
                r = results[strat_name]
                est_f = r.get('estimation_fidelity', 0)
                rec_f = r['recovery_fidelity']
                td = r['trace_distance']
                cr = r['coherence_ratio']
                print(f"  {noise_name:<35s} {strat_name:<20s} "
                      f"{est_f:7.4f} {rec_f:7.4f} {td:7.4f} {cr:6.2f}")
            print()

    # --- Copy sweep summary ---
    print(f"\n{'=' * 80}")
    print(" COPY SWEEP (Combined Noise)")
    print(f"{'=' * 80}")
    for alg_name in sweep_results:
        print(f"\n  {alg_name}")
        header = f"  {'n_copies':>10s}"
        for s in list(STRATEGIES.keys()):
            header += f"  {s:>16s}"
        print(header)
        for ci, nc in enumerate(copy_values):
            row = f"  {int(nc):>10d}"
            for s in STRATEGIES:
                f = sweep_results[alg_name][s][ci]
                row += f"  {f:>16.4f}"
            print(row)

    # --- Step 6: Estimation-recovery correlation statistics ---
    print(f"\n{'=' * 80}")
    print(" ESTIMATION vs RECOVERY CORRELATION (84 points: 4 alg x 3 noise x 7 strat)")
    print(f"{'=' * 80}")

    est_all, rec_all, dims_all = [], [], []
    for alg in algorithms:
        alg_name = alg['name']
        d = alg['dim']
        for noise_name in all_results[alg_name]:
            for strat_name in ['No correction'] + list(STRATEGIES.keys()):
                r = all_results[alg_name][noise_name][strat_name]
                est_all.append(r['estimation_fidelity'])
                rec_all.append(r['recovery_fidelity'])
                dims_all.append(d)
    est_all = np.array(est_all)
    rec_all = np.array(rec_all)
    dims_all = np.array(dims_all)

    n_pts = len(est_all)
    r_pooled = np.corrcoef(est_all, rec_all)[0, 1]
    slope, intercept = np.polyfit(est_all, rec_all, 1)
    pred = slope * est_all + intercept
    ss_res = np.sum((rec_all - pred) ** 2)
    ss_tot = np.sum((rec_all - np.mean(rec_all)) ** 2)
    r2 = 1 - ss_res / ss_tot
    max_dev = np.max(np.abs(rec_all - est_all))

    print(f"  N points          : {n_pts}")
    print(f"  Pooled Pearson r  : {r_pooled:.4f}")
    print(f"  Linear fit        : F_rec = {slope:.4f} * F_est + {intercept:+.4f}")
    print(f"  R^2               : {r2:.4f}")
    print(f"  max |F_rec-F_est| : {max_dev:.4f}")

    print("\n  Per-dimension Pearson r:")
    for d in sorted(set(dims_all)):
        m = dims_all == d
        r_d = np.corrcoef(est_all[m], rec_all[m])[0, 1]
        print(f"    d={d:>3d} (n={np.sum(m):>2d}): r = {r_d:.4f}")

    # --- Step 7: Minimum copies for F >= 0.95 ---
    print(f"\n{'=' * 80}")
    print(" MIN COPIES FOR F_rec >= 0.95 (Combined Noise; '-' = not reached at n<=200)")
    print(f"{'=' * 80}")
    header = f"  {'Strategy':<20s}"
    for alg_name in sweep_results:
        header += f" {alg_name.split('(')[0].strip():>12s}"
    print(header)
    for strat_name in STRATEGIES:
        row = f"  {strat_name:<20s}"
        for alg_name in sweep_results:
            fids = np.array(sweep_results[alg_name][strat_name])
            above = np.where(fids >= 0.95)[0]
            cell = str(int(copy_values[above[0]])) if len(above) else '-'
            row += f" {cell:>12s}"
        print(row)

    print("\n[Done] All benchmarks complete.")

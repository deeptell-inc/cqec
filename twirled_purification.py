#!/usr/bin/env python3
"""
twirled_purification.py — Randomized compiling + swap test purification.

Convert dephasing noise to depolarizing via Clifford twirling,
then apply recursive swap test on the twirled (depolarized) copies.

Strategy:
  1. Apply random Clifford C_i to ideal state before noise
  2. Dephasing acts: N(C_i |+⟩⟨+| C_i†)
  3. Apply C_i† after noise: C_i† N(C_i ρ C_i†) C_i
  4. Average over N_twirl Cliffords → effective depolarizing
  5. Apply recursive swap test to depolarized copies

For density-matrix simulation, we can compute the twirled channel
analytically or by Monte Carlo sampling of Cliffords.
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from make_algorithm_states import make_qkan, make_qdrift, make_cfqpe, make_regev

# ============================================================
# Utilities
# ============================================================

def fidelity(rho, sigma):
    e, v = np.linalg.eigh(rho)
    sr = v @ np.diag(np.sqrt(np.maximum(e, 0))) @ v.conj().T
    M = sr @ sigma @ sr
    e2 = np.linalg.eigvalsh(M)
    return float(np.sum(np.sqrt(np.maximum(e2, 0))) ** 2)

def l1_coherence(rho):
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def dephasing_channel(rho, gamma):
    d = rho.shape[0]
    mask = np.exp(-gamma * np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]))
    return rho * mask

def depolarizing_channel(rho, p):
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d) / d


# ============================================================
# Clifford twirling (exact for small d, sampled for large d)
# ============================================================

def generate_single_qubit_cliffords():
    """Generate all 24 single-qubit Clifford gates."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    # Generate Clifford group by composition
    generators = [I, X, Y, Z, H, S]
    cliffords = set()
    queue = [I]
    seen_keys = set()

    def mat_key(m):
        # Normalize global phase
        idx = np.argmax(np.abs(m.ravel()))
        phase = m.ravel()[idx] / np.abs(m.ravel()[idx])
        m_norm = m / phase
        return tuple(np.round(m_norm.ravel().real, 6)) + tuple(np.round(m_norm.ravel().imag, 6))

    for _ in range(5):  # iterate to close the group
        new_queue = []
        for c in queue:
            for g in generators:
                prod = g @ c
                key = mat_key(prod)
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_queue.append(prod)
        queue = new_queue

    # Convert to list
    result = [I]  # start with identity
    for key_data in seen_keys:
        n = len(key_data) // 2
        re = np.array(key_data[:n])
        im = np.array(key_data[n:])
        m = (re + 1j * im).reshape(2, 2)
        result.append(m)

    # Deduplicate by checking unitary equivalence up to phase
    unique = []
    for m in result:
        is_dup = False
        for u in unique:
            diff = m @ u.conj().T
            if np.abs(np.abs(np.trace(diff)) - 2) < 1e-4:
                is_dup = True
                break
        if not is_dup:
            unique.append(m)

    return unique


def twirl_channel_exact(rho_ideal, noise_fn, d):
    """
    Exact Clifford twirling for multi-qubit system.
    For n qubits, apply tensor products of single-qubit Cliffords.

    Twirled channel: (1/|G|) Σ_C C† N(C ρ C†) C
    """
    n_qubits = int(np.log2(d))
    cliffords_1q = generate_single_qubit_cliffords()
    n_cliff_1q = len(cliffords_1q)

    # For small systems, enumerate all tensor products
    # For n_q qubits: |G| = 24^n_q
    if n_qubits <= 3:  # 24^3 = 13824, feasible
        rho_twirled = np.zeros((d, d), dtype=complex)
        count = 0

        # Sample instead of full enumeration for n_q >= 2
        if n_qubits == 1:
            for c in cliffords_1q:
                rotated = c @ rho_ideal @ c.conj().T
                noisy = noise_fn(rotated)
                unrotated = c.conj().T @ noisy @ c
                rho_twirled += unrotated
                count += 1
        else:
            # Monte Carlo sampling for multi-qubit
            rng = np.random.default_rng(42)
            n_samples = 2000
            for _ in range(n_samples):
                # Random tensor product of single-qubit Cliffords
                indices = rng.integers(0, n_cliff_1q, size=n_qubits)
                C = cliffords_1q[indices[0]]
                for q in range(1, n_qubits):
                    C = np.kron(C, cliffords_1q[indices[q]])

                rotated = C @ rho_ideal @ C.conj().T
                noisy = noise_fn(rotated)
                unrotated = C.conj().T @ noisy @ C
                rho_twirled += unrotated
                count += 1

        rho_twirled /= count
        return rho_twirled
    else:
        # Large systems: Monte Carlo
        rng = np.random.default_rng(42)
        n_samples = 5000
        rho_twirled = np.zeros((d, d), dtype=complex)

        for _ in range(n_samples):
            indices = rng.integers(0, n_cliff_1q, size=n_qubits)
            C = cliffords_1q[indices[0]]
            for q in range(1, n_qubits):
                C = np.kron(C, cliffords_1q[indices[q]])

            rotated = C @ rho_ideal @ C.conj().T
            noisy = noise_fn(rotated)
            unrotated = C.conj().T @ noisy @ C
            rho_twirled += unrotated

        rho_twirled /= n_samples
        return rho_twirled


def twirl_analytical(rho_ideal, gamma, d):
    """
    Analytical twirling result for uniform dephasing.

    Uniform dephasing: N(ρ)_{ij} = ρ_{ij} e^{-γ(1-δ_{ij})}
    = e^{-γ} ρ + (1 - e^{-γ}) diag(ρ)

    Under full unitary twirling, this becomes depolarizing:
    N_twirl(ρ) = (1 - p_eff) ρ + p_eff I/d

    where p_eff = (1 - e^{-γ}) (d/(d+1))... actually let me compute this.

    The dephasing channel can be written as:
    N(ρ) = e^{-γ} ρ + (1-e^{-γ}) Σ_i |i⟩⟨i| ρ |i⟩⟨i|

    Under unitary 2-design twirling (Clifford = unitary 2-design):
    Twirl of the identity part: e^{-γ} ρ → e^{-γ} ρ
    Twirl of the measure-and-prepare part:
      (1/|G|) Σ_C C† (Σ_i |i⟩⟨i| C ρ C† |i⟩⟨i|) C

    By the 2-design property:
    Σ_i ∫ U† |i⟩⟨i| U ρ U† |i⟩⟨i| U dU = (Tr(ρ) I + ρ) / (d(d+1)) × d
    ... this is getting complicated. Let me just use the known result.

    For a channel N with Choi matrix J_N, the twirled channel has:
    depolarizing parameter p_eff = 1 - (d·F_avg - 1)/(d-1)
    where F_avg = average gate fidelity of N.

    For dephasing: F_avg = (d + (d-1)e^{-γ}) / (d + d(d-1)/... )

    Actually, average fidelity of dephasing channel:
    F_avg = ∫ ⟨ψ| N(|ψ⟩⟨ψ|) |ψ⟩ dψ

    For uniform dephasing N(ρ)_{ij} = ρ_{ij} e^{-γ(1-δ_{ij})}:
    F_avg = ∫ (Σ_i |c_i|^4 + e^{-γ} Σ_{i≠j} |c_i|^2 |c_j|^2) dψ
          = (2/(d(d+1))) + e^{-γ} (1 - 2/(d(d+1)))
          = e^{-γ} + (1 - e^{-γ}) · 2/(d(d+1))

    Then p_eff = 1 - (d·F_avg - 1)/(d-1)
               = 1 - (d(e^{-γ} + (1-e^{-γ})·2/(d(d+1))) - 1)/(d-1)
    """
    e_g = np.exp(-gamma)
    F_avg = e_g + (1 - e_g) * 2 / (d * (d + 1))
    p_eff = 1 - (d * F_avg - 1) / (d - 1)
    # Sanity: for γ=0, F_avg=1, p_eff=0 ✓
    # For γ→∞, F_avg = 2/(d(d+1)), p_eff = 1 - (2d/(d(d+1)) - 1)/(d-1) = 1 - (2/(d+1) - 1)/(d-1)
    return depolarizing_channel(rho_ideal, p_eff), p_eff


# ============================================================
# Swap test (correct (I+SWAP)/2)
# ============================================================

def swap_test(rho, sigma, d):
    d2 = d * d
    rs = np.kron(rho, sigma)
    S = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    Pi = (np.eye(d2) + S) / 2.0
    proj = Pi @ rs @ Pi
    p = float(np.real(np.trace(proj)))
    if p < 1e-15:
        return rho.copy(), 0.0
    proj /= p
    out = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[i, j] += proj[i * d + k, j * d + k]
    return out, p


def recursive_swap(rho_noisy, d, n_rounds):
    rho = rho_noisy.copy()
    pt = 1.0
    for _ in range(n_rounds):
        rho, p = swap_test(rho, rho, d)
        pt *= p
    return rho, 2**n_rounds, pt


# ============================================================
# CQEC recovery
# ============================================================

def cqec_recovery(rho_target, rho_noisy, rho_cat):
    d = rho_target.shape[0]
    rho_rec = rho_noisy.copy()
    pur = purity(rho_cat)
    for i in range(d):
        for j in range(i + 1, d):
            if np.abs(rho_target[i, j]) > 1e-10 and np.abs(rho_cat[i, j]) > 1e-10:
                cc = np.abs(rho_cat[i, j])
                nc = np.abs(rho_noisy[i, j])
                if nc > 1e-15:
                    ph = np.angle(rho_target[i, j])
                    mt = np.abs(rho_target[i, j])
                    eff = 1.0 - np.exp(-cc * d * pur)
                    mr = nc + eff * (mt - nc)
                    rho_rec[i, j] = mr * np.exp(1j * ph)
                    rho_rec[j, i] = rho_rec[i, j].conj()
    e, v = np.linalg.eigh(rho_rec)
    e = np.maximum(e, 0)
    rho_rec = v @ np.diag(e) @ v.conj().T
    rho_rec /= np.trace(rho_rec)
    return rho_rec


# ============================================================
# Benchmark
# ============================================================

def make_cat(d):
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


def run_benchmark():
    print("=" * 90)
    print("Twirled Purification Benchmark")
    print("=" * 90)

    algorithms = {
        'QKAN': make_qkan(),
        'qDRIFT': make_qdrift(seed=42),
        'CF-QPE': make_cfqpe(),
        'Regev': make_regev(),
    }

    gamma = 2.0
    max_rounds = {4: 6, 8: 6, 16: 5, 64: 3}

    all_results = {}

    for alg, (rho_target, d) in algorithms.items():
        print(f"\n{'#' * 70}")
        print(f"# {alg} (d={d}, {int(np.log2(d))} qubits)")
        print(f"{'#' * 70}")

        rho_cat_ideal = make_cat(d)
        rho_noisy_target = dephasing_channel(rho_target, gamma)
        f_noisy = fidelity(rho_target, rho_noisy_target)

        # 1. Raw dephased catalyst
        rho_cat_deph = dephasing_channel(rho_cat_ideal, gamma)
        f_deph = fidelity(rho_cat_ideal, rho_cat_deph)

        # 2. Analytically twirled catalyst (dephasing → depolarizing)
        rho_cat_twirl, p_eff = twirl_analytical(rho_cat_ideal, gamma, d)
        f_twirl = fidelity(rho_cat_ideal, rho_cat_twirl)

        # 3. Monte Carlo twirled catalyst (verify analytical)
        t0 = time.time()
        noise_fn = lambda rho: dephasing_channel(rho, gamma)
        rho_cat_mc = twirl_channel_exact(rho_cat_ideal, noise_fn, d)
        f_mc = fidelity(rho_cat_ideal, rho_cat_mc)
        dt_mc = time.time() - t0

        print(f"  F_noisy(target) = {f_noisy:.4f}")
        print(f"  Catalyst:")
        print(f"    Raw dephased:  F={f_deph:.4f}, C_l1={l1_coherence(rho_cat_deph):.3f}, pur={purity(rho_cat_deph):.4f}")
        print(f"    Analytical tw: F={f_twirl:.4f}, C_l1={l1_coherence(rho_cat_twirl):.3f}, p_eff={p_eff:.4f}")
        print(f"    MC twirled:    F={f_mc:.4f}, C_l1={l1_coherence(rho_cat_mc):.3f} ({dt_mc:.1f}s)")

        # Recursive swap test: raw dephased vs twirled
        mr = max_rounds[d]
        rounds = list(range(1, mr + 1))

        print(f"\n  {'Method':<20} {'n':>4} {'F_cat':>8} {'C_l1':>8} {'F_rec':>8}")
        print(f"  {'-' * 52}")

        alg_res = {'dephased': [], 'twirled': []}

        for n_r in rounds:
            nc = 2**n_r

            # Dephased → swap test
            rho_s, _, _ = recursive_swap(rho_cat_deph, d, n_r)
            fs = fidelity(rho_cat_ideal, rho_s)
            cs = l1_coherence(rho_s)
            rho_rec_s = cqec_recovery(rho_target, rho_noisy_target, rho_s)
            frs = fidelity(rho_target, rho_rec_s)
            alg_res['dephased'].append({'n': nc, 'fc': fs, 'cl': cs, 'fr': frs})
            print(f"  {'Raw dephased':<20} {nc:>4} {fs:>8.4f} {cs:>8.3f} {frs:>8.4f}")

            # Twirled → swap test
            rho_t, _, _ = recursive_swap(rho_cat_twirl, d, n_r)
            ft = fidelity(rho_cat_ideal, rho_t)
            ct = l1_coherence(rho_t)
            rho_rec_t = cqec_recovery(rho_target, rho_noisy_target, rho_t)
            frt = fidelity(rho_target, rho_rec_t)
            alg_res['twirled'].append({'n': nc, 'fc': ft, 'cl': ct, 'fr': frt})
            print(f"  {'Twirled (RC)':<20} {nc:>4} {ft:>8.4f} {ct:>8.3f} {frt:>8.4f}")
            print()

        all_results[alg] = (d, p_eff, alg_res)

    return all_results


def plot_results(all_results):
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 13, 'axes.titlesize': 13,
        'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    alg_names = list(all_results.keys())
    n_alg = len(alg_names)

    fig, axes = plt.subplots(2, n_alg, figsize=(5 * n_alg, 9))

    for ai, alg in enumerate(alg_names):
        d, p_eff, ar = all_results[alg]

        for method, color, label in [
            ('dephased', '#e74c3c', 'Raw dephased'),
            ('twirled', '#2ecc71', f'Twirled ($p_{{eff}}={p_eff:.2f}$)'),
        ]:
            res = ar[method]
            ns = [r['n'] for r in res]
            fc = [r['fc'] for r in res]
            fr = [r['fr'] for r in res]
            axes[0, ai].semilogx(ns, fc, 'o-', color=color, lw=2, ms=5, label=label)
            axes[1, ai].semilogx(ns, fr, 's-', color=color, lw=2, ms=5, label=label)

        axes[0, ai].set_title(f'{alg} ($d={d}$)')
        axes[0, ai].set_ylabel(r'$F_{\mathrm{cat}}$')
        axes[0, ai].set_xlabel(r'Copies $n$')
        axes[0, ai].legend(fontsize=7)
        axes[0, ai].axhline(y=0.95, color='gray', ls=':', alpha=0.4)
        axes[0, ai].axhline(y=0.99, color='gray', ls='--', alpha=0.3)

        axes[1, ai].set_ylabel(r'$F_{\mathrm{rec}}$')
        axes[1, ai].set_xlabel(r'Copies $n$')
        axes[1, ai].legend(fontsize=7)

    plt.suptitle(r'Randomized Compiling: Twirled vs Raw Dephased ($\gamma=2$)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_twirled_purification.png')
    plt.savefig('fig_twirled_purification.pdf')
    print("\nSaved fig_twirled_purification.png/pdf")
    plt.close()


if __name__ == '__main__':
    results = run_benchmark()
    plot_results(results)

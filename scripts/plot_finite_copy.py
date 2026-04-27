#!/usr/bin/env python3
"""
plot_finite_copy.py - Finite-copy fidelity bounds for CQEC
Visualizes: F(n, d) when copy number n is insufficient for perfect fidelity.
Based on Eq.(15-16) of the paper:
  ||ρ_out - ρ_0||_1 ≤ C/√n
  1 - F ≤ C²/(4n) + C/√n
where C ≈ d · C_ℓ1(ρ_0) · e^γ / |ρ_min|
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Algorithm parameters from the paper (Table IV, γ = 2.0)
algorithms = {
    'QKAN':      {'d': 4,  'C_l1': 2.1,  'rho_min': 0.18,   'C': 8.5,   'color': '#3498db'},
    'qDRIFT':    {'d': 8,  'C_l1': 3.2,  'rho_min': 0.021,  'C': 42,    'color': '#e74c3c'},
    'CF-QPE':    {'d': 16, 'C_l1': 8.7,  'rho_min': 0.0083, 'C': 170,   'color': '#2ecc71'},
    'Regev':     {'d': 64, 'C_l1': 47,   'rho_min': 0.0012, 'C': 2700,  'color': '#9b59b6'},
}

gamma_default = 2.0

def fidelity_bound(n, C):
    """Lower bound on fidelity: F ≥ 1 - C²/(4n) - C/√n"""
    infidelity = C**2 / (4 * n) + C / np.sqrt(n)
    return np.maximum(1.0 - infidelity, 0.0)

def compute_C(d, C_l1, rho_min, gamma):
    """C ≈ d · C_ℓ1 · e^γ / |ρ_min|"""
    return d * C_l1 * np.exp(gamma) / rho_min


# ============================================================
# Figure: 3-panel layout
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

# ----------------------------------------------------------
# Panel (a): Fidelity vs copy number n (log scale), γ = 2.0
# ----------------------------------------------------------
n_vals = np.logspace(1, 14, 500)

for name, p in algorithms.items():
    C = p['C']
    F = fidelity_bound(n_vals, C)
    ax1.semilogx(n_vals, F, '-', color=p['color'], linewidth=2,
                 label=rf'{name} ($d={p["d"]}$, $C={C}$)')

# Target fidelity lines
ax1.axhline(y=0.999, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax1.text(2e13, 0.9995, r'$F = 0.999$', fontsize=8, color='gray', ha='right')
ax1.text(2e13, 0.9905, r'$F = 0.99$', fontsize=8, color='gray', ha='right')

# Mark n* values from Table IV
for name, p in algorithms.items():
    C = p['C']
    # n* for F >= 0.99
    n_star_99 = C**2 / (4 * 0.01**2)
    ax1.plot(n_star_99, 0.99, 'v', color=p['color'], markersize=8, zorder=5)

ax1.set_xlabel(r'Number of noisy copies $n$')
ax1.set_ylabel(r'Fidelity lower bound $F_{\min}(n)$')
ax1.set_title(r'(a) $F_{\min}$ vs. copy number $n$ ($\gamma = 2.0$)')
ax1.legend(loc='lower right', fontsize=8)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(10, 1e14)

# ----------------------------------------------------------
# Panel (b): Fidelity vs dimension d, at fixed n
# ----------------------------------------------------------
d_vals = np.arange(2, 129)
n_fixed_vals = [1e3, 1e5, 1e7, 1e9]
n_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

# Use generic coherence parameters scaled by d
# C_l1 ~ d (random state), |ρ_min| ~ 1/d² (worst case)
# → C ~ d · d · e^γ · d² = d⁴ e^γ  (very aggressive)
# More realistic: C_l1 ~ √d, |ρ_min| ~ 1/d → C ~ d^(5/2) e^γ
# Use the empirical fit from our 4 data points instead

# Fit log(C) vs log(d) from paper data
ds = np.array([4, 8, 16, 64])
Cs = np.array([8.5, 42, 170, 2700])
coeffs = np.polyfit(np.log(ds), np.log(Cs), 1)
# C(d) ≈ a * d^b
b_exp = coeffs[0]
a_coeff = np.exp(coeffs[1])

for ni, n_fix in enumerate(n_fixed_vals):
    C_d = a_coeff * d_vals**b_exp
    F_d = fidelity_bound(n_fix, C_d)
    ax2.plot(d_vals, F_d, '-', color=n_colors[ni], linewidth=2,
             label=rf'$n = 10^{int(np.log10(n_fix))}$')

ax2.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax2.text(125, 0.995, r'$F = 0.99$', fontsize=8, color='gray', ha='right')

# Mark the 4 algorithms
for name, p in algorithms.items():
    ax2.plot(p['d'], 0, 'D', color=p['color'], markersize=6, zorder=5, clip_on=False)
    ax2.text(p['d'], -0.08, name, fontsize=7, ha='center', color=p['color'])

ax2.set_xlabel(r'Hilbert space dimension $d$ (qubits: $n_q = \log_2 d$)')
ax2.set_ylabel(r'Fidelity lower bound $F_{\min}(n, d)$')
ax2.set_title(r'(b) $F_{\min}$ vs. dimension $d$ ($\gamma = 2.0$)')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_ylim(-0.15, 1.05)
ax2.set_xlim(2, 128)

# Add top axis for qubit count
ax2_top = ax2.twiny()
qubit_ticks = [1, 2, 3, 4, 5, 6, 7]
ax2_top.set_xlim(np.log2(2), np.log2(128))
ax2_top.set_xticks(qubit_ticks)
ax2_top.set_xticklabels([str(q) for q in qubit_ticks])
ax2_top.set_xlabel(r'Number of qubits $n_q = \log_2 d$', fontsize=10)

# ----------------------------------------------------------
# Panel (c): Fidelity vs dephasing strength γ, at fixed n
# ----------------------------------------------------------
gamma_vals = np.linspace(0.01, 5, 200)

for name, p in algorithms.items():
    d = p['d']
    C_l1 = p['C_l1']
    rho_min = p['rho_min']

    # C(γ) = d · C_l1 · e^γ / |ρ_min|
    C_gamma = d * C_l1 * np.exp(gamma_vals) / rho_min

    # Use n = 10^6 as representative
    n_rep = 1e6
    F_gamma = fidelity_bound(n_rep, C_gamma)

    ax3.plot(gamma_vals, F_gamma, '-', color=p['color'], linewidth=2,
             label=rf'{name} ($d={d}$)')

ax3.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
ax3.axhline(y=0.0, color='black', linestyle='-', alpha=0.1, linewidth=0.5)
ax3.text(4.9, 0.995, r'$F = 0.99$', fontsize=8, color='gray', ha='right')

ax3.set_xlabel(r'Dephasing strength $\gamma$')
ax3.set_ylabel(r'Fidelity lower bound $F_{\min}$')
ax3.set_title(r'(c) $F_{\min}$ vs. dephasing $\gamma$ ($n = 10^6$)')
ax3.legend(loc='upper right', fontsize=8)
ax3.set_ylim(-0.05, 1.05)

plt.suptitle(r'Finite-Copy Fidelity Bounds: $1 - F \leq C^2/(4n) + C/\sqrt{n}$',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_finite_copy_bounds.png')
plt.savefig('fig8_finite_copy_bounds.pdf')
print("Saved fig8_finite_copy_bounds.png/pdf")
plt.close()


# ============================================================
# Print summary table
# ============================================================
print("\n" + "=" * 80)
print(f"{'Algorithm':<12} {'d':>4} {'C':>8} {'n*(F≥0.99)':>14} {'n*(F≥0.999)':>14} {'n*(F≥0.9999)':>14}")
print("-" * 80)
for name, p in algorithms.items():
    C = p['C']
    # From Eq.16 dominant term: 1-F ≈ C/√n → n* ≈ C²/(1-F)²
    # Full bound: 1-F ≤ C²/(4n) + C/√n
    # For target F, solve: C²/(4n) + C/√n = 1-F
    # Let x = 1/√n: C²x²/4 + Cx = 1-F → x = (-C + √(C² + C²(1-F))) / (C²/2)
    for F_target, label in [(0.99, '0.99'), (0.999, '0.999'), (0.9999, '0.9999')]:
        delta = 1 - F_target
        # Quadratic in x = 1/√n: (C²/4)x² + Cx - δ = 0
        a_q = C**2 / 4
        b_q = C
        c_q = -delta
        x = (-b_q + np.sqrt(b_q**2 - 4*a_q*c_q)) / (2*a_q)
        n_star = 1.0 / x**2
        if label == '0.99':
            n99 = n_star
        elif label == '0.999':
            n999 = n_star
        else:
            n9999 = n_star
    print(f"{name:<12} {p['d']:>4} {C:>8.1f} {n99:>14.2e} {n999:>14.2e} {n9999:>14.2e}")

print("=" * 80)
print(f"\nEmpirical fit: C ≈ {a_coeff:.2f} × d^{b_exp:.2f} (at γ=2.0)")
print(f"Scaling: n* ~ C² ~ d^{2*b_exp:.1f} × e^(2γ)")

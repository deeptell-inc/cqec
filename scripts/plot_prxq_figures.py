#!/usr/bin/env python3
"""
plot_prxq_figures.py - Generate publication-quality figures for PRX Quantum
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

with open('results_prxq.json') as f:
    data = json.load(f)

colors = {
    'qDRIFT': '#e74c3c',
    'QKAN': '#3498db',
    'CF-QPE': '#2ecc71',
    'Regev': '#9b59b6',
    'TTN-Crypto': '#f39c12',
}
color_list = list(colors.values())

# ============================================================
# Figure 1: Noise Strength Sweep (2 panels)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel a: Dephasing sweep
ax = axes[0]
for alg_name, alg_data in data['exp1_noise_sweep'].items():
    short = alg_name.split(' ')[0]
    sweep = alg_data['dephasing_sweep']
    gammas = [r['gamma'] for r in sweep]
    fid_before = [r['fid_before'] for r in sweep]
    fid_after = [r['fid_after'] for r in sweep]
    c = colors.get(short, '#333')
    ax.plot(gammas, fid_before, '--', color=c, alpha=0.5, linewidth=1)
    ax.plot(gammas, fid_after, '-', color=c, linewidth=2, label=alg_name)

ax.set_xlabel(r'Dephasing strength $\gamma$')
ax.set_ylabel(r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
ax.set_title('(a) Dephasing channel')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.text(0.05, 0.95, 'Solid: after CQEC\nDashed: before', transform=ax.transAxes,
        fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel b: Depolarizing sweep
ax = axes[1]
for alg_name, alg_data in data['exp1_noise_sweep'].items():
    short = alg_name.split(' ')[0]
    sweep = alg_data['depolarizing_sweep']
    ps = [r['p'] for r in sweep]
    fid_before = [r['fid_before'] for r in sweep]
    fid_after = [r['fid_after'] for r in sweep]
    c = colors.get(short, '#333')
    ax.plot(ps, fid_before, '--', color=c, alpha=0.5, linewidth=1)
    ax.plot(ps, fid_after, '-', color=c, linewidth=2, label=alg_name)

ax.set_xlabel(r'Depolarizing probability $p$')
ax.set_ylabel(r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
ax.set_title('(b) Depolarizing channel')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(-0.05, 1.1)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

plt.suptitle('CQEC Recovery across Noise Strengths', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_noise_sweep.png')
plt.savefig('fig3_noise_sweep.pdf')
print("Saved fig3_noise_sweep.png/pdf")
plt.close()


# ============================================================
# Figure 2: Sharp Threshold ε-Scan
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

threshold_data = data['exp5_threshold_scan']
eps_vals = [r['epsilon'] for r in threshold_data]
fid_after = [r['fid_after'] for r in threshold_data]
success = [r['success'] for r in threshold_data]

# Separate zero and nonzero
eps_nonzero = [e for e, s in zip(eps_vals, success) if s and e > 0]
fid_nonzero = [f for f, s, e in zip(fid_after, success, eps_vals) if s and e > 0]
eps_zero = [e for e, s in zip(eps_vals, success) if not s]
fid_zero = [f for f, s in zip(fid_after, success) if not s]

ax.semilogx(eps_nonzero, fid_nonzero, 'o-', color='#2ecc71', markersize=6,
            linewidth=2, label=r'Recoverable ($\varepsilon > 0$)')
if eps_zero:
    ax.plot([1e-11], fid_zero, 'x', color='#e74c3c', markersize=12,
            markeredgewidth=3, label=r'Irrecoverable ($\varepsilon = 0$)')

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.3, label=r'$F = 1/d$ (random)')

# Annotate threshold
ax.annotate(r'Sharp threshold' '\n' r'$\varepsilon \to 0^+$: $F \to 1.0$' '\n' r'$\varepsilon = 0$: $F = 1/d$',
            xy=(1e-9, 0.95), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel(r'Residual coherence $\varepsilon$')
ax.set_ylabel(r'Fidelity after CQEC recovery')
ax.set_title(r'Sharp Zero/Nonzero Threshold of CQEC')
ax.legend(loc='center right')
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(5e-12, 1)
plt.tight_layout()
plt.savefig('fig2_threshold_scan.png')
plt.savefig('fig2_threshold_scan.pdf')
print("Saved fig2_threshold_scan.png/pdf")
plt.close()


# ============================================================
# Figure 4: QEC Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (alg_name, alg_data) in enumerate(data['exp2_qec_comparison'].items()):
    ax = axes[idx // 2][idx % 2]
    ps = [r['p'] for r in alg_data]

    ax.plot(ps, [r['fid_none'] for r in alg_data], 's--', color='gray',
            label='No correction', alpha=0.7, markersize=4)
    ax.plot(ps, [r['fid_steane'] for r in alg_data], '^-', color='#3498db',
            label='Steane [[7,1,3]]', markersize=5)
    ax.plot(ps, [r['fid_surface_d3'] for r in alg_data], 'D-', color='#f39c12',
            label='Surface d=3', markersize=5)
    ax.plot(ps, [r['fid_surface_d5'] for r in alg_data], 'v-', color='#9b59b6',
            label='Surface d=5', markersize=5)
    ax.plot(ps, [r['fid_icec'] for r in alg_data], 'o-', color='#e74c3c',
            label='CQEC', linewidth=2.5, markersize=6)

    ax.set_xlabel(r'Depolarizing probability $p$')
    ax.set_ylabel(r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
    ax.set_title(f'({chr(97+idx)}) {alg_name}')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.2)

plt.suptitle(r'Conventional QEC vs. CQEC Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_qec_comparison.png')
plt.savefig('fig4_qec_comparison.pdf')
print("Saved fig4_qec_comparison.png/pdf")
plt.close()


# ============================================================
# Figure 5: Catalyst Durability
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

cat_data = data['exp4_catalyst_durability']
cycles = [r['cycle'] for r in cat_data]
fid_before = [r['fid_before'] for r in cat_data]
fid_after = [r['fid_after'] for r in cat_data]
cat_dev = [r['catalyst_deviation'] for r in cat_data]

ax1.plot(cycles, fid_before, '-', color='#e74c3c', alpha=0.5, label=r'$F(\rho_{\mathrm{noisy}},\,\rho_0)$', linewidth=1)
ax1.plot(cycles, fid_after, '-', color='#2ecc71', linewidth=2, label=r'$F(\rho_{\mathrm{rec}},\,\rho_0)$')
ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
ax1.set_ylabel(r'Fidelity $F$')
ax1.set_title(r'(a) Recovery fidelity over 100 correction cycles')
ax1.legend()
ax1.set_ylim(0.5, 1.05)

ax2.plot(cycles, cat_dev, 'o-', color='#3498db', markersize=2)
ax2.set_xlabel(r'CQEC cycle $k$')
ax2.set_ylabel(r'Catalyst deviation $\|\rho_C^{(k)} - \rho_C^{(0)}\|_1$')
ax2.set_title(r'(b) Catalyst state integrity')
ax2.set_ylim(-0.001, 0.01)
ax2.text(0.5, 0.7, r'Catalyst deviation $= 0$' '\n' r'(perfect reuse)', transform=ax2.transAxes,
         fontsize=12, ha='center', color='#3498db',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle(r'Catalyst Durability Test ($K = 100$ cycles)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_catalyst_durability.png')
plt.savefig('fig5_catalyst_durability.pdf')
print("Saved fig5_catalyst_durability.png/pdf")
plt.close()


# ============================================================
# Figure 6: TTN Crypto + CQEC (heatmap)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ttn_data = data['exp3_ttn_crypto_icec']
nc_vals = sorted(set(r['Nc'] for r in ttn_data))
noise_names = []
seen = set()
for r in ttn_data:
    if r['noise'] not in seen:
        noise_names.append(r['noise'])
        seen.add(r['noise'])

# Before CQEC
mat_before = np.zeros((len(noise_names), len(nc_vals)))
mat_after = np.zeros((len(noise_names), len(nc_vals)))

for r in ttn_data:
    i = noise_names.index(r['noise'])
    j = nc_vals.index(r['Nc'])
    mat_before[i, j] = r['fid_before']
    mat_after[i, j] = r['fid_after']

im1 = ax1.imshow(mat_before, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(len(nc_vals)))
ax1.set_xticklabels(nc_vals)
ax1.set_yticks(range(len(noise_names)))
ax1.set_yticklabels([n.split('(')[0].strip() for n in noise_names], fontsize=8)
ax1.set_xlabel(r'Plaintext length $N_c$ (qubits)')
ax1.set_title(r'(a) Before CQEC')
plt.colorbar(im1, ax=ax1, label=r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
for i in range(len(noise_names)):
    for j in range(len(nc_vals)):
        ax1.text(j, i, f'{mat_before[i,j]:.2f}', ha='center', va='center', fontsize=7)

im2 = ax2.imshow(mat_after, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax2.set_xticks(range(len(nc_vals)))
ax2.set_xticklabels(nc_vals)
ax2.set_yticks(range(len(noise_names)))
ax2.set_yticklabels([n.split('(')[0].strip() for n in noise_names], fontsize=8)
ax2.set_xlabel(r'Plaintext length $N_c$ (qubits)')
ax2.set_title(r'(b) After CQEC')
plt.colorbar(im2, ax=ax2, label=r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
for i in range(len(noise_names)):
    for j in range(len(nc_vals)):
        ax2.text(j, i, f'{mat_after[i,j]:.2f}', ha='center', va='center', fontsize=7)

plt.suptitle(r'TTN Crypto Protocol Protection via CQEC', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_ttn_crypto_icec.png')
plt.savefig('fig6_ttn_crypto_icec.pdf')
print("Saved fig6_ttn_crypto_icec.png/pdf")
plt.close()


# ============================================================
# Figure 7: Statistical Significance (bar chart with error bars)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

noise_keys = ['Dephasing (γ=2.0)', 'Depolarizing (p=0.3)', 'Combined']
noise_display = [r'Dephasing' '\n' r'($\gamma=2.0$)', r'Depolarizing' '\n' r'($p=0.3$)', 'Combined']

for ni, noise in enumerate(noise_keys):
    ax = axes[ni]
    alg_names = list(data['exp6_statistical'].keys())
    x = np.arange(len(alg_names))
    width = 0.35

    means_before = [data['exp6_statistical'][a][noise]['fid_before_mean'] for a in alg_names]
    stds_before = [data['exp6_statistical'][a][noise]['fid_before_std'] for a in alg_names]
    means_after = [data['exp6_statistical'][a][noise]['fid_after_mean'] for a in alg_names]
    stds_after = [data['exp6_statistical'][a][noise]['fid_after_std'] for a in alg_names]

    bars1 = ax.bar(x - width/2, means_before, width, yerr=stds_before,
                   label='Before CQEC', color='#e74c3c', alpha=0.7, capsize=3)
    bars2 = ax.bar(x + width/2, means_after, width, yerr=stds_after,
                   label='After CQEC', color='#2ecc71', alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(alg_names, rotation=15, fontsize=9)
    ax.set_ylabel(r'Fidelity $F(\rho,\,\rho_{\mathrm{ideal}})$')
    ax.set_title(noise_display[ni], fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

plt.suptitle(r'CQEC Recovery with Error Bars ($N_{\mathrm{seed}} = 10$)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig7_statistical.png')
plt.savefig('fig7_statistical.pdf')
print("Saved fig7_statistical.png/pdf")
plt.close()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("All figures generated:")
print("  fig2_threshold_scan.png/pdf    - Sharp threshold")
print("  fig3_noise_sweep.png/pdf       - Noise strength sweep")
print("  fig4_qec_comparison.png/pdf    - QEC comparison")
print("  fig5_catalyst_durability.png/pdf - 100-cycle durability")
print("  fig6_ttn_crypto_icec.png/pdf   - TTN crypto protection")
print("  fig7_statistical.png/pdf       - Error bars")
print("=" * 60)

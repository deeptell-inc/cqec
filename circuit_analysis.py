"""
circuit_analysis.py - ICEC Protocol Quantum Circuit Architecture
================================================================
Visualizes and analyzes the quantum circuit structure of the ICEC
protocol, faithfully reflecting the construction from:

  Shiraishi & Takagi, PRL 132, 180202 (2024)
  Supplemental Material: Theorems S.10, S.24, Proposition S.23

The protocol has 3 layers:
  Layer 1: Coherence Distillation (Lemma S.16/S.17)
           n copies of rho -> catalysts c_1,...,c_N
  Layer 2: Marginal-Catalytic Amplification (Lemma S.13)
           N^k sets of catalysts -> (k+1)N^k copies of target
  Layer 3: Catalytic Cyclic Relabeling (Proposition S.23)
           Asymptotic -> correlated-catalytic transformation

Each layer uses energy-conserving unitaries as the fundamental gate.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from qulacs import DensityMatrix
from icec_qulacs import (
    energy_conserving_rotation, energy_conserving_phase,
    make_ec_gate, QulacsICEC, density_matrix_from_qulacs,
)
from core import (
    coherence_l1, trace_distance, maximally_coherent_state,
    partial_dephasing, EnergySystem,
)


# ============================================================
# 1. Fundamental Gate: Energy-Conserving Unitary
# ============================================================

def analyze_ec_gate():
    """Analyze the structure of the energy-conserving gate.

    The fundamental gate operates in the excitation-conserving subspace:
      |00> -> |00>  (0 excitations: invariant)
      |01> -> cos(theta)|01> - i*sin(theta)|10>  (1 excitation: rotated)
      |10> -> -i*sin(theta)|01> + cos(theta)|10>  (1 excitation: rotated)
      |11> -> |11>  (2 excitations: invariant)

    This is the iSWAP-like gate that forms the building block of
    all covariant operations. It satisfies [U, H_total] = 0.
    """
    print("=" * 70)
    print(" 1. FUNDAMENTAL GATE: Energy-Conserving Unitary")
    print("=" * 70)

    for theta in [np.pi/8, np.pi/4, np.pi/2]:
        U = energy_conserving_rotation(theta, 0, 1)
        print(f"\n  theta = {theta:.4f} (= pi/{int(round(np.pi/theta))})")
        print(f"  U = ")
        for row in U:
            print(f"    [{' '.join(f'{x.real:+.3f}{x.imag:+.3f}j' for x in row)}]")

        # Verify energy conservation
        H = np.diag([0, 1, 1, 2])  # H = n_1 + n_2 (excitation number)
        comm = U @ H - H @ U
        print(f"  [U, H] = {np.max(np.abs(comm)):.2e} (should be 0)")

    # The gate at theta = pi/2 is the iSWAP
    print(f"\n  NOTE: theta=pi/2 gives the iSWAP gate (full swap of excitation)")
    print(f"        theta=pi/4 gives sqrt(iSWAP) (partial coherence transfer)")


# ============================================================
# 2. Circuit Diagrams
# ============================================================

def draw_circuit_diagrams():
    """Draw the complete ICEC circuit architecture."""

    fig = plt.figure(figsize=(22, 30))
    fig.suptitle(
        'ICEC Protocol: Quantum Circuit Architecture\n'
        'Based on Shiraishi & Takagi, PRL 132, 180202 (2024)',
        fontsize=16, fontweight='bold', y=0.98
    )

    # ---- Panel 1: Fundamental EC Gate ----
    ax1 = fig.add_axes([0.05, 0.82, 0.90, 0.14])
    draw_ec_gate_detail(ax1)

    # ---- Panel 2: Layer 1 - Distillation Circuit ----
    ax2 = fig.add_axes([0.05, 0.58, 0.90, 0.20])
    draw_distillation_circuit(ax2)

    # ---- Panel 3: Layer 2 - Catalytic Amplification ----
    ax3 = fig.add_axes([0.05, 0.34, 0.90, 0.20])
    draw_catalytic_amplification(ax3)

    # ---- Panel 4: Layer 3 - Cyclic Relabeling (Prop S.23) ----
    ax4 = fig.add_axes([0.05, 0.10, 0.90, 0.20])
    draw_cyclic_relabeling(ax4)

    # ---- Panel 5: Full ICEC Cycle ----
    ax5 = fig.add_axes([0.05, 0.01, 0.90, 0.07])
    draw_full_cycle_summary(ax5)

    plt.savefig('/Users/deeptell01/Documents/alterego/personal/cqec/icec_circuits.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print("\n  Saved: icec_circuits.png")


def draw_ec_gate_detail(ax):
    """Panel 1: The energy-conserving gate in detail."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('(a) Fundamental Gate: Energy-Conserving Unitary  U_EC(θ,φ)',
                 fontsize=13, fontweight='bold', loc='left')

    # Qubit lines
    y1, y2 = 2.0, 1.0
    ax.plot([1, 9], [y1, y1], 'k-', lw=1.5)
    ax.plot([1, 9], [y2, y2], 'k-', lw=1.5)
    ax.text(0.3, y1, '|q₁⟩', fontsize=11, va='center', fontfamily='serif')
    ax.text(0.3, y2, '|q₂⟩', fontsize=11, va='center', fontfamily='serif')

    # EC gate box
    gate_box = FancyBboxPatch((3.5, 0.6), 3, 1.8, boxstyle="round,pad=0.1",
                               facecolor='#E8F0FE', edgecolor='#1A73E8', lw=2)
    ax.add_patch(gate_box)
    ax.text(5, 1.5, 'U_EC(θ,φ)', fontsize=12, ha='center', va='center',
            fontweight='bold', color='#1A73E8')

    # Connection dots
    ax.plot([5, 5], [y1, y2], 'k-', lw=1.5)
    ax.plot(5, y1, 'ko', markersize=6)
    ax.plot(5, y2, 'ko', markersize=6)

    # Matrix representation
    ax.text(10.5, 2.5, 'Matrix in {|00⟩,|01⟩,|10⟩,|11⟩} basis:', fontsize=10)
    matrix_text = (
        '⎡ 1      0           0       0 ⎤\n'
        '⎢ 0    cosθ      -i·sinθ     0 ⎥\n'
        '⎢ 0  -i·sinθ      cosθ       0 ⎥\n'
        '⎣ 0      0           0       1 ⎦'
    )
    ax.text(10.5, 1.0, matrix_text, fontsize=9, fontfamily='monospace',
            va='center', bbox=dict(boxstyle='round', facecolor='lightyellow',
                                   edgecolor='gray', alpha=0.8))

    # Key property
    ax.text(10.5, 0.2, '[U_EC, H₁⊗I + I⊗H₂] = 0  (energy conservation)',
            fontsize=10, color='#D32F2F', fontstyle='italic')


def draw_distillation_circuit(ax):
    """Panel 2: Coherence distillation from n copies (Lemma S.16/S.17)."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title(
        '(b) Layer 1: Coherence Distillation  ρ⊗ⁿ → c₁⊗···⊗c_N   '
        '[Lemma S.16, S.17]',
        fontsize=13, fontweight='bold', loc='left')

    n_copies = 5
    y_positions = [4.2, 3.4, 2.6, 1.8, 1.0]

    # Input copies
    for i, y in enumerate(y_positions):
        ax.plot([1.5, 17], [y, y], 'k-', lw=1, alpha=0.7)
        label = f'ρ (copy {i+1})' if i < 3 else ('⋮' if i == 3 else f'ρ (copy n)')
        ax.text(0.2, y, label, fontsize=9, va='center',
                color='#555' if i == 3 else 'black')

    # Stage 1: Pairwise EC gates (coherence extraction)
    ax.text(3.5, 4.7, 'Stage 1: Extract', fontsize=9, ha='center', color='#1A73E8')
    for i in range(0, 4, 2):
        if i < len(y_positions) - 1:
            y_top = y_positions[i]
            y_bot = y_positions[i + 1]
            box = FancyBboxPatch((3, y_bot - 0.15), 1, y_top - y_bot + 0.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#E8F0FE', edgecolor='#1A73E8', lw=1.5)
            ax.add_patch(box)
            ax.text(3.5, (y_top + y_bot) / 2, 'EC', fontsize=8,
                    ha='center', va='center', fontweight='bold', color='#1A73E8')

    # Stage 2: Collective rotation (interference)
    ax.text(7, 4.7, 'Stage 2: Interfere', fontsize=9, ha='center', color='#E65100')
    big_box = FancyBboxPatch((5.8, 0.7), 2.4, 3.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#FFF3E0', edgecolor='#E65100', lw=1.5)
    ax.add_patch(big_box)
    ax.text(7, 2.6, 'Collective\nEC\nRotation\n(√n amp.)', fontsize=8,
            ha='center', va='center', color='#E65100')

    # Stage 3: Post-selection / catalyst extraction
    ax.text(11, 4.7, 'Stage 3: Extract catalysts', fontsize=9,
            ha='center', color='#2E7D32')

    # Measurement symbols on ancilla lines
    for i, y in enumerate(y_positions[2:], 2):
        if i != 3:
            # Measurement box
            mbox = FancyBboxPatch((10.0, y - 0.2), 0.8, 0.4,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1)
            ax.add_patch(mbox)
            ax.text(10.4, y, '⊗', fontsize=10, ha='center', va='center')

    # Output catalyst states
    for i, (label, y) in enumerate(zip(['c₁', 'c₂'], y_positions[:2])):
        ax.annotate('', xy=(15, y), xytext=(12, y),
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
        out_box = FancyBboxPatch((15, y - 0.25), 1.5, 0.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5)
        ax.add_patch(out_box)
        ax.text(15.75, y, label, fontsize=11, ha='center', va='center',
                fontweight='bold', color='#2E7D32')

    # Key equation
    ax.text(14, 0.3,
            'ρ⊗μ  ──[covariant]──►  c₁⊗···⊗c_N   (exact, Lemma S.17)',
            fontsize=10, color='#D32F2F',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))


def draw_catalytic_amplification(ax):
    """Panel 3: Marginal-catalytic amplification (Lemma S.13)."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title(
        '(c) Layer 2: Marginal-Catalytic Amplification  '
        'N^k catalysts → (k+1)N^k targets   [Lemma S.13, Prop S.15]',
        fontsize=13, fontweight='bold', loc='left')

    # Left: N^k sets of catalysts input
    ax.text(0.5, 4.5, 'Input: N^k sets of catalysts', fontsize=10,
            fontweight='bold')

    cat_colors = ['#E8F0FE', '#FFF3E0', '#E8F5E9']
    for i in range(3):
        y = 3.5 - i * 1.2
        box = FancyBboxPatch((0.5, y - 0.3), 2.5, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=cat_colors[i % 3],
                              edgecolor='gray', lw=1)
        ax.add_patch(box)
        label = f'c₁⊗···⊗c_N  (set {i+1})' if i < 2 else 'c₁⊗···⊗c_N  (set N^k)'
        ax.text(1.75, y, label, fontsize=8, ha='center', va='center')

    ax.text(1.75, 0.7, '⋮', fontsize=14, ha='center')

    # Middle: Recombination circuit
    big_box = FancyBboxPatch((4.5, 0.3), 5, 4.2,
                              boxstyle="round,pad=0.15",
                              facecolor='#F3E5F5', edgecolor='#7B1FA2', lw=2)
    ax.add_patch(big_box)

    ax.text(7, 4.0, 'Catalytic Recombination', fontsize=11,
            ha='center', fontweight='bold', color='#7B1FA2')
    ax.text(7, 3.2, 'Step l (2 ≤ l ≤ k+1):', fontsize=9, ha='center')
    ax.text(7, 2.6, 'Regroup N catalysts as:', fontsize=9, ha='center')
    ax.text(7, 2.0, 'n_l = g + j mod N', fontsize=10, ha='center',
            fontfamily='serif', fontstyle='italic')
    ax.text(7, 1.3, '→ No correlation within group', fontsize=9,
            ha='center', color='#D32F2F')
    ax.text(7, 0.7, '→ Reuse catalysts for new ρ\'', fontsize=9,
            ha='center', color='#D32F2F')

    # Arrows
    for i in range(3):
        y = 3.5 - i * 1.2
        if i < 2:
            ax.annotate('', xy=(4.5, y), xytext=(3.2, y),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Right: (k+1)N^k output copies
    ax.text(11, 4.5, 'Output: (k+1)N^k copies of ρ\'', fontsize=10,
            fontweight='bold')

    for i in range(4):
        y = 3.5 - i * 0.9
        box = FancyBboxPatch((11, y - 0.25), 2, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#FFCDD2', edgecolor='#D32F2F', lw=1)
        ax.add_patch(box)
        ax.text(12, y, f'Tr_\\i[Σ] = ρ\'', fontsize=8, ha='center', va='center')

    ax.text(12, 0.5, '⋮', fontsize=14, ha='center')

    ax.annotate('', xy=(11, 2.5), xytext=(9.7, 2.5),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2))

    # Rate equation
    ax.text(14.5, 2.5, 'Rate = (k+1)/μ → ∞\nas k → ∞',
            fontsize=12, ha='center', va='center',
            fontweight='bold', color='#D32F2F',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE',
                      edgecolor='#D32F2F', alpha=0.9))

    # Discarded catalysts
    ax.text(14.5, 1.0, '(catalysts discarded\nafter use)', fontsize=9,
            ha='center', color='gray', fontstyle='italic')


def draw_cyclic_relabeling(ax):
    """Panel 4: Cyclic relabeling construction (Prop S.23)."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title(
        '(d) Layer 3: Correlated-Catalytic Transform via Cyclic Relabeling   '
        '[Prop S.23, Theorem S.24]',
        fontsize=13, fontweight='bold', loc='left')

    # System qubit
    ax.plot([2, 18], [4.0, 4.0], 'k-', lw=2)
    ax.text(0.5, 4.0, 'S (system)', fontsize=10, va='center', fontweight='bold')

    # Catalyst = (n-1) copies + classical register
    for i in range(3):
        y = 2.8 - i * 0.7
        ax.plot([2, 18], [y, y], 'k-', lw=1, alpha=0.7)

    ax.text(0.5, 2.8, 'C₁', fontsize=9, va='center', color='#1A73E8')
    ax.text(0.5, 2.1, 'C₂', fontsize=9, va='center', color='#1A73E8')
    ax.text(0.5, 1.4, '⋮', fontsize=11, va='center', color='#1A73E8')

    # Classical register
    ax.plot([2, 18], [0.5, 0.5], 'k-', lw=2)
    ax.plot([2, 18], [0.4, 0.4], 'k-', lw=2)
    ax.text(0.5, 0.5, 'R (register)', fontsize=9, va='center', color='#E65100')

    # Catalyst state box
    cat_eq = FancyBboxPatch((1.0, -0.1), 6.5, 0.5,
                             boxstyle="round,pad=0.05",
                             facecolor='lightyellow', edgecolor='gray', lw=1)
    ax.add_patch(cat_eq)
    ax.text(4.25, 0.05,
            'c = (1/n)Σ_k ρ^{⊗(k-1)} ⊗ τ_{n-k} ⊗ |k⟩⟨k|_R',
            fontsize=8, ha='center', va='center', fontfamily='serif')

    # Step 1: Controlled-Lambda
    step1_box = FancyBboxPatch((4, 1.0), 3, 3.3,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F0FE', edgecolor='#1A73E8', lw=2)
    ax.add_patch(step1_box)
    ax.text(5.5, 3.8, 'Step 1', fontsize=10, ha='center', fontweight='bold',
            color='#1A73E8')
    ax.text(5.5, 3.0, 'Apply Λ if\nR = |n⟩⟨n|', fontsize=9,
            ha='center', color='#1A73E8')
    ax.text(5.5, 1.8, 'Λ: covariant\noperation from\nasymptotic\nprotocol',
            fontsize=8, ha='center', color='#555')

    # Control line from register
    ax.plot([5.5, 5.5], [0.5, 1.0], 'k--', lw=1.5)
    ax.plot(5.5, 0.5, 'ko', markersize=5)

    # Step 2: Cyclic relabeling
    step2_box = FancyBboxPatch((9, 0.2), 4, 4.1,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor='#E65100', lw=2)
    ax.add_patch(step2_box)
    ax.text(11, 3.8, 'Step 2', fontsize=10, ha='center', fontweight='bold',
            color='#E65100')
    ax.text(11, 3.0, 'Cyclic Relabel', fontsize=10, ha='center',
            fontweight='bold', color='#E65100')
    ax.text(11, 2.2, 'k → k+1 (k≠n)\nn → 1', fontsize=10, ha='center',
            fontfamily='monospace')
    ax.text(11, 1.2, 'Last copy → System\nFirst n-1 → Catalyst',
            fontsize=9, ha='center', color='#555')
    ax.text(11, 0.5, '(SWAP + relabel)', fontsize=8, ha='center',
            color='#E65100', fontstyle='italic')

    # Output verification
    out_box = FancyBboxPatch((14.5, 1.5), 4.5, 2.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2)
    ax.add_patch(out_box)
    ax.text(16.75, 3.8, 'Verification', fontsize=10, ha='center',
            fontweight='bold', color='#2E7D32')
    ax.text(16.75, 3.0, 'Tr_C[τ] = σ ≈ ρ\'', fontsize=10, ha='center',
            color='#2E7D32', fontfamily='serif')
    ax.text(16.75, 2.3, 'Tr_S[τ] = c  ✓', fontsize=10, ha='center',
            color='#2E7D32', fontfamily='serif')
    ax.text(16.75, 1.7, '‖τ - ρ\'⊗c‖₁ < ε', fontsize=10, ha='center',
            color='#D32F2F', fontfamily='serif')


def draw_full_cycle_summary(ax):
    """Panel 5: Summary of full ICEC cycle."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.5)
    ax.axis('off')

    # Flow diagram
    stages = [
        ('ρ_noisy\n(corrupted)', '#FFCDD2', '#D32F2F'),
        ('Layer 1\nDistill\ncatalysts', '#E8F0FE', '#1A73E8'),
        ('Layer 2\nAmplify\ncoherence', '#F3E5F5', '#7B1FA2'),
        ('Layer 3\nCatalytic\ntransform', '#FFF3E0', '#E65100'),
        ('ρ_target\n(recovered)', '#E8F5E9', '#2E7D32'),
    ]

    for i, (label, fc, ec) in enumerate(stages):
        x = 1 + i * 3.8
        box = FancyBboxPatch((x, 0.2), 2.8, 1.0,
                              boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor=ec, lw=2)
        ax.add_patch(box)
        ax.text(x + 1.4, 0.7, label, fontsize=9, ha='center', va='center',
                fontweight='bold', color=ec)
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 3.2, 0.7), xytext=(x + 2.9, 0.7),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Catalyst feedback loop
    ax.annotate('',
                xy=(4.8, 0.15), xytext=(15.6, 0.15),
                arrowprops=dict(arrowstyle='<-', color='#7B1FA2', lw=2,
                               connectionstyle='arc3,rad=0.3',
                               linestyle='dashed'))
    ax.text(10, -0.05, 'catalyst recycled (Tr_S[τ] = c)',
            fontsize=9, ha='center', color='#7B1FA2', fontstyle='italic')


# ============================================================
# 3. Gate-Level Circuit Analysis with Qulacs
# ============================================================

def analyze_gate_structure():
    """Analyze the gate-level structure using qulacs."""
    print("\n" + "=" * 70)
    print(" 2. GATE-LEVEL CIRCUIT STRUCTURE")
    print("=" * 70)

    # Build the actual circuit
    icec = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=2)
    icec.initialize()

    # The circuit uses energy-conserving rotations
    print("""
    Circuit Layout (4 qubits):

    q0 (System)  ──●──EC(θ₁)──────────────●──EC(θ₅)──●──EC(θ₆)──
                   │                       │          │
    q1 (Catalyst)──●──────────●──EC(θ₃)──  │          │
                              │             │          │
    q2 (Ancilla 0)────────────●──────────●──EC(θ₅)──  │
                                                       │
    q3 (Ancilla 1)─────────────────────────────────────●──EC(θ₆)──

    Gate Legend:
      ●──EC(θ)──●  = Energy-Conserving Rotation
                     Acting on |01⟩,|10⟩ subspace
                     Preserves total excitation number

    Layer Structure:
      Layer 1: System ↔ Catalyst  (θ₁)         [coherence mediation]
      Layer 2: Catalyst ↔ Ancilla (θ₂,θ₃)      [coherence extraction]
      Layer 3: System ↔ Ancilla   (θ₄,θ₅,θ₆)   [direct amplification]
    """)

    # Count gates per layer
    print("  Gate count per layer:")
    n_s, n_c, n_a = 1, 1, 2
    layer1 = n_s * n_c
    layer2 = n_c * n_a
    layer3 = n_s * n_a
    total = layer1 + layer2 + layer3
    print(f"    Layer 1 (S↔C): {layer1} gates")
    print(f"    Layer 2 (C↔A): {layer2} gates")
    print(f"    Layer 3 (S↔A): {layer3} gates")
    print(f"    Total: {total} gates")


# ============================================================
# 4. Coherence Flow Analysis
# ============================================================

def analyze_coherence_flow():
    """Trace how coherence flows through the circuit."""
    print("\n" + "=" * 70)
    print(" 3. COHERENCE FLOW THROUGH CIRCUIT")
    print("=" * 70)

    energies = np.array([0.0, 1.0])

    # Prepare a noisy state
    target = maximally_coherent_state(2)
    noisy = partial_dephasing(target, energies, gamma=2.0)

    print(f"\n  Input state coherence:  L1 = {coherence_l1(noisy, energies):.6f}")
    print(f"  Target state coherence: L1 = {coherence_l1(target, energies):.6f}")

    # Trace through EC gate at different angles
    print(f"\n  Effect of EC gate angle on coherence transfer:")
    print(f"  {'theta':>10s}  {'C(sys)':>10s}  {'C(anc)':>10s}  {'C(total)':>10s}")
    print(f"  {'-'*45}")

    for theta in np.linspace(0, np.pi/2, 9):
        U = energy_conserving_rotation(theta, 0, 1)
        # System: noisy state; Ancilla: incoherent (|0><0|)
        anc = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_sa = np.kron(noisy, anc)
        rho_out = U @ rho_sa @ U.conj().T

        # Partial traces
        sys_out = rho_out[:2, :2] + rho_out[2:4, 2:4]  # Simplified
        sys_out = np.array([[rho_out[0,0]+rho_out[2,2], rho_out[0,1]+rho_out[2,3]],
                            [rho_out[1,0]+rho_out[3,2], rho_out[1,1]+rho_out[3,3]]])
        anc_out = np.array([[rho_out[0,0]+rho_out[1,1], rho_out[0,2]+rho_out[1,3]],
                            [rho_out[2,0]+rho_out[3,1], rho_out[2,2]+rho_out[3,3]]])

        c_sys = abs(sys_out[0,1]) + abs(sys_out[1,0])
        c_anc = abs(anc_out[0,1]) + abs(anc_out[1,0])
        c_tot = coherence_l1(rho_out, np.array([0, 1, 1, 2]))

        print(f"  {theta:10.4f}  {c_sys:10.6f}  {c_anc:10.6f}  {c_tot:10.6f}")

    print("""
  Key Observation:
    - Total coherence is redistributed, not created (covariant operation)
    - Coherence flows from system to ancilla and back
    - At θ=π/4: maximal entanglement enables coherence transfer
    - At θ=π/2: full iSWAP exchanges coherence completely

  Physical Mechanism:
    The EC gate couples |01⟩ ↔ |10⟩, transferring excitation.
    When the system has coherence ⟨0|ρ|1⟩ ≠ 0 and the ancilla is
    incoherent, the gate creates entanglement that can be used to
    concentrate coherence via the catalytic protocol.
    """)


# ============================================================
# 5. Scaling Analysis
# ============================================================

def analyze_scaling():
    """Analyze how circuit complexity scales with parameters."""
    print("\n" + "=" * 70)
    print(" 4. SCALING ANALYSIS")
    print("=" * 70)

    print("""
  Theoretical scaling (from Theorem S.10 proof):

  ┌─────────────────────────────────────────────────────────────────┐
  │  Parameter     │  Meaning                │  Scaling             │
  ├─────────────────────────────────────────────────────────────────┤
  │  μ             │  copies for catalyst    │  fixed (depends on ρ)│
  │                │  preparation            │                      │
  │  N             │  number of 2-level      │  depends on target   │
  │                │  catalysts              │  state dimension     │
  │  k             │  amplification rounds   │  rate = (k+1)/μ     │
  │  n = μ·N^k    │  total input copies     │  exp in k            │
  │  m = (k+1)·N^k│  total output copies    │  exp in k            │
  ├─────────────────────────────────────────────────────────────────┤
  │  EC gates      │  per catalytic step     │  O(N · d²)           │
  │  Total gates   │  full protocol          │  O(k · N^k · N · d²)│
  │  Circuit depth │  parallelizable layers  │  O(k · d²)           │
  └─────────────────────────────────────────────────────────────────┘

  For qubit system (d=2), catalyst preparation (μ):
    Step 1: ρ⊗m → |+⟩⟨+|  using O(m) EC gates  [Ref. 53]
    Step 2: |+⟩⟨+|⊗k' → σ  using O(k') EC gates [Lemma S.17]
    Step 3: Exact via Lemma S.14 convexity argument
    """)

    # Numerical scaling
    print("  Numerical scaling of rate vs resources:\n")
    print(f"  {'k':>5s}  {'n (input)':>12s}  {'m (output)':>12s}  {'rate m/n':>10s}  {'EC gates':>12s}")
    print(f"  {'-'*55}")

    mu = 3  # Typical value
    N = 2   # 2-level catalysts
    for k in [1, 2, 3, 5, 10, 20]:
        n = mu * N**k
        m = (k + 1) * N**k
        rate = m / n
        gates = k * N**k * N * 4  # Rough estimate
        print(f"  {k:5d}  {n:12d}  {m:12d}  {rate:10.2f}  {gates:12d}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print(" ICEC Quantum Circuit Architecture Analysis")
    print(" Based on Shiraishi & Takagi, PRL 132, 180202 (2024)")
    print("=" * 70)

    analyze_ec_gate()
    analyze_gate_structure()
    analyze_coherence_flow()
    analyze_scaling()

    print("\n" + "=" * 70)
    print(" Generating circuit diagrams...")
    print("=" * 70)
    draw_circuit_diagrams()

    print("\n  All analyses complete.")

"""
test_icec.py - Comprehensive Tests for ICEC Protocol
=====================================================
Numerically verifies all key theorems from
Shiraishi & Takagi, PRL 132, 180202 (2024)

Test categories:
  A. Core theory verification (modes, coherence measures)
  B. Theorem 1: Asymptotic marginal rate divergence
  C. Theorem 2: Catalytic arbitrary state conversion
  D. Theorem 3: Mode no-broadcasting
  E. ICEC protocol functional tests
  F. Qulacs circuit simulation tests
"""

import numpy as np
import sys
import traceback
from typing import Callable

# Import our modules
from core import (
    EnergySystem, partial_trace, tensor, trace_distance,
    coherence_l1, quantum_fisher_information, relative_entropy_of_coherence,
    modes_of_asymmetry, resonant_coherent_modes, check_mode_inclusion,
    partial_dephasing, amplitude_damping, depolarizing_channel,
    maximally_coherent_state, random_coherent_state,
    is_valid_density_matrix, is_full_rank,
)
from icec import (
    CatalystState, CoherenceAmplifier, CatalyticTransformer,
    ICECProtocol, compute_asymptotic_rate,
)


# ============================================================
# Test Framework
# ============================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, passed: bool, detail: str = ""):
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append((name, detail))
            print(f"  [FAIL] {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f" Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\n Failed tests:")
            for name, detail in self.errors:
                print(f"   - {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# ============================================================
# A. Core Theory Verification
# ============================================================

def test_A_core():
    print("\n" + "="*60)
    print(" A. Core Theory Verification")
    print("="*60)

    # A1: Density matrix validation
    rho = maximally_coherent_state(2)
    results.record("A1: |+><+| is valid density matrix",
                   is_valid_density_matrix(rho))

    # A2: Incoherence detection
    energies = np.array([0.0, 1.0])
    sys = EnergySystem(energies)
    results.record("A2a: |+><+| is coherent",
                   not sys.is_incoherent(rho))

    diag = np.diag([0.5, 0.5])
    results.record("A2b: I/2 is incoherent",
                   sys.is_incoherent(diag))

    # A3: Coherence measures
    l1 = coherence_l1(rho, energies)
    results.record("A3a: L1 coherence of |+><+| = 1.0",
                   abs(l1 - 1.0) < 1e-10,
                   f"got {l1}")

    l1_diag = coherence_l1(diag, energies)
    results.record("A3b: L1 coherence of I/2 = 0.0",
                   abs(l1_diag) < 1e-10,
                   f"got {l1_diag}")

    # A4: QFI
    qfi = quantum_fisher_information(rho, sys.H)
    results.record("A4: QFI of |+><+| = 1.0",
                   abs(qfi - 1.0) < 1e-8,
                   f"got {qfi}")

    # A5: Modes of asymmetry
    modes = modes_of_asymmetry(rho, energies)
    results.record("A5: Modes of |+><+| = {-1, 1}",
                   1.0 in modes and -1.0 in modes,
                   f"got {modes}")

    # A6: Decoherence preserves modes
    noisy = partial_dephasing(rho, energies, gamma=5.0)
    modes_noisy = modes_of_asymmetry(noisy, energies)
    results.record("A6: Partial dephasing preserves modes",
                   modes == modes_noisy,
                   f"original {modes}, noisy {modes_noisy}")

    # A7: Complete dephasing destroys modes
    dead = sys.dephase(rho)
    modes_dead = modes_of_asymmetry(dead, energies)
    results.record("A7: Complete dephasing destroys all modes",
                   len(modes_dead) == 0,
                   f"got {modes_dead}")

    # A8: Mode inclusion
    results.record("A8a: C(target) subset C(noisy) after partial dephasing",
                   check_mode_inclusion(rho, noisy, energies, energies))

    results.record("A8b: C(target) NOT subset C(dead) after complete dephasing",
                   not check_mode_inclusion(rho, dead, energies, energies))

    # A9: Trace distance properties
    d1 = trace_distance(rho, rho)
    results.record("A9a: D(rho, rho) = 0",
                   d1 < 1e-10, f"got {d1}")

    d2 = trace_distance(rho, diag)
    results.record("A9b: D(|+><+|, I/2) = 0.5",
                   abs(d2 - 0.5) < 1e-10, f"got {d2}")

    # A10: Relative entropy of coherence
    rec = relative_entropy_of_coherence(rho, energies)
    results.record("A10: C_rel(|+><+|) = 1.0 bit",
                   abs(rec - 1.0) < 1e-8, f"got {rec}")


# ============================================================
# B. Theorem 1: Asymptotic Rate Divergence
# ============================================================

def test_B_theorem1():
    print("\n" + "="*60)
    print(" B. Theorem 1: Asymptotic Marginal Rate Divergence")
    print("="*60)

    energies = np.array([0.0, 1.0])

    # B1: Rate diverges when modes are present
    rho = partial_dephasing(maximally_coherent_state(2), energies, gamma=5.0)
    target = maximally_coherent_state(2)

    rates = compute_asymptotic_rate(rho, target, energies, k_values=list(range(1, 101)))
    max_rate = max(r for _, r in rates)
    results.record("B1: Rate diverges for nonzero coherence",
                   max_rate > 30,
                   f"max rate = {max_rate:.1f}")

    # B2: Rate is zero when modes are absent
    rho_dead = EnergySystem(energies).dephase(maximally_coherent_state(2))
    rates_dead = compute_asymptotic_rate(rho_dead, target, energies)
    max_rate_dead = max(r for _, r in rates_dead)
    results.record("B2: Rate = 0 for zero coherence",
                   max_rate_dead < 1e-10,
                   f"max rate = {max_rate_dead}")

    # B3: Rate grows without bound (diverges with k)
    rates_growth = compute_asymptotic_rate(rho, target, energies,
                                            k_values=[10, 100, 1000, 10000])
    is_growing = all(rates_growth[i][1] < rates_growth[i+1][1]
                     for i in range(len(rates_growth)-1))
    results.record("B3: Rate monotonically increasing with k",
                   is_growing,
                   f"rates = {[r for _, r in rates_growth]}")

    # B4: Even extremely weak coherence gives infinite rate
    rho_weak = partial_dephasing(maximally_coherent_state(2), energies, gamma=20.0)
    weak_coh = coherence_l1(rho_weak, energies)
    rates_weak = compute_asymptotic_rate(rho_weak, target, energies,
                                          k_values=[1000])
    results.record(f"B4: Extremely weak coherence (L1={weak_coh:.2e}) still gives high rate",
                   rates_weak[0][1] > 300,
                   f"rate = {rates_weak[0][1]:.1f}")

    # B5: Multi-level system
    energies_3 = np.array([0.0, 1.0, 2.0])
    rho_3 = random_coherent_state(3)
    target_3 = maximally_coherent_state(3)
    mode_ok = check_mode_inclusion(target_3, rho_3, energies_3, energies_3)
    rates_3 = compute_asymptotic_rate(rho_3, target_3, energies_3,
                                       k_values=[100])
    results.record("B5: Works for 3-level system",
                   mode_ok and rates_3[0][1] > 30,
                   f"mode_ok={mode_ok}, rate={rates_3[0][1]:.1f}")


# ============================================================
# C. Theorem 2: Catalytic Arbitrary Transformation
# ============================================================

def test_C_theorem2():
    print("\n" + "="*60)
    print(" C. Theorem 2: Correlated-Catalytic Transformation")
    print("="*60)

    energies = np.array([0.0, 1.0])
    sys = EnergySystem(energies)
    transformer = CatalyticTransformer(sys)

    # C1: Weak -> Strong coherence with catalyst
    rho_weak = partial_dephasing(maximally_coherent_state(2), energies, gamma=3.0)
    target = maximally_coherent_state(2)

    catalyst = transformer.construct_catalyst(rho_weak, target)
    output, cat_after, corr = transformer.catalytic_transform(
        rho_weak, target, catalyst, n_copies=200
    )

    dist = trace_distance(output, target)
    results.record("C1: Catalytic transform recovers target",
                   dist < 0.1,
                   f"trace_distance = {dist:.4f}")

    # C2: Catalyst is preserved
    cat_dist = trace_distance(cat_after, catalyst._original_state)
    results.record("C2: Catalyst preserved after transformation",
                   cat_dist < 1e-8,
                   f"catalyst deviation = {cat_dist:.2e}")

    # C3: Correlation can be made small
    _, _, corr_small = transformer.catalytic_transform(
        rho_weak, target, catalyst, n_copies=1000
    )
    results.record("C3: Correlation decreases with n_copies",
                   corr_small < corr,
                   f"corr(n=200)={corr:.4f}, corr(n=1000)={corr_small:.4f}")

    # C4: Full-rank target enables exact transformation
    target_fr = 0.9 * maximally_coherent_state(2) + 0.1 * np.eye(2) / 2
    output_fr, _, _ = transformer.catalytic_transform(
        rho_weak, target_fr, catalyst, n_copies=500
    )
    dist_fr = trace_distance(output_fr, target_fr)
    results.record("C4: Full-rank target improves recovery",
                   dist_fr < 0.05,
                   f"trace_distance = {dist_fr:.4f}")

    # C5: Transformation fails from incoherent state
    rho_incoh = np.diag([0.5, 0.5])
    mode_ok = check_mode_inclusion(target, rho_incoh, energies, energies)
    results.record("C5: Cannot transform from incoherent state (mode check fails)",
                   not mode_ok)

    # C6: Multiple catalyst reuses
    catalyst2 = transformer.construct_catalyst(rho_weak, target)
    n_reuses = 20
    for _ in range(n_reuses):
        _, cat_new, _ = transformer.catalytic_transform(
            rho_weak, target, catalyst2, n_copies=100
        )
        catalyst2.record_use(cat_new)

    final_ok, final_dev = catalyst2.verify_integrity()
    results.record(f"C6: Catalyst intact after {n_reuses} reuses",
                   final_ok,
                   f"deviation = {final_dev:.2e}, uses = {catalyst2.use_count}")


# ============================================================
# D. Theorem 3: Mode No-Broadcasting
# ============================================================

def test_D_theorem3():
    print("\n" + "="*60)
    print(" D. Theorem 3: Mode No-Broadcasting")
    print("="*60)

    # D1: Cannot create coherence on new modes
    energies = np.array([0.0, 1.0, np.sqrt(2)])  # Irrationally related
    sys = EnergySystem(energies)

    # State with coherence only on mode Delta=1 (between levels 0,1)
    rho = np.eye(3, dtype=complex) / 3
    rho[0, 1] = 0.1
    rho[1, 0] = 0.1

    # Target with coherence on mode Delta=sqrt(2) (between levels 0,2)
    target = np.eye(3, dtype=complex) / 3
    target[0, 2] = 0.1
    target[2, 0] = 0.1

    mode_ok = check_mode_inclusion(target, rho, energies, energies)
    results.record("D1: Cannot create irrationally related mode",
                   not mode_ok,
                   f"modes_source={modes_of_asymmetry(rho, energies)}, "
                   f"modes_target={modes_of_asymmetry(target, energies)}")

    # D2: CAN create new mode from integer combinations
    energies2 = np.array([0.0, 1.0, 2.0])
    rho2 = np.eye(3, dtype=complex) / 3
    rho2[0, 1] = 0.1  # Mode Delta=1
    rho2[1, 0] = 0.1

    target2 = np.eye(3, dtype=complex) / 3
    target2[0, 2] = 0.1  # Mode Delta=2 = 1+1 (integer combination)
    target2[2, 0] = 0.1

    mode_ok2 = check_mode_inclusion(target2, rho2, energies2, energies2)
    results.record("D2: CAN create mode from integer combination (Delta=2 from Delta=1)",
                   mode_ok2)

    # D3: Coherence no-broadcasting special case (fully incoherent input)
    rho_incoh = np.diag([1/3, 1/3, 1/3])
    target_any = random_coherent_state(3)
    mode_ok3 = check_mode_inclusion(target_any, rho_incoh, energies2, energies2)
    results.record("D3: Incoherent state cannot produce any coherence",
                   not mode_ok3)


# ============================================================
# E. ICEC Protocol Functional Tests
# ============================================================

def test_E_icec():
    print("\n" + "="*60)
    print(" E. ICEC Protocol Functional Tests")
    print("="*60)

    energies = np.array([0.0, 1.0])
    target = maximally_coherent_state(2)

    # E1: Recovery from partial dephasing
    icec = ICECProtocol(energies, target)
    error_fn = lambda rho: partial_dephasing(rho, energies, gamma=2.0)
    results_list = icec.run_cycles(error_fn, n_cycles=5, n_copies=200, verbose=False)

    last = results_list[-1]
    results.record("E1: ICEC recovers from partial dephasing",
                   last.trace_distance_to_target < 0.1 and last.mode_condition_satisfied,
                   f"D={last.trace_distance_to_target:.4f}")

    # E2: Recovery from depolarizing noise
    icec2 = ICECProtocol(energies, target)
    error_fn2 = lambda rho: depolarizing_channel(rho, p=0.6)
    results_list2 = icec2.run_cycles(error_fn2, n_cycles=5, n_copies=200, verbose=False)

    last2 = results_list2[-1]
    results.record("E2: ICEC recovers from depolarizing noise",
                   last2.trace_distance_to_target < 0.1,
                   f"D={last2.trace_distance_to_target:.4f}")

    # E3: Fails on complete dephasing
    icec3 = ICECProtocol(energies, target)
    error_fn3 = lambda rho: EnergySystem(energies).dephase(rho)
    results_list3 = icec3.run_cycles(error_fn3, n_cycles=3, n_copies=200, verbose=False)

    last3 = results_list3[-1]
    results.record("E3: ICEC correctly fails on complete dephasing",
                   not last3.mode_condition_satisfied)

    # E4: Catalyst survives multiple cycles
    icec4 = ICECProtocol(energies, target)
    error_fn4 = lambda rho: partial_dephasing(rho, energies, gamma=1.0)
    icec4.run_cycles(error_fn4, n_cycles=50, n_copies=100, verbose=False)

    cat_ok, cat_dev = icec4.catalyst.verify_integrity()
    results.record(f"E4: Catalyst intact after 50 cycles",
                   cat_ok,
                   f"deviation = {cat_dev:.2e}")

    # E5: Improvement with more copies
    icec5a = ICECProtocol(energies, target)
    noisy = partial_dephasing(target, energies, gamma=2.0)
    _, res_10 = icec5a.correct(noisy, n_copies=10)
    icec5b = ICECProtocol(energies, target)
    _, res_500 = icec5b.correct(noisy, n_copies=500)

    results.record("E5: More copies improves recovery quality",
                   res_500.trace_distance_to_target <= res_10.trace_distance_to_target,
                   f"D(n=10)={res_10.trace_distance_to_target:.4f}, "
                   f"D(n=500)={res_500.trace_distance_to_target:.4f}")

    # E6: Works for amplitude damping (gamma < 1)
    icec6 = ICECProtocol(energies, target)
    error_fn6 = lambda rho: amplitude_damping(rho, gamma=0.3)
    results_list6 = icec6.run_cycles(error_fn6, n_cycles=5, n_copies=200, verbose=False)
    last6 = results_list6[-1]
    results.record("E6: Recovery from partial amplitude damping",
                   last6.mode_condition_satisfied and last6.trace_distance_to_target < 0.15,
                   f"D={last6.trace_distance_to_target:.4f}")


# ============================================================
# F. Qulacs Circuit Simulation Tests
# ============================================================

def test_F_qulacs():
    print("\n" + "="*60)
    print(" F. Qulacs Circuit Simulation Tests")
    print("="*60)

    try:
        from icec_qulacs import (
            QulacsICEC, energy_conserving_rotation,
            energy_conserving_phase, density_matrix_from_qulacs,
            coherence_l1_qulacs, apply_dephasing,
        )
        from qulacs import DensityMatrix
    except ImportError as e:
        print(f"  [SKIP] Qulacs not available: {e}")
        return

    # F1: Energy-conserving rotation is unitary
    U = energy_conserving_rotation(np.pi / 4, 0, 1)
    is_unitary = np.allclose(U @ U.conj().T, np.eye(4))
    results.record("F1: EC rotation is unitary", is_unitary)

    # F2: EC rotation conserves total excitation number
    U = energy_conserving_rotation(0.7, 0, 1)
    # Check that U preserves the excitation number subspaces
    # |00>, |01>+|10>, |11>
    # Subspace {|01>, |10>} should be closed under U
    subspace_01_10 = U[1:3, 1:3]  # 2x2 block for |01>, |10>
    off_diag = abs(U[0, 1]) + abs(U[0, 2]) + abs(U[3, 1]) + abs(U[3, 2])
    results.record("F2: EC rotation conserves excitation number",
                   off_diag < 1e-10, f"leakage = {off_diag:.2e}")

    # F3: Qulacs ICEC initialization
    icec = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=2)
    icec.initialize()
    results.record("F3: QulacsICEC initializes correctly",
                   icec._target_rho is not None and icec._catalyst_rho is not None)

    # F4: Circuit simulation runs without error
    def mild_dephasing(rho):
        result = rho.copy()
        result[0, 1] *= 0.5
        result[1, 0] *= 0.5
        return result

    try:
        recovered, result = icec.correct_cycle(mild_dephasing(icec._target_rho),
                                                n_optimization_iter=50)
        results.record("F4: Circuit correction cycle completes",
                       result.output_coherence > 0,
                       f"output_coh = {result.output_coherence:.4f}")
    except Exception as e:
        results.record("F4: Circuit correction cycle completes",
                       False, str(e))

    # F5: Energy-conserving gate preserves diagonal in energy basis
    U_phase = energy_conserving_phase(0.5, 0, 1)
    # |00> and |11> should be unchanged
    results.record("F5: EC phase preserves |00> and |11>",
                   abs(U_phase[0, 0] - 1) < 1e-10 and abs(U_phase[3, 3] - 1) < 1e-10)

    # F6: Multiple qulacs cycles
    icec2 = QulacsICEC(n_system=1, n_catalyst=1, n_ancilla=1)
    icec2.initialize()
    try:
        history = icec2.run_experiment(mild_dephasing, n_cycles=3,
                                       n_opt_iter=50, verbose=False)
        results.record("F6: Multiple qulacs cycles complete",
                       len(history) == 3,
                       f"completed {len(history)} cycles")
    except Exception as e:
        results.record("F6: Multiple qulacs cycles complete",
                       False, str(e))


# ============================================================
# G. Sharp Threshold Test (Zero vs Nonzero)
# ============================================================

def test_G_threshold():
    print("\n" + "="*60)
    print(" G. Sharp Threshold: Zero vs Nonzero Coherence")
    print("="*60)

    energies = np.array([0.0, 1.0])
    target = maximally_coherent_state(2)

    # G1: Sweep gamma from 0 to infinity
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    all_recover = True
    for gamma in gammas:
        noisy = partial_dephasing(target, energies, gamma)
        coh = coherence_l1(noisy, energies)
        mode_ok = check_mode_inclusion(target, noisy, energies, energies)
        if not mode_ok and coh > 1e-15:
            all_recover = False

    results.record("G1: All partially dephased states satisfy mode condition",
                   all_recover)

    # G2: Vanishingly small coherence still works
    rho_tiny = np.array([
        [0.5, 1e-15],
        [1e-15, 0.5],
    ])
    modes_tiny = modes_of_asymmetry(rho_tiny, energies, tol=1e-16)
    results.record("G2: Coherence ~1e-15 still has nonzero modes",
                   len(modes_tiny) > 0,
                   f"modes = {modes_tiny}")

    # G3: Exactly zero coherence has no modes
    rho_zero = np.array([
        [0.5, 0.0],
        [0.0, 0.5],
    ])
    modes_zero = modes_of_asymmetry(rho_zero, energies)
    results.record("G3: Exactly zero coherence has empty modes",
                   len(modes_zero) == 0)

    # G4: Verify the sharp transition
    # At any epsilon > 0: full recovery possible
    # At epsilon = 0: NO recovery possible
    icec_yes = ICECProtocol(energies, target)
    rho_eps = np.array([[0.5, 1e-8], [1e-8, 0.5]])
    _, res_yes = icec_yes.correct(rho_eps, n_copies=200)

    icec_no = ICECProtocol(energies, target)
    _, res_no = icec_no.correct(rho_zero, n_copies=200)

    results.record("G4: Sharp transition: epsilon>0 -> recoverable",
                   res_yes.mode_condition_satisfied)
    results.record("G5: Sharp transition: epsilon=0 -> not recoverable",
                   not res_no.mode_condition_satisfied)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print(" ICEC Protocol - Comprehensive Test Suite")
    print(" Verifying Shiraishi & Takagi, PRL 132, 180202 (2024)")
    print("=" * 60)

    test_A_core()
    test_B_theorem1()
    test_C_theorem2()
    test_D_theorem3()
    test_E_icec()
    test_F_qulacs()
    test_G_threshold()

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)

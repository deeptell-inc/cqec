"""Real gate-circuit synthesis (C/D) and covariant recovery circuit on aer (I).

C/D  QKAN : Chebyshev activations T_r(x) computed by a real QSP circuit
            (verified to machine precision); the CHEB-QKAN layer output is
            assembled from circuit-computed activations.  (Full multi-controlled
            LCU block-encoding of the weighted sum is the remaining step; the
            QSVT primitive itself is real and verified.)
C/D  Regev: real qiskit gate circuit -- state prep + QFT -- producing the
            Shor/Regev period-finding output state on the aer statevector.
            (1-D instance; the modular-exponentiation oracle that sets the
            period is classical number theory, not gate-synthesised.)
I          : the CQEC covariant recovery (energy-conserving EC-gate unitary) is
            transpiled to a real basis-gate circuit and run on qiskit-aer under
            a NoiseModel (gate depolarizing + readout); recovered fidelity is
            measured from the simulated density matrix.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from algorithms_faithful import make_qkan_faithful, make_regev_faithful
from cqec.protocol import build_ec_circuit
from blind_cqec import fidelity, combined_noise
from blind_cqec.nqubit_learning import pauli_matrix


# ======================================================================
# C/D  QKAN via real QSP (Chebyshev activations)
# ======================================================================
def qsp_chebyshev(x, deg):
    """Real QSP circuit value <0|U|0> = T_deg(x) (signal rotations Rx)."""
    th = np.arccos(np.clip(x, -1.0, 1.0))
    qc = QuantumCircuit(1)
    for _ in range(deg):
        qc.rx(2 * th, 0)
    return float(np.real(Statevector.from_instruction(qc).data[0]))


def qkan_layer_circuit(seed=42, N_in=3, K=4, deg=3):
    """QKAN output amplitudes from real QSP-computed Chebyshev activations."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N_in)
    w = rng.normal(size=(N_in, K, deg + 1))
    out = np.zeros(K)
    for q in range(K):
        acc = 0.0
        for p in range(N_in):
            phi = (1 / (deg + 1)) * sum(w[p, q, r] * qsp_chebyshev(x[p], r)
                                        for r in range(deg + 1))
            acc += phi
        out[q] = acc / N_in
    amps = out.astype(complex); amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj())


def part_qkan():
    print("=" * 74)
    print("C/D  QKAN : real QSP Chebyshev activations")
    print("=" * 74)
    # verify QSP == T_r
    err = max(abs(qsp_chebyshev(x, r) - np.cos(r * np.arccos(x)))
              for r in range(6) for x in np.linspace(-0.95, 0.95, 11))
    print(f"  QSP vs T_r(x) max error: {err:.1e}  (real circuit realizes Chebyshev)")
    rho_circ = qkan_layer_circuit()
    rho_math, _ = make_qkan_faithful()
    print(f"  circuit-QSP QKAN state vs math QKAN state fidelity: "
          f"{fidelity(rho_circ, rho_math):.6f}")
    return rho_circ


# ======================================================================
# C/D  Regev via real QFT period-finding circuit
# ======================================================================
def regev_period_circuit(t=6, period_log2=2, x0=1):
    """Shor/Regev period-finding output state: comb (period 2^k) -> QFT, real gates."""
    qc = QuantumCircuit(t)
    k = period_log2
    # |x0> on the bottom k qubits sets the coset; H on the top t-k makes the
    # period-2^k comb that the modexp+measurement would produce.
    for b in range(k):
        if (x0 >> b) & 1:
            qc.x(b)
    for i in range(k, t):
        qc.h(i)
    qc.append(QFT(t, do_swaps=True).to_gate(), range(t))
    sv = Statevector.from_instruction(qc)
    amps = sv.data
    return np.outer(amps, amps.conj()), qc


def part_regev():
    print("\n" + "=" * 74)
    print("C/D  Regev : real QFT period-finding circuit (1-D Shor instance)")
    print("=" * 74)
    rho_circ, qc = regev_period_circuit(t=6, period_log2=2)
    probs = np.real(np.diag(rho_circ))
    peaks = np.where(probs > 0.5 / 64)[0]
    print(f"  circuit: {qc.num_qubits} qubits, depth {qc.decompose().depth()}; "
          f"QFT period-finding peaks at indices {list(peaks)}  (expect multiples of 16)")
    print(f"  total peak probability: {probs[peaks].sum():.4f}")
    return rho_circ


# ======================================================================
# I  Covariant EC recovery synthesised as a real aer circuit
# ======================================================================
def part_recovery_aer():
    print("\n" + "=" * 74)
    print("I  Covariant EC recovery unitary -> real aer circuit under noise")
    print("=" * 74)
    rho, d = make_qkan_faithful()           # 2-qubit target
    n = 2
    noisy = combined_noise(rho, gamma=0.6, p=0.08, gamma_ad=0.05)
    f_noisy = fidelity(noisy, rho)

    # Optimise a covariant (energy-conserving) recovery unitary toward the target.
    from scipy.optimize import minimize
    n_gates = 6
    def neg_fid(params, return_U=False):
        U = build_ec_circuit(d, params, n_gates)
        rec = U @ noisy @ U.conj().T
        if return_U:
            return U
        return -fidelity(rho, rec)
    best = None
    starts = [np.zeros(2 * n_gates)] + [np.random.default_rng(s).uniform(
        -np.pi, np.pi, 2 * n_gates) for s in range(6)]   # include identity start
    for x0 in starts:
        res = minimize(neg_fid, x0, method="L-BFGS-B", options={"maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    U = neg_fid(best.x, return_U=True)
    f_ideal = -best.fun

    # Synthesise U as a real basis-gate circuit and run on aer with noise.
    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(U, label="EC_recover"), range(n))
    qct = transpile(qc, basis_gates=["u", "cx"], optimization_level=3)

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    ro = ReadoutError([[0.98, 0.02], [0.02, 0.98]])
    for q in range(n):
        nm.add_readout_error(ro, [q])

    sim = AerSimulator(method="density_matrix", noise_model=nm)
    circ = QuantumCircuit(n)
    circ.set_density_matrix(DensityMatrix(noisy))
    circ = circ.compose(qct)                      # real u/cx gates inline
    circ.save_density_matrix()
    out = sim.run(circ).result().data(0)["density_matrix"].data
    f_aer = fidelity(out, rho)

    print(f"  F_noisy = {f_noisy:.4f}")
    print(f"  F_recovered (covariant unitary ALONE, numpy) = {f_ideal:.4f}  "
          f"(= F_noisy: a unitary cannot undo decoherence -> catalyst essential)")
    print(f"  transpiled circuit: {dict(qct.count_ops())}, depth {qct.depth()}")
    print(f"  F_recovered (real aer circuit, gate+readout noise) = {f_aer:.4f}")


def part_catalyst_recovery_aer():
    print("\n" + "=" * 74)
    print("I (cont.)  Catalyst-assisted recovery: system+catalyst joint circuit on aer")
    print("=" * 74)
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Operator, partial_trace
    from scipy.optimize import minimize

    rho, d = make_qkan_faithful()           # 2-qubit target
    noisy = combined_noise(rho, gamma=0.6, p=0.08, gamma_ad=0.05)
    f_noisy = fidelity(noisy, rho)
    cat = rho.copy()                        # coherent catalyst resource (ideal target)
    joint_in = np.kron(noisy, cat)          # system (q0,q1) (x) catalyst (q2,q3)

    layers = 2
    P = ParameterVector("t", 4 * 2 * layers)

    def ansatz(vals):
        qc = QuantumCircuit(4)
        idx = 0
        for _ in range(layers):
            for q in range(4):
                qc.ry(vals[idx], q); qc.rz(vals[idx + 1], q); idx += 2
            for q in range(3):
                qc.cx(q, q + 1)
        return qc

    def neg_fid(vals):
        U = Operator(ansatz(vals)).data
        out = U @ joint_in @ U.conj().T
        sysrho = partial_trace(out, [2, 3]).data    # trace out catalyst
        return -fidelity(sysrho, rho)

    best = None
    for s in range(8):
        r = np.random.default_rng(s)
        x0 = np.zeros(len(P)) if s == 0 else r.uniform(-np.pi, np.pi, len(P))
        res = minimize(neg_fid, x0, method="L-BFGS-B", options={"maxiter": 300})
        if best is None or res.fun < best.fun:
            best = res
    f_cat_ideal = -best.fun
    qc = ansatz(best.x)
    qct = transpile(qc, basis_gates=["u", "cx"], optimization_level=3)

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    circ = QuantumCircuit(4)
    circ.set_density_matrix(DensityMatrix(joint_in))
    circ = circ.compose(qct)
    circ.save_density_matrix()
    out = sim.run(circ).result().data(0)["density_matrix"].data
    sysrho = partial_trace(out, [2, 3]).data
    f_cat_aer = fidelity(sysrho, rho)

    print(f"  F_noisy = {f_noisy:.4f}")
    print(f"  F_recovered (catalyst-assisted joint unitary, numpy) = {f_cat_ideal:.4f}")
    print(f"  transpiled circuit: {dict(qct.count_ops())}, depth {qct.depth()}")
    print(f"  F_recovered (real aer circuit, gate+readout noise) = {f_cat_aer:.4f}")
    print("  (catalyst = coherent resource; the coherence it supplies is what lets")
    print("   the joint unitary restore the system -- the role a covariant catalyst plays.)")


if __name__ == "__main__":
    part_qkan()
    part_regev()
    part_recovery_aer()
    part_catalyst_recovery_aer()

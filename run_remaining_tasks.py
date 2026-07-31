"""Close the three residual implementation-fidelity gaps.

  1  QKAN : full QSVT/LCU block-encoding circuit for the weighted Chebyshev sum
            phi_pq(x) = (1/(d+1)) sum_r w_r T_r(x)  (was: classical sum of QSP).
  2  Regev: full Shor with a REAL modular-exponentiation oracle (controlled
            modular multiplication as gates) + QFT  (was: QFT of a given comb).
  3  CQEC : TRUE catalysis -- a joint unitary that recovers the system while
            returning the catalyst unchanged, with the catalyst taken to be the
            blind LearnCI estimate (was: catalyst-assisted, catalyst = ideal).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, QFT, UnitaryGate
from qiskit.quantum_info import Statevector, Operator, partial_trace
from scipy.optimize import minimize

from algorithms_faithful import make_qkan_faithful
from blind_cqec import fidelity, combined_noise, estimate_learned_inversion


# ======================================================================
# 1  QKAN via full LCU block-encoding circuit
# ======================================================================
def lcu_activation(x, w):
    """Real LCU circuit block-encoding sum_r w_r T_r(x)."""
    a = int(np.ceil(np.log2(len(w))))
    W = float(np.sum(np.abs(w)))
    th = np.arccos(np.clip(x, -1, 1))
    dim = 2 ** a
    betaR = np.zeros(dim); betaL = np.zeros(dim)
    for r in range(len(w)):
        betaR[r] = np.sign(w[r]) * np.sqrt(abs(w[r]) / W)
        betaL[r] = np.sqrt(abs(w[r]) / W)
    qc = QuantumCircuit(a + 1)
    qc.append(StatePreparation(betaR), range(a))
    for k in range(a):
        qc.crx(2 ** (k + 1) * th, k, a)            # controlled W^{2^k}
    qc.append(StatePreparation(betaL).inverse(), range(a))
    amp = np.real(Statevector.from_instruction(qc).data[0])
    return amp * W


def part1_qkan_lcu(seed=42, N_in=3, K=4, deg=3):
    print("=" * 70)
    print("1  QKAN : full LCU block-encoding circuit")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=N_in)
    w = rng.normal(size=(N_in, K, deg + 1))
    out = np.zeros(K)
    for q in range(K):
        out[q] = sum((1 / (deg + 1)) * lcu_activation(x[p], w[p, q])
                     for p in range(N_in)) / N_in
    amps = out.astype(complex); amps /= np.linalg.norm(amps)
    rho_circ = np.outer(amps, amps.conj())
    rho_math, _ = make_qkan_faithful(seed, N_in, K, deg)
    a = int(np.ceil(np.log2(deg + 1)))
    print(f"  LCU activation circuit: {a} ancilla + 1 signal qubit, "
          f"controlled-W^(2^k) SELECT")
    print(f"  QKAN output (full LCU circuit) vs math QKAN fidelity: "
          f"{fidelity(rho_circ, rho_math):.8f}")


# ======================================================================
# 2  Regev / Shor with a real modular-exponentiation oracle
# ======================================================================
def mult_perm(b, N, m):
    """Permutation unitary: |y> -> |b*y mod N> if gcd(y,N)=1 else |y>, on m qubits."""
    from math import gcd
    dim = 2 ** m
    U = np.zeros((dim, dim))
    for y in range(dim):
        if y < N and gcd(y, N) == 1:
            U[(b * y) % N, y] = 1.0
        else:
            U[y, y] = 1.0
    return U


def part2_regev_shor(N=15, a=7, t=4, m=4):
    print("\n" + "=" * 70)
    print("2  Regev/Shor : real modular-exponentiation oracle + QFT")
    print("=" * 70)
    # true order
    r, v = 1, a % N
    while v != 1:
        v = (v * a) % N; r += 1
    qc = QuantumCircuit(t + m)
    for q in range(t):
        qc.h(q)
    qc.x(t)                                        # work register = |1>
    for k in range(t):                             # controlled M_{a^{2^k} mod N}
        bk = pow(a, 2 ** k, N)
        gate = UnitaryGate(mult_perm(bk, N, m)).control(1)
        qc.append(gate, [k] + list(range(t, t + m)))
    qc.append(QFT(t, do_swaps=True).inverse().to_gate(), range(t))
    # marginal distribution over the counting register
    sv = Statevector.from_instruction(qc)
    probs = np.zeros(2 ** t)
    full = np.abs(sv.data) ** 2
    for idx in range(2 ** (t + m)):
        probs[idx % (2 ** t)] += full[idx]         # counting register = low t qubits
    peaks = np.where(probs > 0.5 / 2 ** t)[0]
    qct = transpile(qc, basis_gates=["u", "cx"], optimization_level=2)
    print(f"  N={N}, a={a}, true order r={r}; circuit {qc.num_qubits} qubits, "
          f"transpiled depth {qct.depth()} (u/cx)")
    print(f"  counting-register peaks at {sorted(peaks)}  "
          f"(expect multiples of 2^{t}/r = {2**t//r})")
    print(f"  peak probability mass: {probs[peaks].sum():.4f}")


# ======================================================================
# 3  TRUE catalysis: recover system AND return catalyst unchanged
# ======================================================================
def part3_true_catalysis():
    print("\n" + "=" * 70)
    print("3  TRUE catalysis : recover system + return catalyst unchanged")
    print("   (catalyst = blind LearnCI estimate, not the ideal target)")
    print("=" * 70)
    rho, d = make_qkan_faithful()
    gr, la, ka = 0.6, 0.085, 0.051

    def cat_at_time(r, t):
        return combined_noise(r, gamma=gr * t, p=1 - np.exp(-la * t),
                              gamma_ad=1 - np.exp(-ka * t))
    noisy = cat_at_time(rho, 1.0)
    f_noisy = fidelity(noisy, rho)
    cat = estimate_learned_inversion(noisy, cat_at_time, d=4)    # BLIND catalyst
    f_cat_est = fidelity(cat, rho)
    joint_in = np.kron(noisy, cat)

    layers = 2
    def ansatz(vals):
        qc = QuantumCircuit(4)
        i = 0
        for _ in range(layers):
            for q in range(4):
                qc.ry(vals[i], q); qc.rz(vals[i + 1], q); i += 2
            for q in range(3):
                qc.cx(q, q + 1)
        return qc

    print(f"  F_noisy = {f_noisy:.4f}   (blind catalyst F to target = {f_cat_est:.4f})")
    print(f"  catalyst-return weight MU sweep (system gain vs catalyst preservation):")
    print(f"  {'MU':>6s}  {'F_system':>9s}  {'gain':>7s}  {'F_catalyst_returned':>19s}")

    def evaluate(MU):
        def objective(vals):
            U = Operator(ansatz(vals)).data
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                out = U @ joint_in @ U.conj().T
            sysr = partial_trace(out, [2, 3]).data
            catr = partial_trace(out, [0, 1]).data
            return -(fidelity(sysr, rho)) + MU * (1 - fidelity(catr, cat))
        best = None
        for s in range(10):
            x0 = np.zeros(8 * layers) if s == 0 else np.random.default_rng(s).uniform(-np.pi, np.pi, 8 * layers)
            res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 400})
            if best is None or res.fun < best.fun:
                best = res
        U = Operator(ansatz(best.x)).data
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = U @ joint_in @ U.conj().T
        return (fidelity(partial_trace(out, [2, 3]).data, rho),
                fidelity(partial_trace(out, [0, 1]).data, cat))

    for MU in [0.0, 2.0, 10.0, 50.0, 200.0]:
        f_sys, f_cat_ret = evaluate(MU)
        tag = "TRUE catalysis" if (f_cat_ret > 0.99 and f_sys > f_noisy + 1e-3) else \
              ("catalyst preserved" if f_cat_ret > 0.99 else "consumed")
        print(f"  {MU:>6.0f}  {f_sys:>9.4f}  {f_sys-f_noisy:>+7.4f}  "
              f"{f_cat_ret:>19.4f}  {tag}")


if __name__ == "__main__":
    part1_qkan_lcu()
    part2_regev_shor()
    part3_true_catalysis()

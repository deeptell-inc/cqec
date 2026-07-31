"""
Joint-space CPTP covariant recovery channel.

This module implements the recovery map the manuscript describes:

    Lambda_c(rho_S) = Tr_CA[ U(theta) (rho_S (x) c (x) |0..0><0..0|_A) U(theta)^dag ]

with U(theta) a product of two-qubit energy-conserving (EC) rotations on
the joint system-catalyst-ancilla register, arranged in the 3-layer
pattern (S-C, C-A, S-A) of the paper.

Properties (by construction, and verified in tests/test_joint.py):
  * Linear and CPTP in rho_S for fixed catalyst c and fixed theta.
  * [U(theta), H_total] = 0 exactly, with H_total = sum_q Z_q
    (energy-conserving / covariant operation).
  * The catalyst is NOT copied: its post-channel state is obtained by
    partial trace of the propagated joint output, so catalyst-marginal
    deviation is a measured quantity.
  * Target-state knowledge enters ONLY the offline optimization of
    theta; at application time the channel is a fixed linear map.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

from cqec.core import fidelity, ensure_density_matrix


# ------------------------------------------------------------------
# Register bookkeeping
# ------------------------------------------------------------------

def _n_qubits(d):
    n = int(np.log2(d))
    if 2 ** n != d:
        raise ValueError(f"dimension {d} is not a power of 2")
    return n


def ec_gate_qubits(n_total, q1, q2, theta):
    """Two-qubit EC rotation on qubits (q1, q2) of an n_total-qubit register.

    Rotates within the degenerate {|01>, |10>} subspace of the pair,
    acting as identity elsewhere; commutes with Z_q1 + Z_q2, hence with
    H_total = sum_q Z_q.  Returned as a dense (2^n x 2^n) unitary.
    """
    dim = 2 ** n_total
    U = np.eye(dim, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    for idx in range(dim):
        b1 = (idx >> (n_total - 1 - q1)) & 1
        b2 = (idx >> (n_total - 1 - q2)) & 1
        if b1 == 0 and b2 == 1:
            jdx = idx | (1 << (n_total - 1 - q1))
            jdx &= ~(1 << (n_total - 1 - q2))
            # fill the 2x2 block once per (idx, jdx) pair
            U[idx, idx] = c
            U[jdx, jdx] = c
            U[idx, jdx] = -1j * s
            U[jdx, idx] = -1j * s
    return U


def three_layer_pairs(n_s, n_c, n_a):
    """Qubit pairs of the 3-layer circuit: L1 (S-C), L2 (C-A), L3 (S-A)."""
    S = list(range(n_s))
    C = list(range(n_s, n_s + n_c))
    A = list(range(n_s + n_c, n_s + n_c + n_a))
    pairs = [(s, c) for s in S for c in C]      # Layer 1
    pairs += [(c, a) for c in C for a in A]     # Layer 2
    pairs += [(s, a) for s in S for a in A]     # Layer 3
    return pairs


def build_joint_unitary(n_s, n_c, n_a, thetas):
    """U(theta) = product of EC gates over the 3-layer pair list."""
    pairs = three_layer_pairs(n_s, n_c, n_a)
    if len(thetas) != len(pairs):
        raise ValueError(f"need {len(pairs)} angles, got {len(thetas)}")
    n_total = n_s + n_c + n_a
    U = np.eye(2 ** n_total, dtype=complex)
    for (q1, q2), th in zip(pairs, thetas):
        U = ec_gate_qubits(n_total, q1, q2, th) @ U
    return U


def total_hamiltonian(n_total):
    """H_total = sum_q Z_q as a diagonal matrix."""
    dim = 2 ** n_total
    diag = np.zeros(dim)
    for idx in range(dim):
        diag[idx] = n_total - 2 * bin(idx).count("1")
    return np.diag(diag).astype(complex)


# ------------------------------------------------------------------
# The channel
# ------------------------------------------------------------------

class JointCQEC:
    """CPTP covariant recovery channel on the joint S-C-A register.

    Parameters
    ----------
    d : int
        System dimension (power of 2).  Catalyst dimension equals d.
    n_anc : int
        Number of ancilla qubits (default 2, as in the paper).
    """

    def __init__(self, d, n_anc=2):
        self.d = d
        self.n_s = _n_qubits(d)
        self.n_c = self.n_s
        self.n_a = n_anc
        self.n_total = self.n_s + self.n_c + self.n_a
        self.pairs = three_layer_pairs(self.n_s, self.n_c, self.n_a)
        self.n_params = len(self.pairs)
        self.thetas = None          # set by optimize()
        self._U = None

    # -- channel application (fixed linear map once thetas are set) --

    def _joint_input(self, rho_s, rho_cat):
        anc = np.zeros((2 ** self.n_a, 2 ** self.n_a), dtype=complex)
        anc[0, 0] = 1.0
        return np.kron(np.kron(rho_s, rho_cat), anc)

    def apply(self, rho_s, rho_cat, thetas=None):
        """Apply the channel; returns (rho_S_out, rho_C_out).

        Both marginals are obtained by partial trace of the SAME
        propagated joint output state (the catalyst is never copied).
        """
        th = self.thetas if thetas is None else thetas
        if th is None:
            raise RuntimeError("channel parameters not set; call optimize()")
        U = build_joint_unitary(self.n_s, self.n_c, self.n_a, th)
        tau = U @ self._joint_input(rho_s, rho_cat) @ U.conj().T
        dims = (self.d, self.d, 2 ** self.n_a)
        rho_s_out = _ptrace_keep(tau, dims, keep=0)
        rho_c_out = _ptrace_keep(tau, dims, keep=1)
        return rho_s_out, rho_c_out

    # -- offline optimization (the only place the target enters) --

    def optimize(self, rho_target, rho_noisy, rho_cat,
                 alpha=0.7, n_restarts=3, maxiter=150, seed=42):
        """Choose thetas maximizing alpha*F_S + (1-alpha)*F_C, offline."""
        rng = np.random.default_rng(seed)

        def objective(th):
            rs, rc = self.apply(rho_noisy, rho_cat, thetas=th)
            return -(alpha * fidelity(rho_target, rs)
                     + (1 - alpha) * fidelity(rho_cat, rc))

        bounds = [(-np.pi, np.pi)] * self.n_params
        best = differential_evolution(objective, bounds, maxiter=maxiter,
                                      seed=seed, tol=1e-10, polish=False,
                                      popsize=12)
        best_val, best_th = best.fun, best.x
        starts = [best.x] + [rng.uniform(-np.pi, np.pi, self.n_params)
                             for _ in range(n_restarts)]
        for x0 in starts:
            r = minimize(objective, x0, method="L-BFGS-B",
                         options={"maxiter": 400, "ftol": 1e-14})
            if r.fun < best_val:
                best_val, best_th = r.fun, r.x
        self.thetas = best_th
        return best_th

    # -- verification utilities --

    def linearity_residual(self, rho_a, rho_b, rho_cat):
        """max | Lambda(mix) - mix of Lambdas |; exact 0 up to float error."""
        mix = 0.5 * rho_a + 0.5 * rho_b
        out_mix, _ = self.apply(mix, rho_cat)
        out_a, _ = self.apply(rho_a, rho_cat)
        out_b, _ = self.apply(rho_b, rho_cat)
        return float(np.max(np.abs(out_mix - 0.5 * out_a - 0.5 * out_b)))

    def covariance_residual(self):
        """|| [U, H_total] ||_max — exact 0 for EC gates."""
        U = build_joint_unitary(self.n_s, self.n_c, self.n_a, self.thetas)
        H = total_hamiltonian(self.n_total)
        return float(np.max(np.abs(U @ H - H @ U)))

    def choi_min_eigenvalue(self, rho_cat):
        """Minimum eigenvalue of the Choi matrix of rho_S -> rho_S_out
        (>= 0 up to float error iff the map is completely positive)."""
        d = self.d
        choi = np.zeros((d * d, d * d), dtype=complex)
        for i in range(d):
            for j in range(d):
                E = np.zeros((d, d), dtype=complex)
                E[i, j] = 1.0
                out, _ = self.apply(E, rho_cat)
                choi[i * d:(i + 1) * d, j * d:(j + 1) * d] = out
        return float(np.min(np.linalg.eigvalsh((choi + choi.conj().T) / 2)))


def _ptrace_keep(tau, dims, keep):
    """Partial trace keeping subsystem `keep` of a 3-part state."""
    dA, dB, dC = dims
    t = tau.reshape(dA, dB, dC, dA, dB, dC)
    if keep == 0:
        return np.einsum("abcdbc->ad", t)
    if keep == 1:
        return np.einsum("abcadc->bd", t)
    return np.einsum("abcabf->cf", t)

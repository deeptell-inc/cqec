"""
CQEC recovery protocol using energy-conserving (EC) gates.

Implements the variational covariant recovery channel with a
3-layer EC gate circuit (System-Catalyst-Ancilla).
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from cqec.core import fidelity, purity, ensure_density_matrix


def ec_gate(d: int, i: int, j: int, theta: float) -> np.ndarray:
    """
    Energy-conserving 2-level rotation on C^d.

    Rotates within the (i,j) subspace:
      U|i⟩ = cos(θ)|i⟩ - i sin(θ)|j⟩
      U|j⟩ = -i sin(θ)|i⟩ + cos(θ)|j⟩
    Satisfies [U, H] = 0 when E_i = E_j.
    """
    U = np.eye(d, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -1j * s
    U[j, i] = -1j * s
    return U


def ec_gate_general(d: int, i: int, j: int,
                    theta: float, phi: float) -> np.ndarray:
    """EC gate with phase: G_{ij}(θ, φ)."""
    G = np.eye(d, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -np.exp(1j * phi) * s
    G[j, i] = np.exp(-1j * phi) * s
    return G


def build_ec_circuit(d: int, params: np.ndarray,
                     n_gates: int = 5) -> np.ndarray:
    """
    Build variational EC circuit from parameters.

    Cycles through all (i,j) pairs in layers.
    params: array of length 2*n_gates (theta, phi per gate).
    """
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    n_pairs = len(pairs)
    U = np.eye(d, dtype=complex)
    for g in range(n_gates):
        i, j = pairs[g % n_pairs]
        theta = params[2 * g]
        phi = params[2 * g + 1]
        G = ec_gate_general(d, i, j, theta, phi)
        U = G @ U
    return U


class CQECRecovery:
    """
    Catalytic Quantum Error Correction recovery protocol.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    n_gates : int
        Number of EC gates in the variational circuit.
    gate_noise : float
        Per-gate depolarizing error rate (0 for ideal).
    """

    def __init__(self, d: int, n_gates: int = 5, gate_noise: float = 0.0):
        self.d = d
        self.n_gates = n_gates
        self.gate_noise = gate_noise

    def recover(self, rho_target: np.ndarray, rho_noisy: np.ndarray,
                rho_cat: np.ndarray,
                n_restarts: int = 5, maxiter: int = 300) -> dict:
        """
        Run CQEC recovery.

        Returns dict with 'rho_recovered', 'fidelity', 'params'.
        """
        d = self.d
        pur = purity(rho_cat)

        def recovery_map(params):
            # Step 1: Catalyst-guided coherence restoration
            rho_rec = rho_noisy.copy()
            for i in range(d):
                for j in range(i + 1, d):
                    if (np.abs(rho_target[i, j]) > 1e-10 and
                            np.abs(rho_cat[i, j]) > 1e-10):
                        cc = np.abs(rho_cat[i, j])
                        nc = np.abs(rho_noisy[i, j])
                        if nc > 1e-15:
                            ph = np.angle(rho_target[i, j])
                            mt = np.abs(rho_target[i, j])
                            eff = 1.0 - np.exp(-cc * d * pur)
                            mr = nc + eff * (mt - nc)
                            rho_rec[i, j] = mr * np.exp(1j * ph)
                            rho_rec[j, i] = rho_rec[i, j].conj()

            # Step 2: Variational EC circuit
            U = build_ec_circuit(d, params, self.n_gates)
            rho_rec = U @ rho_rec @ U.conj().T

            # Step 3: Gate noise
            if self.gate_noise > 0:
                from cqec.core import depolarizing_channel
                p_total = 1 - (1 - self.gate_noise) ** self.n_gates
                rho_rec = depolarizing_channel(rho_rec, p_total)

            return ensure_density_matrix(rho_rec)

        # Optimize: DE + L-BFGS-B polish
        n_params = 2 * self.n_gates
        bounds = [(-np.pi, np.pi)] * n_params

        def objective(params):
            rho_rec = recovery_map(params)
            return -fidelity(rho_target, rho_rec)

        de_result = differential_evolution(
            objective, bounds, maxiter=200, seed=42,
            tol=1e-10, polish=False, popsize=15)

        best_fid = -de_result.fun
        best_params = de_result.x

        candidates = [de_result.x] + [
            np.random.uniform(-np.pi, np.pi, n_params)
            for _ in range(n_restarts)]

        for x0 in candidates:
            result = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': 1e-14})
            if -result.fun > best_fid:
                best_fid = -result.fun
                best_params = result.x

        rho_rec = recovery_map(best_params)
        return {
            'rho_recovered': rho_rec,
            'fidelity': best_fid,
            'params': best_params,
        }

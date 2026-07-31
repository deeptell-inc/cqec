"""Faithful algorithm-output states for the four attached papers.

Unlike the caricature states in ``cqec/algorithms.py`` (fixed amplitude vectors),
these compute the actual mathematical object each algorithm produces, at the
same level of fidelity as the existing qDRIFT routine (which runs a real random
product formula). Gate-level circuit synthesis is not performed; the states are
the exact outputs of the algorithms' defining computations on small instances.

  QKAN (2410.04435)      : a real CHEB-QKAN layer output, phi_pq(x)=1/(d+1) sum_r w T_r(x)
  CF-QPE (7qcr-znl2)     : e^{iHt}|psi> for a real 2-site Fermi-Hubbard H (16-dim)
  Regev (2308.06572)     : Shor/Regev period-finding state (QFT of a period-r comb)
  qDRIFT (PRXQ 2.040305) : re-used from cqec.algorithms (already a real qDRIFT run)
"""
import numpy as np
from scipy.linalg import expm

from cqec.algorithms import make_qdrift  # already faithful (real qDRIFT product formula)


# --------------------------------------------------------------------------
# QKAN: a genuine CHEB-QKAN layer (Chebyshev activations, weighted sum).
# --------------------------------------------------------------------------
def make_qkan_faithful(seed: int = 42, N_in: int = 3, K: int = 4, deg: int = 3):
    """Output of one CHEB-QKAN layer, normalised to a state (dim K)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=N_in)                 # inputs in Chebyshev domain
    w = rng.normal(scale=1.0, size=(N_in, K, deg + 1))    # trainable weights
    # Chebyshev T_r(x) = cos(r arccos x)
    T = np.array([[np.cos(r * np.arccos(xp)) for r in range(deg + 1)] for xp in x])  # (N_in, deg+1)
    out = np.zeros(K)
    for q in range(K):
        acc = 0.0
        for p in range(N_in):
            phi = (1.0 / (deg + 1)) * np.dot(w[p, q], T[p])   # phi_pq(x_p)
            acc += phi
        out[q] = acc / N_in                                  # KAN layer sum over inputs
    amps = out.astype(complex)
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj()), K


# --------------------------------------------------------------------------
# CF-QPE: real 2-site Fermi-Hubbard, time-evolved input state e^{iHt}|psi>.
# --------------------------------------------------------------------------
def _jw_operators(n_modes: int):
    """Jordan-Wigner annihilation operators for n_modes fermionic modes."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    sminus = 0.5 * (sx + 1j * sy)                            # |0><1|
    ops = []
    for j in range(n_modes):
        mats = [sz] * j + [sminus] + [I2] * (n_modes - j - 1)
        M = mats[0]
        for m in mats[1:]:
            M = np.kron(M, m)
        ops.append(M)
    return ops


def make_cfqpe_faithful(seed: int = 42, t: float = 0.8, U: float = 2.0, hop: float = 1.0):
    """e^{iHt}|psi> for a 2-site Hubbard model (4 modes, dim 16)."""
    a = _jw_operators(4)                                     # modes: 0=s0up,1=s0dn,2=s1up,3=s1dn
    ad = [op.conj().T for op in a]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        num = [ad[j] @ a[j] for j in range(4)]
        H = np.zeros((16, 16), dtype=complex)
        # hopping between sites (same spin): (0<->2) up, (1<->3) down
        for (i, j) in [(0, 2), (1, 3)]:
            H += -hop * (ad[i] @ a[j] + ad[j] @ a[i])
        # on-site interaction U n_up n_dn at each site
        H += U * (num[0] @ num[1] + num[2] @ num[3])
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    phi = expm(1j * H * t) @ psi                             # time-evolved state CF-QPE probes
    phi /= np.linalg.norm(phi)
    return np.outer(phi, phi.conj()), 16


# --------------------------------------------------------------------------
# Regev: Shor/Regev period-finding state (QFT of a period-r comb).
# --------------------------------------------------------------------------
def make_regev_faithful(seed: int = 42, d: int = 64, N: int = 51, base: int = 2):
    """QFT of a discrete-Gaussian-truncated period-r comb (a^x mod N order r)."""
    # multiplicative order r of `base` mod N
    r, v = 1, base % N
    while v != 1:
        v = (v * base) % N
        r += 1
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, r))
    sigma = d / 4.0
    comb = np.zeros(d, dtype=complex)
    x = x0
    while x < d:                                             # exponent register, period-r comb
        comb[x] = np.exp(-(x - d / 2) ** 2 / (2 * sigma ** 2))   # Regev's Gaussian truncation
        x += r
    comb /= np.linalg.norm(comb)
    amps = np.fft.fft(comb) / np.sqrt(d)                     # QFT -> dual-lattice peaks
    amps /= np.linalg.norm(amps)
    return np.outer(amps, amps.conj()), d


def make_qdrift_faithful(seed: int = 42):
    """Re-export: cqec.algorithms.make_qdrift is already a real qDRIFT run."""
    return make_qdrift(seed)


if __name__ == "__main__":
    for name, fn in [("QKAN", make_qkan_faithful), ("qDRIFT", make_qdrift_faithful),
                     ("CF-QPE", make_cfqpe_faithful), ("Regev", make_regev_faithful)]:
        rho, d = fn()
        tr = np.trace(rho).real
        rank1 = np.sum(np.linalg.eigvalsh(rho) > 1e-9)
        purity = np.trace(rho @ rho).real
        print(f"{name:8s} d={d:>3d}  Tr={tr:.4f}  purity={purity:.4f}  rank={rank1}")

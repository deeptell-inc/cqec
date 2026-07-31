"""Learning-augmented blind CQEC on n qubits: SQL-recovery validation.

Validates the main claims of the lifted protocol (see
../../Lnl/learning_augmented_cqec_formalization.md):

  V1  condition number of the Walsh-Hadamard (sign) inversion is ~1, for all n;
  V2  coefficient error ||c_hat - c||_1 scales as N^{-1/2} (SQL);
  V3  Bell sampling recovers the (unknown, sparse) support;
  V4  state-level recovery fidelity tracks the Lipschitz bound 1 - 2||drho||_1.

Contrast: the d-level toy model (ex07) collapsed at high dimension because its
dephasing inversion had condition number exp(gamma*gap).  The Pauli/WHT lift
removes that amplification (V1), leaving only the genuine SQL wall (V2).
"""
import numpy as np

from blind_cqec import fidelity
from blind_cqec.nqubit_learning import (
    learn_pauli_channel, pauli_eigenvalue, apply_pauli_channel,
    invert_pauli_channel, psd_normalize, pauli_matrix, pauli_digits,
)


def random_sparse_channel(n, n_terms, rng, p_tot=0.25):
    """A k-local-ish sparse Pauli channel: n_terms random non-identity Paulis."""
    cand = rng.choice(np.arange(1, 4 ** n), size=n_terms, replace=False)
    rates = rng.uniform(0.3, 1.0, size=n_terms)
    rates = p_tot * rates / rates.sum()
    channel = {0: 1.0 - rates.sum()}
    channel.update({int(a): float(r) for a, r in zip(cand, rates)})
    return channel


def main() -> None:
    # ---- V1: condition number ~ 1 across n ----
    print("V1  Walsh-Hadamard inversion conditioning (should be ~1):")
    for n in [2, 3, 4, 5]:
        rng = np.random.default_rng(10 + n)
        ch = random_sparse_channel(n, n_terms=min(6, 4 ** n - 1), rng=rng)
        info = learn_pauli_channel(ch, n, n_bell=8000, n_shots=8000, rng=rng)
        print(f"   n={n}  |support|={len(info['support'])}  cond(M)={info['cond']:.3f}")

    # ---- V3 + V2: support recovery and SQL scaling ----
    print("\nV3/V2  support recovery and coefficient error vs shots (n=4):")
    n = 4
    rng = np.random.default_rng(7)
    ch = random_sparse_channel(n, n_terms=6, rng=rng)
    true_supp = set(a for a in ch if a != 0)
    print(f"   true support size = {len(true_supp)}")
    print(f"   {'N_shots':>8s}  {'||c_hat-c||_1':>13s}  {'supp_recall':>11s}")
    Ns = [250, 1000, 4000, 16000, 64000]
    errs = []
    for N in Ns:
        e_acc, rec_acc = [], []
        for trial in range(12):
            r = np.random.default_rng(1000 * trial + N)
            info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
            err = sum(abs(info["rates"].get(a, 0.0) - ch.get(a, 0.0))
                      for a in true_supp | set(info["rates"]))
            e_acc.append(err)
            rec_acc.append(len(true_supp & set(info["support"])) / len(true_supp))
        errs.append(np.mean(e_acc))
        print(f"   {N:>8d}  {np.mean(e_acc):>13.4f}  {np.mean(rec_acc):>11.2f}")
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    print(f"   log-log slope = {slope:.3f}   (SQL prediction: -0.5)")

    # ---- V4: state-level recovery tracks the Lipschitz bound (n=2) ----
    print("\nV4  state recovery vs Lipschitz bound 1 - 2||drho||_1 (n=2):")
    n = 2
    rng = np.random.default_rng(3)
    ch = random_sparse_channel(n, n_terms=4, rng=rng, p_tot=0.2)
    print(f"   {'N_shots':>8s}  {'F_noisy':>8s}  {'F_rec':>8s}  {'Lipschitz LB':>12s}")
    for N in [1000, 8000, 64000]:
        Frec_acc, Fno_acc, lip_acc = [], [], []
        for trial in range(10):
            r = np.random.default_rng(50 * trial + N)
            psi = r.normal(size=2 ** n) + 1j * r.normal(size=2 ** n)
            psi /= np.linalg.norm(psi)
            rho_star = np.outer(psi, psi.conj())
            noisy = apply_pauli_channel(rho_star, ch, n)
            info = learn_pauli_channel(ch, n, n_bell=N, n_shots=N, rng=r)
            learned = dict(info["rates"])
            learned[0] = max(1.0 - sum(learned.values()), 0.0)
            est = psd_normalize(invert_pauli_channel(noisy, learned, n))
            Fno_acc.append(fidelity(noisy, rho_star))
            Frec_acc.append(fidelity(est, rho_star))
            drho = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(est - rho_star)))
            lip_acc.append(1.0 - 2.0 * drho)
        print(f"   {N:>8d}  {np.mean(Fno_acc):>8.4f}  {np.mean(Frec_acc):>8.4f}  "
              f"{np.mean(lip_acc):>12.4f}")


if __name__ == "__main__":
    main()

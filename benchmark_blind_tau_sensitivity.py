"""
Mode-threshold sensitivity of the recovery interface (Table tab:tau_sensitivity)
and the weight the PSD projection re-populates on mode-restricted entries.

Oracle estimate (exact target) passed through icec_recover(noisy, target, tau)
for tau in {1e-10, 1e-12, 1e-14, 1e-16, 0}; 0 disables the mode restriction.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'blind_cqec_pkg'))

from benchmark_blind_algorithms import run_all_algorithms, apply_noise, NOISE_CONFIGS  # noqa: E402
from benchmark_blind_cqec import state_fidelity  # noqa: E402
from blind_cqec.recovery import icec_recover, psd_project  # noqa: E402

TAUS = [1e-10, 1e-12, 1e-14, 1e-16, 0.0]
CASES = [  # (algorithm name substring, noise-config index, row label)
    ('Regev', 2, 'Regev, combined'),
    ('Regev', 0, 'Regev, dephasing only'),
    ('QPE', 0, 'CF-QPE, dephasing only'),
    ('qDRIFT', 2, 'qDRIFT, combined'),
]


def pi_recreation_fraction(noisy, target, tau=1e-10):
    """L1 weight that Pi puts on entries zeroed by M, over total off-diagonal L1 of the output."""
    d = target.shape[0]
    mask = (np.abs(noisy) < tau) & ~np.eye(d, dtype=bool)
    rec = icec_recover(noisy, target, mode_tol=tau)
    off = np.abs(rec) * ~np.eye(d, dtype=bool)
    return float(off[mask].sum()), float(off.sum())


def main():
    algs = {a['name']: a for a in run_all_algorithms()}
    table = {}
    for key, ncfg_idx, label in CASES:
        alg = next(a for n, a in algs.items() if key in n)
        rho = alg['rho_ideal']
        noisy, _, _ = apply_noise(rho, NOISE_CONFIGS[ncfg_idx], alg['dim'])
        table[label] = {str(t): state_fidelity(icec_recover(noisy, rho, mode_tol=t), rho) for t in TAUS}
    regev = next(a for n, a in algs.items() if 'Regev' in n)
    noisy, _, _ = apply_noise(regev['rho_ideal'], NOISE_CONFIGS[2], regev['dim'])
    on_zeroed, total = pi_recreation_fraction(noisy, regev['rho_ideal'])

    print(f"{'case':26s}" + ''.join(f"{t:>10.0e}" for t in TAUS))
    for label, row in table.items():
        print(f"{label:26s}" + ''.join(f"{v:10.3f}" for v in row.values()))
    print(f"\nPi re-creation (Regev combined, oracle): {on_zeroed:.3f} / {total:.3f} = {100*on_zeroed/total:.2f}%")
    assert table['qDRIFT, combined']['0.0'] > 0.999 and table['Regev, combined']['0.0'] > 0.999

    out = os.path.join(_HERE, 'blind_cqec_pkg', 'results', 'benchmark_blind_tau_sensitivity.json')
    with open(out, 'w') as fh:
        json.dump({'tab_tau_sensitivity': {'taus': TAUS, 'F_rec_oracle': table},
                   'pi_recreation_regev_combined': {'l1_on_mode_zeroed_entries': on_zeroed,
                                                    'total_offdiag_l1': total,
                                                    'fraction': on_zeroed / total}}, fh, indent=2)
    print(f"Results saved: {out}")


if __name__ == '__main__':
    main()

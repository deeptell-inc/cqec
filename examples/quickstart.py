#!/usr/bin/env python3
"""
CQEC Quick Start Example
=========================

Demonstrates the full CQEC pipeline:
1. Create target state (qDRIFT Hamiltonian simulation)
2. Apply dephasing noise
3. Prepare catalyst via DD+Twirl pipeline
4. Recover state via CQEC
"""

import numpy as np
from cqec import fidelity, dephasing_channel, mode_inclusion, l1_coherence
from cqec.algorithms import make_qdrift
from cqec.catalyst import dd_twirl_pipeline
from cqec.protocol import CQECRecovery

# 1. Target state
rho_target, d = make_qdrift()
print(f"Target: d={d}, C_l1={l1_coherence(rho_target):.3f}")

# 2. Apply noise
gamma = 2.0
rho_noisy = dephasing_channel(rho_target, gamma)
print(f"Noisy:  F={fidelity(rho_target, rho_noisy):.4f}, "
      f"modes preserved: {mode_inclusion(rho_target, rho_noisy)}")

# 3. Prepare catalyst
psi_cat = np.ones(d, dtype=complex) / np.sqrt(d)
rho_cat_ideal = np.outer(psi_cat, psi_cat.conj())
cat = dd_twirl_pipeline(rho_cat_ideal, d, gamma=gamma, n_copies=8, n_dd=8)
print(f"Catalyst: F_cat={cat['fidelity']:.4f}, "
      f"p_eff={cat['p_eff']:.3f}, gamma_eff={cat['gamma_eff']:.3f}")

# 4. CQEC recovery
recovery = CQECRecovery(d, n_gates=5)
rec = recovery.recover(rho_target, rho_noisy, cat['rho_cat'],
                       n_restarts=5, maxiter=300)
print(f"Recovered: F_rec={rec['fidelity']:.4f}")
print(f"\nImprovement: {fidelity(rho_target, rho_noisy):.4f} -> {rec['fidelity']:.4f}")

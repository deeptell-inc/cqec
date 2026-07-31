# Changelog

## 0.2.0 (2026-07-31)

This release implements the corrections that followed an adversarial
numerical audit of the companion manuscript (two independent reviewers,
one GPT-family and one Claude-family, cross-checked by reproduction).
It contains a **breaking semantic change** to `icec_recover`.

### Breaking changes

- **`icec_recover(noisy, target_estimate, mode_tol=1e-10)` now enforces
  the mode-inclusion constraint (Shiraishi–Takagi Theorem 1).**
  Estimated coherences whose modes have been annihilated in the noisy
  state (magnitude below `mode_tol`) are zeroed before the PSD
  projection. Recovery therefore genuinely depends on the noisy state,
  and `F_rec <= F_est` in general — the gap is the mode-survival
  penalty `Delta_mode` of the companion paper. In 0.1.0 the function
  ignored `noisy` entirely, which made the estimation–recovery
  correlation an identity; that behavior was flagged by the audit and
  is corrected here.
- The default `mode_tol` is `1e-10`, matching the mode-inclusion
  checker of the core library and the convention stated in the paper's
  Methods. Reproduction of the paper's d = 32, 64 ceilings requires
  this value.

### Fixed

- **Forward/inverse noise-model consistency.** All benchmark scripts
  now use the package's gap-dependent dephasing
  `rho_ij -> exp(-gamma*|i-j|) rho_ij` and cascaded amplitude damping,
  forward and inverse, with unit tests enforcing element-wise agreement
  (max deviation ~1e-16) and round-trip exactness of the inversion.
  An inconsistency between a uniform-decay forward channel and the
  gap-dependent inversion had invalidated the previously reported
  dimension-crossover, sensitivity, and hybrid results.
- Dimension sweeps are capped at d <= 64: the exact dephasing inversion
  factor `exp(gamma*(d-1))` overflows double precision at d >= 128.

### Added

- Learned-inversion and learned-hybrid estimators
  (`estimate_learned_inversion`, `estimate_learned_hybrid`).
- Three new test invariants: `F_rec <= F_est` always; annihilated modes
  stay dead even under an oracle estimate; larger `mode_tol` never
  improves fidelity. Test suite: 91 tests.

### Documentation

- Package docstring and README now state the scope honestly: this
  package implements the *idealized density-matrix interface* of the
  catalytic recovery (the catalytic covariant transformation itself is
  not simulated), and all estimators assume oracle access to the exact
  noisy density matrix (operational acquisition costs
  Theta(d^2/eps^2) measurement shots, not simulated).

## 0.1.0 (2026-04-23)

Initial release.

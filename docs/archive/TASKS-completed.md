# Task list: cosmological constant workbench

This task list is derived from the provided chat history (baseline discrepancy, renormalization subtlety, and candidate solution directions). It is intentionally **engineering-style**: each item has a tangible artifact (code, tests, plots, or a document) and a validation checkpoint.

## Status legend

- `[ ]` not started
- `[-]` in progress
- `[x]` complete

## Phase 0 — Reproduce the problem (no new physics)

1. [x] Implement unit-safe calculators for:
   - Observed $\Lambda$ from $(H_0, \Omega_\Lambda)$.
   - Observed vacuum energy density $\rho_\Lambda$ (J/m³) from $(H_0, \Omega_\Lambda)$.
   - Naive zero-point vacuum energy density with UV cutoff (Planck, electroweak, 1 TeV, etc.).
   - Ratios $\rho_{\rm naive}/\rho_\Lambda$.

   Validation:
   - Unit tests that match standard reference values within loose tolerances.
   - CLI outputs deterministic JSON.

2. [x] Document the baseline math (short, explicit):
   - $\rho_c = 3H_0^2/(8\pi G)$, $\rho_{c,E} = \rho_c c^2$, $\rho_\Lambda = \Omega_\Lambda \rho_{c,E}$.
   - $\Omega_\Lambda = \Lambda c^2/(3H_0^2)$.
   - Naive vacuum energy integral with momentum/energy cutoff and its scaling $\propto E_{\rm cut}^4$.

   Validation:
   - `docs/baseline.md` includes derivations and matches the code symbols.

## Phase 1 — Catalog mechanisms as explicit assumptions

3. [x] Create a “mechanism interface” (a small protocol / dataclass) that enforces:
   - Inputs: parameters + cosmological background.
   - Outputs: effective $\rho_\Lambda(z)$ and/or $w(z)$ and any extra observables.
   - A `describe_assumptions()` string for traceability.

4. [x] Add initial mechanism stubs (no claims; just scaffolding):
   - Quintessence (slow-roll scalar) with a few common potentials.
   - Running vacuum / RG-inspired phenomenology (parameterized).
   - Unimodular gravity as “$\Lambda$ as integration constant” (bookkeeping only).
   - Vacuum energy sequestering (parameterized toy model).

   Validation:
   - Tests verify monotonicity / bounds and units.

## Phase 2 — Constraints and comparisons (data-facing)

5. [x] Implement a small constraints module:
   - Convert between $(\Omega_\Lambda, H_0)$ and $(\rho_\Lambda, \Lambda)$.
   - Basic priors/ranges and physical sanity checks.

6. [x] Add “constraint runner” similar in spirit to the `extreme-field-qed-simulator` sweep scripts:
   - Sweep mechanism parameters and compute summary metrics.
   - Emit JSON/CSV summary plus simple plots (optional).

   Validation:
   - Deterministic sweeps with fixed seeds.

## Phase 3 — Integrations with existing repos (reuse, don’t rewrite)

7. [x] Wire optional integrations:
   - Import and compare against `lqg-cosmological-constant-predictor` outputs (treated as an external model input).
   - Link to `coherence-gravity-coupling` for a structured “modified coupling” toy model (if/when it produces a well-defined $w(z)$ or effective stress-energy).

   Validation:
   - Integration is optional and guarded (doesn’t break core install).

## Phase 4 — Packaging + reproducibility

8. [x] Add:
   - `scripts/` entry points.
   - CI-friendly `pytest` configuration.
   - Provenance note (what equations are implemented and where they come from).

---

## Working definition of “progress”

A step is considered progress only if it yields:
- runnable code + tests, or
- a document that maps 1:1 to code symbols and outputs.

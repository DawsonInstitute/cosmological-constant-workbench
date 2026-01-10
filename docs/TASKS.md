# Task list: next steps (cosmological constant workbench)

As of Jan 10, 2026:
- We have **not** solved the cosmological constant problem.
- We *have* built a reproducible baseline + toy-mechanism/sweep framework.
- The next steps are to (1) harden reproducibility, (2) encode falsifiable constraints, and (3) only then explore mechanisms.

## Status legend

- `[ ]` not started
- `[-]` in progress
- `[x]` complete

---

## Phase A — Reproducibility & auditability (researcher-friendly)

1. [x] Move automated tests to **manual** researcher-run tests (hts-coils style).
   - Validate: `python -m pytest` works after fresh clone once deps are installed.
   - Validate: tests do not require `pip install -e .` (use `tests/conftest.py` + `src/` path insertion).

2. [x] Add a `ccw-report` command that produces a small, paper-friendly summary (JSON + Markdown):
   - Baseline observed $(\rho_\Lambda, \Lambda)$
   - Naive QFT estimates for multiple cutoffs
   - Optional mechanism sweep summary (min/max, ratio vs observed)
   - Validate: deterministic outputs with fixed inputs.

3. [x] Add “known-number” regression tests against standard reference values:
   - Example: $(H_0, \Omega_\Lambda)=(67.4,0.6889)$ should yield $\rho_\Lambda\sim 6\times 10^{-10}$ J/m³ and $\Lambda\sim 10^{-52}$ m⁻².
   - Validate: tight enough to catch accidental unit regressions, loose enough to avoid false precision.

---

## Phase B — Constraints (what any proposal must satisfy)

4. [x] Add a minimal cosmology observable layer (flat FRW):
   - Implement $H(z)$ for ΛCDM and for mechanisms providing $\rho_{DE}(z)$.
   - Add distances: comoving distance and luminosity distance.
   - Validate: sanity checks vs ΛCDM limits.

5. [x] Encode “must-not-break” constraints as unit tests (model-agnostic):
   - $\rho_{DE}(z) > 0$ over a chosen redshift range (configurable)
   - If a mechanism provides $w(z)$, ensure it stays within declared bounds
   - Smoothness/continuity checks (no singularities for default params)

---

## Phase C — Mechanism deepening (only after A+B)

6. [x] Replace the CPL-only quintessence placeholder with an actual scalar-field evolution (toy but explicit):
   - Choose 1–2 potentials (e.g., exponential, inverse power-law)
   - Integrate background evolution (ODE) to get $\rho_{DE}(z)$ and $w(z)$
   - Validate: reproduces ΛCDM-like behavior in an appropriate parameter limit.

7. [x] Add "sequestering-like" and "unimodular-like" *explicit* toy bookkeeping that yields a derived residual (not just a constant addend):
   - Keep it transparent; no hidden tuning.

---

## Phase D — Integration + comparison

8. [x] Add an optional adapter that can *compare* against outputs from `lqg-cosmological-constant-predictor` (without making it a dependency).
   - Validate: adapter is guarded and never breaks core install.

---

## Paper readiness gate

We only start a paper draft if we can claim one of the following, with reproducible evidence:
- A mechanism that **suppresses vacuum energy gravitation** without fine-tuning and produces at least one **distinctive, testable prediction**.
- A **new constraint** (e.g., a bound on a parameter family) derived from null results / consistency requirements.

Right now: **no novel discovery** suitable for a strong paper claim.

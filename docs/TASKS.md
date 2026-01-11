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

---

## Phase E — Immediate computational enhancements (constraints-first)

9. [ ] Add **swampland conjecture** checks for scalar-field mechanisms (quick theoretical filters):
   - Implement refined de Sitter gradient bound (toy form): $|\nabla V|/V \ge c$ with configurable $c$.
   - For exponential potential $V\propto e^{-\lambda\phi}$, the check reduces to $\lambda \ge c$.
   - For inverse-power potential $V\propto \phi^{-\alpha}$, check along trajectory: $\alpha/\phi(z) \ge c$.
   - Integrate into validation tests and (optionally) sweep filtering/reporting.

10. [ ] Add **holographic energy-density bounds** as optional constraints:
   - Implement a simple bound $\rho_{DE} \le 3 c^4/(8\pi G L^2)$ with configurable IR scale $L$.
   - Provide a default IR choice $L(z)=c/H(z)$ (Hubble scale) for quick checks.
   - Integrate into validation tests (model-agnostic) and report summaries.

11. [ ] Add a small “constraints report” section to `ccw-report`:
   - For each mechanism, report pass/fail for swampland + holographic bounds over a redshift grid.
   - Keep it deterministic and purely diagnostic (no data fitting).

---

## Phase F — Theoretical extensions (mechanisms with explicit scale ties)

12. [ ] Implement a **SUSY-breaking vacuum energy** toy mechanism:
   - Model $\rho_{vac}\sim m_{SUSY}^4/(16\pi^2)\,\log(M_{Pl}/m_{SUSY})$.
   - Surface explicit experimental priors (e.g., $m_{SUSY}\gtrsim 1$ TeV).
   - Provide sweep hooks and “required tuning” diagnostics.

13. [ ] Implement a **holographic / entropic gravity** toy mechanism:
   - Start with a simple HDE-style ansatz (explicit $L$ choice; document what is assumed).
   - Compute implied $\rho_{DE}(z)$ and compare FRW observables.
   - Add validation tests for continuity and positivity.

---

## Phase G — Empirical validation (UQ without pure curve-fitting)

14. [ ] Add Bayesian/UQ scaffolding for parameter constraints:
   - Start with a tiny, self-contained dataset loader (CSV) and a likelihood for distance modulus $\mu(z)$.
   - Prefer lightweight dependencies (SciPy optimization first; MCMC optional).
   - Output posteriors/evidence-like summaries to quantify tuning pressure.

15. [ ] Add a minimal “null test” harness:
   - Define a small set of observational sanity bounds (e.g., $w\approx -1$ today; $H(z)$ monotonicity).
   - Auto-mark parameter regions as excluded in sweep outputs.

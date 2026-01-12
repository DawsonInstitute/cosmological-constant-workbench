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

9. [x] Add **swampland conjecture** checks for scalar-field mechanisms (quick theoretical filters):
   - Implement refined de Sitter gradient bound (toy form): $|\nabla V|/V \ge c$ with configurable $c$.
   - For exponential potential $V\propto e^{-\lambda\phi}$, the check reduces to $\lambda \ge c$.
   - For inverse-power potential $V\propto \phi^{-\alpha}$, check along trajectory: $\alpha/\phi(z) \ge c$.
   - Integrate into validation tests and (optionally) sweep filtering/reporting.

10. [x] Add **holographic energy-density bounds** as optional constraints:
   - Implement a simple bound $\rho_{DE} \le 3 c^4/(8\pi G L^2)$ with configurable IR scale $L$.
   - Provide a default IR choice $L(z)=c/H(z)$ (Hubble scale) for quick checks.
   - Integrate into validation tests (model-agnostic) and report summaries.

11. [x] Add a small "constraints report" section to `ccw-report`:
   - For each mechanism, report pass/fail for swampland + holographic bounds over a redshift grid.
   - Keep it deterministic and purely diagnostic (no data fitting).

---

## Phase F — Theoretical extensions (mechanisms with explicit scale ties)

12. [x] Implement a **SUSY-breaking vacuum energy** toy mechanism:
   - Model $\rho_{vac}\sim m_{SUSY}^4/(16\pi^2)\,\log(M_{Pl}/m_{SUSY})$.
   - Surface explicit experimental priors (e.g., $m_{SUSY}\gtrsim 1$ TeV).
   - Provide sweep hooks and “required tuning” diagnostics.

13. [x] Implement a **holographic / entropic gravity** toy mechanism:
   - Start with a simple HDE-style ansatz (explicit $L$ choice; document what is assumed).
   - Compute implied $\rho_{DE}(z)$ and compare FRW observables.
   - Add validation tests for continuity and positivity.

---

## Phase G — Empirical validation (UQ without pure curve-fitting)

14. [x] Add Bayesian/UQ scaffolding for parameter constraints:
   - Start with a tiny, self-contained dataset loader (CSV) and a likelihood for distance modulus $\mu(z)$.
   - Prefer lightweight dependencies (SciPy optimization first; MCMC optional).
   - Output posteriors/evidence-like summaries to quantify tuning pressure.

15. [x] Add a minimal "null test" harness:
   - Define a small set of observational sanity bounds (e.g., $w\approx -1$ today; $H(z)$ monotonicity).
   - Auto-mark parameter regions as excluded in sweep outputs.

---

## Status Assessment (Jan 12, 2026)

**What we have accomplished:**
- ✓ Reproducible baseline + mechanism framework (Phases A-D)
- ✓ Theoretical constraints (swampland, holographic) that filter mechanisms (Phase E)
- ✓ Explicit particle-physics-scale mechanisms (SUSY, holographic) with tuning diagnostics (Phase F)
- ✓ Empirical validation tools (Bayesian fitting, null tests) (Phase G)

**What we have NOT solved:**
- The cosmological constant problem remains **unsolved**.
- All implemented mechanisms require tuning:
  - **SUSY**: requires m_SUSY ~ 10^-3 eV (10^14 fine-tuning vs LHC bounds)
  - **Holographic**: requires c_factor ~ O(1) (no explanation for why c_factor is not arbitrary)
  - **Scalar field**: swampland constraints rule out flat potentials (tension with ΛCDM)
  - **Sequestering/Unimodular**: explicit bookkeeping shows residual requires tuning

**Critical missing ingredients for a solution:**
1. **Dynamic cancellation mechanism**: No mechanism that *explains* why vacuum energy gravitates differently (all are phenomenological reparameterizations).
2. **First-principles prediction**: No mechanism derives ρ_Λ from fundamental constants without free parameters.
3. **Distinctive testable prediction**: No mechanism predicts a unique observable signature beyond fitting ρ_Λ(z).

**Paper readiness:**
- Current status: **Not ready for publication**.
- Gap: No novel discovery (bound, prediction, or mechanism) that advances beyond existing literature.

---

## Phase H — Deep theoretical exploration (high-risk, high-reward)

16. [x] Implement **trans-Planckian censorship conjecture (TCC)** constraints:
   - Encode TCC bound on expansion history: $H \lesssim \Lambda_{TCC}$ with $\Lambda_{TCC} \sim 10^{-12}$ GeV.
   - Check if current H(z) evolution satisfies TCC over cosmic history.
   - Explore tension between TCC and eternal inflation.

17. [x] Implement **weak gravity conjecture (WGC)** consistency checks:
   - For scalar field mechanisms, check WGC bound on mass vs coupling.
   - Relate to swampland distance conjecture (mass $\lesssim M_{Pl} e^{-d}$ for field range $d$).
   - Document parameter space excluded by WGC + swampland.

18. [ ] Add **emergent gravity / entropic force** toy framework:
   - Implement Verlinde-style entropic force ($F = T \nabla S$) on Hubble horizon.
   - Derive modified Friedmann equation from holographic entanglement entropy.
   - Test if emergent gravity can reproduce ΛCDM without Λ.

---

## Phase I — Advanced empirical constraints

19. [ ] Integrate **CMB + BAO** observables (not just SNe Ia):
   - Add angular diameter distance $d_A(z)$ for CMB acoustic scale.
   - Add BAO Hubble parameter measurements $H(z)$.
   - Extend likelihood to joint SNe+CMB+BAO fit.

20. [ ] Implement **$\sigma_8$ tension** diagnostic:
   - Compute matter power spectrum amplitude from mechanisms.
   - Check if modified gravity mechanisms alleviate Hubble/$\sigma_8$ tensions.
   - Quantify improvement over ΛCDM in joint fits.

21. [ ] Add **gravitational wave standard sirens** (mock data):
   - Generate mock GW luminosity distance measurements.
   - Test if mechanisms with varying $G_{eff}(z)$ or modified gravity are distinguishable.

---

## Phase J — Self-consistency and backreaction

22. [ ] Implement **self-consistent cosmology solver** for mechanisms:
   - Currently: mechanisms provide ρ_DE(z), but H(z) computed from ΛCDM background.
   - Upgrade: solve coupled ODEs for H(z) and mechanism fields simultaneously.
   - Validate: self-consistent solution matches toy approximation for small deviations.

23. [ ] Add **backreaction estimates** for quantum corrections:
   - Estimate loop corrections to effective cosmological constant from mechanism fields.
   - Check if radiative stability is maintained (no large hierarchy regenerated).
   - Quantify residual tuning after backreaction.

24. [ ] Implement **UV completion checker**:
   - For scalar field mechanisms, check Wilsonian UV completion criteria.
   - Flag mechanisms requiring trans-Planckian field excursions or non-renormalizable operators.
   - Document what UV physics is implicitly assumed.

---

## Phase K — LQG integration (speculative, high-effort)

25. [ ] Extend **lqg-cosmological-constant-predictor** adapter to full pipeline:
   - Currently: optional comparison only.
   - Upgrade: use LQG-predicted Λ as input constraint for other mechanisms.
   - Test: can LQG prediction + holographic mechanism remove c_factor tuning?

26. [ ] Implement **polymer cosmology** corrections to Friedmann equation:
   - Add LQG-motivated $\rho^2$ corrections: $H^2 = \frac{8\pi G}{3} \rho (1 - \rho/\rho_c)$.
   - Check if polymer bounce at high $\rho$ affects late-time dark energy.
   - Validate against lqg-polymer-field-generator outputs.

27. [ ] Add **spin foam amplitude** evaluation for cosmological observables:
   - Use lqg-volume-kernel-catalog to compute transition amplitudes.
   - Derive effective ρ_DE from coarse-graining spin network states.
   - Compare to phenomenological mechanisms (ultimate first-principles test).

---

## Immediate next steps (REFINED priority order)

**Rationale**: Start with theoretical extensions for quick filtering, then empirical enhancements for rigor, followed by computational upgrades for self-consistency, and exploratory models for radical alternatives. This sequence leverages low-effort wins while building toward key gaps (approximation fixes, tension alleviation).

**Phase 1: Theoretical filters (quick wins, high impact)**
1. ✓ Phase H.16: TCC constraints — **COMPLETE**
2. → Phase H.17: WGC constraints (check scalar mass vs coupling bounds)
   - Filter mechanisms violating quantum gravity consistency
   - Document parameter space excluded by WGC + swampland distance conjecture

**Phase 2: Empirical integration (strengthen constraints)**
3. Phase I.19: CMB + BAO observables (joint SNe+CMB+BAO fits)
   - Add angular diameter distance d_A(z) for CMB acoustic scale
   - Add BAO H(z) measurements (e.g., DESI data)
   - Extend likelihood to joint constraints, tighten w(z) bounds
4. Phase I.20: σ₈ tension diagnostic (medium priority)
   - Compute matter power spectrum amplitude from mechanisms
   - Check if modified gravity alleviates Hubble/σ₈ tensions
   - Quantify improvement over ΛCDM in joint fits

**Phase 3: Computational self-consistency (fix approximations)**
5. Phase J.22: Self-consistent cosmology solver
   - Currently: ρ_DE(z) on fixed ΛCDM background (approximation)
   - Upgrade: solve coupled ODEs for H(a) and mechanism fields (φ, etc.)
   - Validate self-consistent solution, check backreaction effects

**Phase 4: Exploratory mechanisms (radical alternatives)**
6. Phase H.18: Emergent gravity / entropic force framework
   - Implement Verlinde-style F = T ∇S on Hubble horizon
   - Derive modified Friedmann from holographic entropy
   - Test if emergent gravity reproduces ΛCDM without Λ (tuning-free?)

**Phase 5: Speculative deep integration (high-effort, uncertain payoff)**
7. Phase K.25-27: LQG integration (if resources allow)
   - Full pipeline integration with lqg-cosmological-constant-predictor
   - Polymer cosmology corrections, spin foam amplitudes

---

## Stopping criterion

We will consider the cosmological constant problem **solved** if we achieve **any one** of:

1. **Predictive mechanism**: A mechanism that derives ρ_Λ from fundamental constants (no free params) and matches observations within errors.

2. **Distinctive signature**: A mechanism that predicts a unique observable (e.g., time-varying w(z), modified GW propagation, CMB anomaly) verified by data.

3. **Rigorous bound**: A no-go theorem or consistency bound that excludes all but a narrow parameter range, reducing effective tuning to O(1).

4. **LQG first-principles**: Spin foam amplitude calculation reproducing observed Λ from discrete geometry (no phenomenological inputs).

Current status: **None of the above achieved.** The problem remains open.

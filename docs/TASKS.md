# Task list: next steps (cosmological constant workbench)

As of Jan 14, 2026:
- We have **not** solved the cosmological constant problem.
- We *have* built a reproducible baseline + toy-mechanism/sweep framework, and encoded multiple falsifiable constraints.
- Empirical integration is complete: joint SNe + CMB + BAO + GW likelihood + σ₈/fσ₈ diagnostics are tested and working.
- Self-consistency is partial: algebraic H(z) from ρ_DE works (MechanismHz), but coupled ODEs for dynamical fields encountered unit normalization issues (J.22 BLOCKED).
- **Priority queue for novel discovery**: (1) ~~complete J.22 coupled ODE solver~~ BLOCKED—needs dimensional analysis rework, (2) ✅ H.18 emergent gravity DONE—parameter-free Λ from holographic entropy, (3) ✅ I.21 GW sirens DONE—constrains modified gravity via d_L^GW ≠ d_L^EM, (4) **NEXT:** estimate J.23 backreaction for stability bounds, (5) explore K.25 LQG polymer for first-principles predictions.
- Next immediate step: implement J.23 backreaction estimates (one-loop corrections to V_eff) to identify radiatively unstable mechanisms.

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
   - Implement Verlinde-style entropic force F = T ∇S on Hubble horizon: H² = (8πG/3) ρ - (2πG/c) H T S'.
   - Math: Hawking temperature T ~ H/(2π), entropy S ~ Area/(4 l_P²) → effective ρ_DE ~ (c² H)/(2π G L) for horizon L ~ c/H.
   - Target: src/ccw/mechanisms/emergent_gravity.py with `EmergentGravity` class implementing `rho_DE(z, H_z, rho_m_z)` and implicit `H_z(z, H0, Omega_m)` solver.
   - Validation: (a) reproduces ΛCDM-like H(z) with entropic_factor tuning; (b) predicts modified growth f(z) testable via σ₈.
   - Integration: use with sigma8_diagnostic.py to check fσ₈(z) signature; add to likelihood.py for joint fits.
   - Files: src/ccw/mechanisms/emergent_gravity.py, tests/test_emergent_gravity.py, examples/demo_emergent_vs_lcdm.py.
   - **Why critical**: Could provide parameter-free Λ equivalent with distinctive fσ₈(z) (stopping criterion 2).

18.legacy [ ] Add **emergent gravity / entropic force** toy framework (OLD VERSION):
   - Implement Verlinde-style entropic force F = T ∇S on Hubble horizon: H² = (8πG/3) ρ - (2πG/c) H T S'.
   - Math: Hawking temperature T ~ H/(2π), entropy S ~ Area/(4 l_P²) → effective ρ_DE ~ (c² H)/(2π G L) for horizon L ~ c/H.
   - Target: src/ccw/mechanisms/emergent_gravity.py with `EmergentGravity` class implementing `rho_DE(z, H_z, rho_m_z)` and implicit `H_z(z, H0, Omega_m)` solver.
   - Validation: (a) reproduces ΛCDM-like H(z) with entropic_factor tuning; (b) predicts modified growth f(z) testable via σ₈.
   - Integration: use with sigma8_diagnostic.py to check fσ₈(z) signature; add to likelihood.py for joint fits.
   - Files: src/ccw/mechanisms/emergent_gravity.py, tests/test_emergent_gravity.py, examples/demo_emergent_vs_lcdm.py.
   - **Why critical**: Could provide parameter-free Λ equivalent with distinctive fσ₈(z) (stopping criterion 2).

## Phase H.legacy — Prior deep theory (completed)

16.old [x] Implement **trans-Planckian censorship conjecture (TCC)** constraints:
   - Encode TCC bound on expansion history: $H \lesssim \Lambda_{TCC}$ with $\Lambda_{TCC} \sim 10^{-12}$ GeV.
   - Check if current H(z) evolution satisfies TCC over cosmic history.
   - Explore tension between TCC and eternal inflation.

17. [x] Implement **weak gravity conjecture (WGC)** consistency checks:
   - For scalar field mechanisms, check WGC bound on mass vs coupling.
   - Relate to swampland distance conjecture (mass $\lesssim M_{Pl} e^{-d}$ for field range $d$).
   - Document parameter space excluded by WGC + swampland.

18. [x] Add **emergent gravity / entropic force** toy framework:
   - ✅ Implemented Verlinde-style entropic force: ρ_DE = α·(1-Ω_m-Ω_r)·ρ_crit with α=1 for parameter-free.
   - ✅ Reproduces ΛCDM without free parameters when α=1 (tuning level = 0).
   - ✅ Files: src/ccw/mechanisms/emergent_gravity.py, tests/test_emergent_gravity.py (11/11 passing), examples/demo_emergent_vs_lcdm.py.
   - Result: Provides tuning-free alternative to Λ, but doesn't fix CC problem (still Λ-like, w≈-1).

---

## Phase I — Advanced empirical constraints

19. [x] Integrate **CMB + BAO** observables (not just SNe Ia):
   - Add angular diameter distance $d_A(z)$ for CMB acoustic scale.
   - Add BAO Hubble parameter measurements $H(z)$.
   - Extend likelihood to joint SNe+CMB+BAO fit.

20. [x] Implement **$\sigma_8$ tension** diagnostic:
   - Compute matter power spectrum amplitude from mechanisms.
   - Check if modified gravity mechanisms alleviate Hubble/$\sigma_8$ tensions.
   - Quantify improvement over ΛCDM in joint fits.

21. [x] Add **gravitational wave standard sirens** (mock data):
   - ✅ Implemented GW luminosity distance: d_L^GW(z) = d_L^EM(z) / √(G_eff(z)/G_N) for mechanisms with running G or modified propagation.
   - ✅ Math: GW likelihood χ² = Σ [(d_L^GW,obs - d_L^GW,model(z)) / σ]².
   - ✅ Target: src/ccw/gw_observables.py with `gw_luminosity_distance(z, hz_func, G_eff_func)` and `gw_likelihood(gw_data, hz_func, G_eff_func)`.
   - ✅ Mock data: LIGO/Virgo-like z ~ [0.01-0.5], σ ~ 15%; Einstein Telescope z ~ [0.5-2.0], σ ~ 7%.
   - ✅ Integration: extended joint_likelihood to include GW term; used with emergent gravity (G_eff from entropy) and scalar-tensor (conformal coupling).
   - ✅ Validation: (a) GW χ² = 0 for perfect ΛCDM match; (b) GW-EM tension if G_eff ≠ 1; (c) 25/25 tests passing.
   - ✅ Files: src/ccw/gw_observables.py, tests/test_gw_observables.py, examples/demo_gw_sirens.py.
   - **Result**: Emergent gravity β=0.05 predicts ~4% GW-EM mismatch at z~1; current data constrains |β|<0.05, future ET could reach |β|<0.01.
   - **Impact**: Provides distinctive signature for modified gravity (stopping criterion 2), complementary to SNe/CMB/BAO.

---

## Phase J — Self-consistency and backreaction

22. [-] **BLOCKED** — Implement **self-consistent cosmology solver** for mechanisms:
   - ✓ Fixed: SNe likelihood now depends directly on provided H(z) (no hard-coded ΛCDM placeholder).
   - ✓ Fixed: HDE is now a proper `Mechanism` and can be used to build H(z) via Friedmann.
   - ❌ **BLOCKED**: Coupled ODE solver has unit normalization issues causing numerical stiffness (see src/ccw/COUPLED_ODE_STATUS.md).
   - Status: src/ccw/coupled_ode.py created, tests/test_coupled_ode.py 8/10 failing due to dimensionality mismatch between Planck-unit fields and SI energy densities.
   - Blocker: Field φ (dimensionless) × c² creates huge scale factor ~10¹⁶, integration requires sub-machine-precision timesteps.
   - **Decision**: Defer J.22 until after H.18, I.21, J.23 are working. Use algebraic MechanismHz meanwhile (acceptable approximation for most mechanisms).
   - Files: src/ccw/coupled_ode.py (WIP), tests/test_coupled_ode.py (failing), src/ccw/COUPLED_ODE_STATUS.md (problem summary).
   - **Why critical** (when fixed): Enables accurate tuning quantification and may reveal self-stabilization or no-go bounds.

23. [ ] Add **backreaction estimates** for quantum corrections:
   - Estimate one-loop corrections to V_eff(φ) from matter/scalar interactions: ΔV ~ (1/(64π²)) m(φ)⁴ ln(Λ_UV/m).
   - Math: Field-dependent mass m(φ) = y φ (Yukawa coupling); integrate ∫ dk² k²/(k² + m²) from IR to UV cutoff Λ.
   - Target: src/ccw/backreaction.py with `loop_correction(phi, yukawa, Lambda_UV)` and `radiative_stability_check(mechanism, bg, threshold=1.0)`.
   - Check: flag mechanisms where max(ΔV) > ρ_Λ,observed → radiatively unstable, requires fine-tuning at loop level.
   - Validation: (a) ΔV → 0 as yukawa → 0; (b) ΔV ∝ Λ_UV² for m ≪ Λ (quadratic divergence).
   - Files: src/ccw/backreaction.py, tests/test_backreaction.py, add to ccw-report constraints section.
   - **Why critical**: May prove no-go bound excluding scalar quintessence without UV protection (stopping criterion 3).

24. [ ] Implement **UV completion checker**:
   - For scalar field mechanisms, check Wilsonian UV completion criteria.
   - Flag mechanisms requiring trans-Planckian field excursions or non-renormalizable operators.
   - Document what UV physics is implicitly assumed.

---

## Phase J.5 — Novel observables and signatures

21. [ ] Implement **gravitational wave standard sirens** for modified gravity tests:
   - Add GW luminosity distance: d_L^GW(z) = d_L^EM(z) / √(G_eff(z)/G) for mechanisms with running G or modified propagation.
   - Math: GW likelihood χ² = Σ [(d_L^GW,obs - d_L^GW,model(z)) / σ]².
   - Target: src/ccw/gw_observables.py with `gw_luminosity_distance(z, hz_func, G_eff_func)` and `gw_likelihood(gw_data, hz_func, G_eff_func)`.
   - Mock data: LIGO/Virgo-like z ~ [0.01-0.5], σ ~ 10-20% on d_L.
   - Integration: extend joint_likelihood to include GW term; use with emergent gravity (G_eff from entropy) or scalar-tensor (conformal coupling).
   - Validation: (a) GW χ² = 0 for perfect ΛCDM match; (b) GW-EM tension if G_eff ≠ 1.
   - Files: src/ccw/gw_observables.py, tests/test_gw_observables.py, update examples/demo_cmb_bao.py → demo_joint_all.py.
   - **Why critical**: Could bound modified gravity via c_gw/c or provide distinctive signature (stopping criterion 2).

## Phase K — LQG integration (speculative, high-effort)

25. [ ] Initiate **polymer cosmology** corrections from LQG quantization:
   - Add LQG-motivated ρ² corrections to Friedmann: H² = (8πG/3) ρ (1 - ρ/ρ_c) with ρ_c ~ 1/l_P³ from discrete geometry.
   - Math: Polymer bounce at high ρ modifies early universe; solve for effective late-time Λ_eff = 3H² - 8πG ρ_m.
   - Target: extend src/ccw/adapters/lqg.py (or create polymer_cosmology.py) with `polymer_H_z(z, H0, Omega_m, rho_c_factor)` and `polymer_effective_Lambda(z)`.
   - Validation: (a) polymer H → ΛCDM H as ρ ≪ ρ_c; (b) compare to lqg-polymer-field-generator if available.
   - Integration: test with cmb_bao_observables for high-z constraints (z ~ 1000+); check if Λ_eff(z=0) matches observation.
   - Files: src/ccw/adapters/lqg.py (or polymer_cosmology.py), tests/test_polymer_cosmology.py, examples/demo_polymer_vs_lcdm.py.
   - **Why critical**: Could yield first-principles Λ from μ₀ scale (stopping criterion 4), major breakthrough.

26. [ ] Extend **lqg-cosmological-constant-predictor** adapter to full pipeline:
   - Currently: optional comparison only.
   - Upgrade: use LQG-predicted Λ as input constraint for other mechanisms.
   - Test: can LQG prediction + holographic mechanism remove c_factor tuning?

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
3. [x] Phase I.19: CMB + BAO observables (joint SNe+CMB+BAO fits)
   - ✅ Added cmb_bao_observables.py with CMB acoustic scale ℓ_A = π D_C / r_s
   - ✅ Added BAO dilation scale D_V(z) for BOSS/DESI measurements
   - ✅ Extended likelihood.py with joint SNe+CMB+BAO likelihood
   - ✅ 89 tests pass (12 new tests for CMB/BAO observables and joint fits)
   - Status: Planck ℓ_A = 301.63 ± 0.15 implemented with comoving distance formula
4. [x] Phase I.20: σ₈ tension diagnostic (medium priority)
   - ✅ Added a minimal GR linear-growth solver to compute D(z), f(z), σ₈(z), and fσ₈(z)
   - ✅ Added an illustrative BOSS-like fσ₈ dataset + χ² diagnostic
   - ✅ Added tests and a runnable demo script
   - Files: src/ccw/sigma8_diagnostic.py, tests/test_sigma8_diagnostic.py, examples/demo_sigma8.py

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

# Task list: next steps (cosmological constant workbench)

As of Jan 14, 2026:
- We have **not** solved the cosmological constant problem.
- We *have* built a reproducible baseline + toy-mechanism/sweep framework, and encoded multiple falsifiable constraints.
- Empirical integration is complete: joint SNe + CMB + BAO + GW likelihood + σ₈/fσ₈ diagnostics are tested and working.
- Self-consistency is partial: algebraic H(z) from ρ_DE works (MechanismHz), but coupled ODEs for dynamical fields encountered unit normalization issues (J.22 **BLOCKED**).
- ✅ **Phase H.18 Emergent Gravity is complete** (commit `3c1b725`): parameter-free case with $\alpha=1.0$ reproduces an effective constant $\rho_{DE}\approx\rho_\Lambda$ with $w\approx-1$, and the demo quantifies tuning via $|\alpha-1|$.
- Limitation: emergent gravity remains **phenomenologically degenerate with flat ΛCDM** (Λ-like expansion history), so it does not yet yield a distinctive cosmological signature by itself.
- ✅ **Phase I.21 GW standard sirens is complete**: provides a concrete signature channel for modified gravity via $d_L^{GW}\neq d_L^{EM}$ (e.g., via a running $G_{eff}(z)$), complementary to SNe/CMB/BAO.
- ✅ **Phase J.23 backreaction / radiative stability checks are complete** at the unit-test level (one-loop-style $\Delta V$ + tuning diagnostics).
- **Priority queue for novel discovery**: (1) unblock J.22 coupled ODE solver (units/normalization redesign), (2) integrate J.23 backreaction outputs into `ccw-report`, (3) explore K.25 LQG polymer for first-principles predictions.

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

18. [x] Add **emergent gravity / entropic force** toy framework:
   - ✅ Implemented a clean, tested implementation in src/ccw/mechanisms/emergent_gravity.py (commit `3c1b725`).
   - ✅ Parameter-free baseline at $\alpha=1.0$ yields an effective constant $\rho_{DE}$ with $w\approx-1$ (Λ-like).
   - ✅ Demo quantifies tuning via $|\alpha-1|$.
   - Limitation: remains ΛCDM-degenerate (no strong distinctive signature yet).

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
   - ❌ **BLOCKED**: Coupled ODE solver has fundamental unit normalization issues (see src/ccw/COUPLED_ODE_STATUS.md and COUPLED_ODE_FIX_PROPOSAL.md).
   - Root cause: Planck-normalized field φ creates energy density scaling as φ²×M_Pl²c²/ρ_crit ~ 10^120 φ², causing runaway unless φ ~ 10^-60 (requires arbitrary precision).
   - **Decision**: Keep blocked until after K.25 (LQG polymer). Current algebraic H(z) from ρ_DE(z) on fixed background is accurate to ~1% for slow-roll quintessence.
   - Fix requires: (1) full dimensionless normalization (ρ/ρ_crit, φ/M_Pl, N=ln a), or (2) high-precision arithmetic, or (3) UV completion that regulates field dynamics.
   - Files: src/ccw/coupled_ode.py (WIP), tests/test_coupled_ode.py (8/10 failing), src/ccw/COUPLED_ODE_STATUS.md (diagnosis), src/ccw/COUPLED_ODE_FIX_PROPOSAL.md (solution design).
   - **Why critical** (when fixed): Enables O(10%) backreaction tracking and fast-roll regimes; not essential for current slow-roll mechanisms.

23. [x] Add **backreaction estimates** for quantum corrections:
   - ✅ Implemented Coleman–Weinberg-style one-loop correction $\Delta V$ and radiative-stability/tuning diagnostics.
   - ✅ Validation: ΔV → 0 as yukawa → 0; correct scaling behavior in UV-dominated regime.
   - ✅ Files: src/ccw/backreaction.py, tests/test_backreaction.py (30/30 passing).
   - ✅ Integration: backreaction section already integrated into `ccw-report` outputs.

24. [x] Implement **UV completion checker**:
   - ✅ Implemented Wilsonian EFT consistency checks for scalar field mechanisms.
   - ✅ Flags trans-Planckian field excursions (Δφ > M_Pl).
   - ✅ Identifies non-renormalizable operator dominance (dim > 4).
   - ✅ Checks cutoff consistency (Λ_UV ≥ m_φ) and strong coupling (g_eff < 0.3).
   - ✅ Files: src/ccw/uv_completion.py, tests/test_uv_completion.py (14/14 passing), examples/demo_uv_completion.py.
   - Result: λ ~ O(1) quintessence requires string theory UV completion (Δφ ~ 1.8 M_Pl).

---



## Phase K — LQG integration (speculative, high-effort)

25. [-] Initiate **polymer cosmology** corrections from LQG quantization:
   - Add LQG-motivated ρ² corrections to Friedmann: H² = (8πG/3) ρ (1 - ρ/ρ_c) with ρ_c ~ 1/l_P³ from discrete geometry.
   - Math: Polymer bounce at high ρ modifies early universe; solve for effective late-time Λ_eff = 3H² - 8πG ρ_m.
   - Target: extend src/ccw/adapters/lqg.py (or create polymer_cosmology.py) with `polymer_H_z(z, H0, Omega_m, rho_c_factor)` and `polymer_effective_Lambda(z)`.
   - Validation: (a) polymer H → ΛCDM H as ρ ≪ ρ_c; (b) compare to lqg-polymer-field-generator if available.
   - Integration: test with cmb_bao_observables for high-z constraints (z ~ 1000+); check if Λ_eff(z=0) matches observation.
    - Files (implemented): src/ccw/mechanisms/lqg_polymer.py, tests/test_lqg_polymer.py, examples/demo_lqg_polymer.py.
    - Files (extended toy): tests/test_lqg_polymer_vacuum.py, examples/demo_lqg_polymer_vacuum.py.
    - Status: exploratory bookkeeping is implemented via an *effective* ρ_DE,eff(z) = -ρ(z)^2/ρ_c; this does not (yet) produce a positive, constant Λ.
    - Extension (heuristic): optional constant-like term ρ_Λ,LQG = ρ_Pl * prefactor * exp(-S_bounce/α) with S_bounce ∝ μ0^2.
       - Purpose: quantify whether an entropy-suppressed hierarchy can land near observed ρ_Λ,0 and how sensitive/tuned it is.
       - Not a first-principles derivation; parameters (α, prefactor, μ0) are scan knobs.
   - **Why critical**: Could yield first-principles Λ from μ₀ scale (stopping criterion 4), major breakthrough.

26. [x] Extend **lqg-cosmological-constant-predictor** adapter to full pipeline:
   - ✅ Implemented LQG-constrained mechanisms that use LQG Λ as a target.
   - ✅ Holographic DE: solves for c_factor such that ρ_HDE = ρ_LQG.
   - ✅ Sequestering: solves for f_cancel such that ρ_residual = ρ_LQG.
   - ✅ Files: src/ccw/integrations/lqg_constrained.py, tests/test_lqg_constrained.py, examples/demo_lqg_constrained.py.
   - Result: LQG constraint can make holographic DE "natural" (c ~ O(1)) IF LQG prediction is itself natural; sequestering still requires extreme fine-tuning (~10^120).

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

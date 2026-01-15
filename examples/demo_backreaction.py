#!/usr/bin/env python3
"""
Demonstration: Backreaction and Radiative Stability Checks

This example shows how one-loop quantum corrections can destabilize scalar field
mechanisms for the cosmological constant, requiring fine-tuning at the loop level.

Key results:
1. SUSY breaking at m_SUSY = 1 TeV produces ΔV ~ 10^8 times observed ρ_Λ → extreme tuning
2. Quintessence with tiny Yukawa (y ~ 10^-20) can remain radiatively stable
3. Quadratic divergences flag mechanisms requiring UV protection

This demonstrates Phase J.23: adding radiative stability bounds to filter mechanisms.
"""

import numpy as np

from ccw.backreaction import (
    coleman_weinberg_correction,
    radiative_stability_check,
    estimate_required_tuning,
    susy_breaking_backreaction,
    quintessence_backreaction,
)


def main():
    print("=" * 70)
    print("Backreaction / Radiative Stability Checks for Cosmological Constant")
    print("=" * 70)
    print()

    # =========================================================================
    # Part 1: SUSY Breaking Scenario
    # =========================================================================
    print("Part 1: SUSY Breaking at 1 TeV")
    print("-" * 70)
    
    result_susy = susy_breaking_backreaction(m_susy_gev=1e3, lambda_uv_gev=1e16)
    
    print(f"Scenario: m_SUSY = 1 TeV, Λ_UV = 10^16 GeV (GUT scale)")
    print(f"  max(ΔV): {result_susy.max_delta_V:.3e} (SI)")
    print(f"  ρ_Λ,obs:  {result_susy.rho_lambda_observed:.3e} (SI)")
    print(f"  Tuning level: {result_susy.tuning_level:.3e}")
    tuning_factor, verdict = estimate_required_tuning(result_susy.max_delta_V, result_susy.rho_lambda_observed)
    print(f"  Verdict: {verdict}")
    print(f"  Radiatively stable: {result_susy.is_stable}")
    print(f"  Quadratic divergence flagged: {result_susy.quadratic_divergence_detected}")
    print()
    
    if not result_susy.is_stable:
        print("⚠️  SUSY breaking at 1 TeV is radiatively UNSTABLE!")
        print(f"   ΔV exceeds ρ_Λ by factor of {result_susy.tuning_level:.2e}")
        print("   Requires extreme fine-tuning to match observed Λ.")
    print()
    
    # =========================================================================
    # Part 2: Quintessence with Tiny Yukawa
    # =========================================================================
    print("Part 2: Quintessence Toy Model")
    print("-" * 70)
    
    result_quint = quintessence_backreaction(phi_today=1.0, yukawa=1e-20, lambda_uv_gev=1e3)
    
    print(f"Scenario: φ₀ = 1 M_Pl, y = 10^-20, Λ_UV = 1 TeV")
    print(f"  max(ΔV): {result_quint.max_delta_V:.3e} (SI)")
    print(f"  ρ_Λ,obs:  {result_quint.rho_lambda_observed:.3e} (SI)")
    print(f"  Tuning level: {result_quint.tuning_level:.3e}")
    tuning_factor_q, verdict_q = estimate_required_tuning(result_quint.max_delta_V, result_quint.rho_lambda_observed)
    print(f"  Verdict: {verdict_q}")
    print(f"  Radiatively stable: {result_quint.is_stable}")
    print(f"  Quadratic divergence flagged: {result_quint.quadratic_divergence_detected}")
    print()
    
    if result_quint.is_stable:
        print("✓  Quintessence with y=10^-20 is radiatively STABLE")
        print("   Loop corrections are negligible compared to ρ_Λ")
        print("   (But this requires an extremely small Yukawa coupling)")
    print()
    
    # =========================================================================
    # Part 3: Scan over Yukawa Coupling
    # =========================================================================
    print("Part 3: Yukawa Coupling Scan")
    print("-" * 70)
    
    yukawa_values = np.logspace(-25, -10, 16)
    tuning_factors = []
    
    print("Scanning Yukawa couplings y ∈ [10^-25, 10^-10]...")
    for y in yukawa_values:
        result = quintessence_backreaction(phi_today=1.0, yukawa=float(y), lambda_uv_gev=1e3)
        tuning_factors.append(result.tuning_level)
    
    print()
    print(f"{'Yukawa y':<15} {'Tuning Factor':<20} {'Verdict':<20}")
    print("-" * 55)
    for y, tf in zip(yukawa_values, tuning_factors):
        verdict = "Acceptable" if tf < 10.0 else "Fine-tuned" if tf < 1e10 else "Extreme tuning"
        print(f"{y:<15.2e} {tf:<20.2e} {verdict:<20}")
    
    print()
    print("Key finding:")
    print("  - Yukawa couplings y < 10^-18 keep ΔV below ρ_Λ (radiatively stable)")
    print("  - Larger couplings require fine-tuning to suppress loop corrections")
    print()
    
    # =========================================================================
    # Part 4: Coleman-Weinberg Correction Behavior
    # =========================================================================
    print("Part 4: Coleman-Weinberg Correction Scaling")
    print("-" * 70)
    
    phi_scan = np.linspace(0.01, 2.0, 20)
    yukawa_test = 1e-15
    lambda_uv_test = 1e3
    
    delta_v_scan = [
        coleman_weinberg_correction(phi, yukawa_test, lambda_uv_test)
        for phi in phi_scan
    ]
    
    print(f"Scanning φ ∈ [0.01, 2.0] M_Pl with y={yukawa_test:.0e}, Λ_UV={lambda_uv_test:.0e} GeV")
    print()
    print(f"{'φ (M_Pl)':<15} {'ΔV (SI)':<20}")
    print("-" * 35)
    for phi, dv in zip(phi_scan[::4], delta_v_scan[::4]):  # Show every 4th point
        print(f"{phi:<15.2f} {dv:<20.3e}")
    
    print()
    print("Key observation:")
    print(f"  - ΔV grows as φ⁴ (field-dependent mass m(φ) = y·φ)")
    print(f"  - For φ ~ M_Pl and y ~ 10^-15, ΔV ~ 10^-25 (SI)")
    print(f"  - Still below ρ_Λ ~ 6×10^-27, but approaching instability threshold")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Radiative stability analysis provides a powerful theoretical filter:")
    print()
    print("1. SUSY breaking scenarios:")
    print("   - Typically require O(10^8) fine-tuning to match ρ_Λ")
    print("   - Excluded unless protected by symmetries or UV completion")
    print()
    print("2. Quintessence scenarios:")
    print("   - Radiatively stable IF Yukawa couplings y < 10^-18")
    print("   - But this itself requires explanation (why is y so small?)")
    print()
    print("3. General lesson:")
    print("   - Loop corrections generically destabilize small vacuum energies")
    print("   - Any solution to CC problem must address radiative stability")
    print()
    print("Next steps:")
    print("  - Integrate backreaction checks into mechanism sweeps")
    print("  - Add UV completion requirements (e.g., SUSY, string embedding)")
    print("  - Explore LQG polymer quantization as alternative UV completion")
    print()


if __name__ == "__main__":
    main()

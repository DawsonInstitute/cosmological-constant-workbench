#!/usr/bin/env python3
"""Demonstration: SUSY-breaking tuning pressure.

This example quantifies the extreme fine-tuning required to match observed Λ
using a SUSY-breaking mechanism.

Key result: To reproduce ρ_Λ ~ 6×10^{-10} J/m³, we need m_SUSY ~ 10^{-3} eV,
which is ~10^12 times smaller than LHC bounds (m_SUSY > 1 TeV).

This highlights the "little hierarchy problem" within the cosmological constant context.
"""

from ccw.mechanisms import CosmologyBackground, SUSYBreaking, required_m_susy_for_observed_lambda


def main() -> None:
    print("=== SUSY-Breaking Tuning Pressure Demo ===\n")

    bg = CosmologyBackground()
    rho_obs = bg.rho_lambda0_j_m3

    print(f"Observed dark energy density: ρ_Λ = {rho_obs:.3e} J/m³\n")

    # Find required m_SUSY
    m_req_gev = required_m_susy_for_observed_lambda(rho_obs, log_enhancement=True)
    m_req_ev = m_req_gev * 1e9  # Convert to eV

    print(f"Required m_SUSY (with log enhancement): {m_req_ev:.3e} eV")
    print(f"                                       = {m_req_gev:.3e} GeV\n")

    # Compare to LHC bounds
    m_lhc_tev = 1.0  # TeV
    m_lhc_gev = m_lhc_tev * 1e3
    tuning_factor = m_lhc_gev / m_req_gev

    print(f"LHC lower bound (conservative):        m_SUSY > {m_lhc_tev} TeV = {m_lhc_gev:.0e} GeV")
    print(f"Tuning factor:                          {tuning_factor:.2e}\n")

    # Verify by evaluating mechanism
    mech = SUSYBreaking(m_susy_gev=m_req_gev, log_enhancement=True)
    rho_pred = mech.evaluate(0.0, bg).result.rho_de_j_m3

    print(f"Verification: ρ_vac(m_SUSY={m_req_gev:.3e} GeV) = {rho_pred:.3e} J/m³")
    print(f"              Relative error: {abs(rho_pred - rho_obs) / rho_obs * 100:.2f}%\n")

    # Sweep over realistic LHC-scale values
    print("Sweeping LHC-scale m_SUSY values (1-10 TeV):")
    print("  m_SUSY (GeV)     ρ_vac (J/m³)     ρ_vac / ρ_obs")
    print("  " + "-" * 50)
    for m_tev in [1.0, 2.0, 5.0, 10.0]:
        m = m_tev * 1e3
        rho = SUSYBreaking(m_susy_gev=m, log_enhancement=True).evaluate(0.0, bg).result.rho_de_j_m3
        ratio = rho / rho_obs
        print(f"  {m:.1e}       {rho:.3e}       {ratio:.3e}")

    print("\nConclusion:")
    print("  SUSY-breaking provides a *particle-physics scale* for vacuum energy,")
    print("  but requires m_SUSY ~ meV to match observations — a tuning of ~10^12.")
    print("  This is the 'little hierarchy problem' transplanted to cosmology.")
    print("  No dynamical selection mechanism is present; the tuning is explicit.")


if __name__ == "__main__":
    main()

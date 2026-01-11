from __future__ import annotations

import math

from ccw.mechanisms import CosmologyBackground, SUSYBreaking, required_m_susy_for_observed_lambda


def test_susy_breaking_produces_finite_rho() -> None:
    """SUSY mechanism should produce finite positive vacuum energy."""

    bg = CosmologyBackground()
    mech = SUSYBreaking(m_susy_gev=1e3, log_enhancement=True)
    out = mech.evaluate(0.0, bg).result

    assert math.isfinite(out.rho_de_j_m3)
    assert out.rho_de_j_m3 > 0
    assert out.w_de == -1.0


def test_susy_breaking_scales_with_m_susy_fourth_power() -> None:
    """ρ_vac should scale roughly as m_SUSY^4 (ignoring log corrections for simplicity)."""

    bg = CosmologyBackground()
    m1 = SUSYBreaking(m_susy_gev=1e3, log_enhancement=False)
    m2 = SUSYBreaking(m_susy_gev=2e3, log_enhancement=False)

    rho1 = m1.evaluate(0.0, bg).result.rho_de_j_m3
    rho2 = m2.evaluate(0.0, bg).result.rho_de_j_m3

    # Without log: rho ~ m^4, so rho2/rho1 should be ~ (2e3/1e3)^4 = 16
    ratio = rho2 / rho1
    assert 15.0 < ratio < 17.0


def test_required_m_susy_diagnostic() -> None:
    """Diagnostic function should produce sensible m_SUSY for observed ρ_Λ."""

    bg = CosmologyBackground()
    rho_obs = bg.rho_lambda0_j_m3

    m_req = required_m_susy_for_observed_lambda(rho_obs, log_enhancement=False)

    # Should be tiny (~meV scale) to match observed Λ
    assert math.isfinite(m_req)
    assert m_req > 0
    assert m_req < 1.0  # Far below GeV scale


def test_required_m_susy_roundtrip() -> None:
    """Verify that using the required m_SUSY reproduces the target ρ."""

    bg = CosmologyBackground()
    rho_target = bg.rho_lambda0_j_m3

    m_req = required_m_susy_for_observed_lambda(rho_target, log_enhancement=True)
    mech = SUSYBreaking(m_susy_gev=m_req, log_enhancement=True)
    rho_actual = mech.evaluate(0.0, bg).result.rho_de_j_m3

    # Should match within a few percent
    assert abs(rho_actual - rho_target) / rho_target < 0.05

"""Tests for UV completion checks."""

import pytest

from ccw.uv_completion import (
    UVCompletionCheck,
    check_uv_completion_scalar_field,
    planck_mass_gev,
    quintessence_uv_check,
)


def test_planck_mass_reasonable():
    m_pl = planck_mass_gev()
    # Reduced Planck mass ~ 1.22e19 GeV
    assert 1e19 < m_pl < 2e19


def test_sub_planckian_excursion_passes():
    # Sub-Planckian field excursion should pass
    m_pl = planck_mass_gev()
    result = check_uv_completion_scalar_field(
        delta_phi_gev=0.5 * m_pl,
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert result.field_excursion_ok
    assert result.delta_phi_over_m_pl < 1.0


def test_trans_planckian_excursion_fails():
    # Trans-Planckian field excursion should fail
    m_pl = planck_mass_gev()
    result = check_uv_completion_scalar_field(
        delta_phi_gev=2.0 * m_pl,
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert not result.field_excursion_ok
    assert result.delta_phi_over_m_pl > 1.0


def test_renormalizable_operators_ok():
    # Small φ/Λ should keep operators renormalizable
    m_pl = planck_mass_gev()
    result = check_uv_completion_scalar_field(
        delta_phi_gev=0.01 * 1e16,  # φ ~ 0.01 Λ_UV
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert result.operators_ok
    assert result.max_operator_dim == 4


def test_nonrenormalizable_operators_flagged():
    # Large φ/Λ should flag non-renormalizable operators
    m_pl = planck_mass_gev()
    result = check_uv_completion_scalar_field(
        delta_phi_gev=2.0 * 1e16,  # φ ~ 2 Λ_UV
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert not result.operators_ok
    assert result.max_operator_dim is not None
    assert result.max_operator_dim >= 6


def test_cutoff_above_mass_is_consistent():
    # Λ_UV > m_φ should pass cutoff consistency
    result = check_uv_completion_scalar_field(
        delta_phi_gev=1e15,
        mass_gev=1e3,
        cutoff_gev=1e16,
    )
    assert result.cutoff_consistent


def test_cutoff_below_mass_is_inconsistent():
    # Λ_UV < m_φ should fail cutoff consistency
    result = check_uv_completion_scalar_field(
        delta_phi_gev=1e15,
        mass_gev=1e17,
        cutoff_gev=1e16,
    )
    assert not result.cutoff_consistent


def test_weak_coupling_ok():
    # g_eff = φ/Λ ≪ 1 should pass strong coupling check
    result = check_uv_completion_scalar_field(
        delta_phi_gev=1e14,  # φ/Λ = 0.01
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert result.strong_coupling_ok
    assert result.g_eff < 0.3


def test_strong_coupling_fails():
    # g_eff = φ/Λ ~ 1 should fail strong coupling check
    result = check_uv_completion_scalar_field(
        delta_phi_gev=0.8 * 1e16,  # φ/Λ = 0.8
        mass_gev=1e-33,
        cutoff_gev=1e16,
    )
    assert not result.strong_coupling_ok
    assert result.g_eff >= 0.3


def test_overall_uv_complete_verdict():
    # All checks passing → UV complete
    m_pl = planck_mass_gev()
    result = check_uv_completion_scalar_field(
        delta_phi_gev=0.1 * m_pl,  # Sub-Planckian
        mass_gev=1e-33,  # < cutoff
        cutoff_gev=1e16,  # φ/Λ ~ 1e3
    )
    # Note: φ ~ 0.1 M_Pl ~ 1e18 GeV, Λ ~ 1e16 GeV → φ/Λ ~ 100, so strong coupling will fail
    # Let's use a higher cutoff
    result = check_uv_completion_scalar_field(
        delta_phi_gev=0.1 * m_pl,
        mass_gev=1e-33,
        cutoff_gev=1e19,  # φ/Λ ~ 0.1
    )
    assert result.uv_complete


def test_quintessence_lambda_1_trans_planckian():
    # λ=1, z_max=5 → Δφ ~ 1.8 M_Pl (trans-Planckian)
    result = quintessence_uv_check(lambda_coupling=1.0, z_max=5.0)
    assert not result.field_excursion_ok
    assert result.delta_phi_over_m_pl > 1.0


def test_quintessence_lambda_01_sub_planckian():
    # λ=0.1, z_max=5 → Δφ ~ 0.18 M_Pl (sub-Planckian)
    result = quintessence_uv_check(lambda_coupling=0.1, z_max=5.0)
    assert result.field_excursion_ok
    assert result.delta_phi_over_m_pl < 1.0


def test_quintessence_details_string_nonempty():
    result = quintessence_uv_check(lambda_coupling=1.0, z_max=5.0)
    assert len(result.details) > 0
    assert "Field excursion" in result.details
    assert "Operator dimension" in result.details
    assert "Cutoff" in result.details
    assert "Strong coupling" in result.details


def test_invalid_inputs_raise_errors():
    with pytest.raises(ValueError, match="delta_phi_gev must be non-negative"):
        check_uv_completion_scalar_field(delta_phi_gev=-1.0, mass_gev=1.0, cutoff_gev=1e16)

    with pytest.raises(ValueError, match="mass_gev must be positive"):
        check_uv_completion_scalar_field(delta_phi_gev=1.0, mass_gev=0.0, cutoff_gev=1e16)

    with pytest.raises(ValueError, match="cutoff_gev must be positive"):
        check_uv_completion_scalar_field(delta_phi_gev=1.0, mass_gev=1.0, cutoff_gev=-1.0)

    with pytest.raises(ValueError, match="g_eff_threshold must be in"):
        check_uv_completion_scalar_field(
            delta_phi_gev=1.0, mass_gev=1.0, cutoff_gev=1e16, g_eff_threshold=1.5
        )

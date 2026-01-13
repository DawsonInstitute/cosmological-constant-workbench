"""
Tests for emergent gravity dark energy mechanism.
"""

import numpy as np
import pytest

from ccw.mechanisms.base import CosmologyBackground
from ccw.mechanisms.emergent_gravity import EmergentGravity


def test_emergent_gravity_initialization():
    """Verify EmergentGravity initializes correctly."""
    mech = EmergentGravity(alpha=1.0)
    
    assert mech.alpha == 1.0
    assert mech.name == "emergent_gravity"


def test_emergent_gravity_invalid_alpha():
    """Verify negative alpha raises error."""
    with pytest.raises(ValueError, match="alpha must be positive"):
        EmergentGravity(alpha=-0.5)


def test_emergent_gravity_evaluate():
    """Verify evaluate method works."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3, omega_lambda=0.7)
    
    output = mech.evaluate(z=0.0, bg=bg)
    
    assert output.result.z == 0.0
    assert output.result.rho_de_j_m3 > 0
    assert output.result.w_de == -1.0
    assert "entropic" in output.assumptions.lower()


def test_emergent_gravity_rho_de_positive():
    """Verify dark energy density is positive."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground(omega_m=0.3, omega_lambda=0.7)
    
    output_0 = mech.evaluate(z=0.0, bg=bg)
    output_1 = mech.evaluate(z=1.0, bg=bg)
    
    assert output_0.result.rho_de_j_m3 > 0
    assert output_1.result.rho_de_j_m3 > 0


def test_emergent_gravity_rho_de_approximately_constant():
    """Verify emergent DE density is approximately constant (Λ-like)."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground(omega_m=0.3, omega_lambda=0.7)
    
    z_arr = [0.0, 0.5, 1.0, 1.5, 2.0]
    rho_arr = [mech.evaluate(z, bg).result.rho_de_j_m3 for z in z_arr]
    
    # Should be exactly constant for this model
    rho_variation = (max(rho_arr) - min(rho_arr)) / np.mean(rho_arr)
    assert rho_variation < 1e-10, "ρ_DE should be constant for emergent Λ"


def test_emergent_gravity_w_minus_one():
    """Verify equation of state is -1 (cosmological constant)."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground(omega_m=0.3, omega_lambda=0.7)
    
    w_0 = mech.evaluate(z=0.0, bg=bg).result.w_de
    w_1 = mech.evaluate(z=1.0, bg=bg).result.w_de
    
    assert w_0 == -1.0
    assert w_1 == -1.0


def test_emergent_gravity_alpha_scaling():
    """Verify ρ_DE scales with alpha."""
    mech1 = EmergentGravity(alpha=1.0)
    mech2 = EmergentGravity(alpha=2.0)
    bg = CosmologyBackground(omega_m=0.3, omega_lambda=0.7)
    
    rho1 = mech1.evaluate(z=0.0, bg=bg).result.rho_de_j_m3
    rho2 = mech2.evaluate(z=0.0, bg=bg).result.rho_de_j_m3
    
    # ρ_DE should scale linearly with alpha
    np.testing.assert_allclose(rho2 / rho1, 2.0, rtol=0.01)


def test_emergent_gravity_parameter_free():
    """Verify alpha=1 gives parameter-free prediction."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground(omega_m=0.3, omega_r=0.0, omega_lambda=0.7)
    
    output = mech.evaluate(z=0.0, bg=bg)
    
    # For alpha=1, should use all "missing" density as emergent DE
    # Omega_DE,eff = 1 - 0.3 - 0.0 = 0.7
    # Should match bg.omega_lambda
    expected_rho_de = bg.rho_lambda0_j_m3
    
    # Since we compute omega_Lambda_eff = alpha * (1 - omega_m - omega_r) = 1.0 * 0.7
    # and rho_de = omega_Lambda_eff * rho_lambda0 / omega_lambda = 0.7 * rho_lambda0 / 0.7
    np.testing.assert_allclose(output.result.rho_de_j_m3, expected_rho_de, rtol=1e-10)


def test_emergent_gravity_describe_assumptions():
    """Verify describe_assumptions returns meaningful text."""
    mech = EmergentGravity(alpha=1.5)
    
    desc = mech.describe_assumptions()
    assert "entropic" in desc.lower()
    assert "1.50" in desc or "1.5" in desc  # alpha value
    assert "verlinde" in desc.lower()


def test_emergent_gravity_different_backgrounds():
    """Verify mechanism works with different background cosmologies."""
    mech = EmergentGravity(alpha=1.0)
    
    bg1 = CosmologyBackground(omega_m=0.3, omega_lambda=0.7)
    bg2 = CosmologyBackground(omega_m=0.25, omega_lambda=0.75)
    
    output1 = mech.evaluate(z=0.5, bg=bg1)
    output2 = mech.evaluate(z=0.5, bg=bg2)
    
    # Different backgrounds should give different rho_DE
    assert output1.result.rho_de_j_m3 != output2.result.rho_de_j_m3


def test_emergent_gravity_nonnegative_z():
    """Verify negative z raises error."""
    mech = EmergentGravity(alpha=1.0)
    bg = CosmologyBackground()
    
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        mech.evaluate(z=-0.5, bg=bg)

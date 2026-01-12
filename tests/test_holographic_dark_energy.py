"""
Tests for holographic dark energy mechanism.
"""

import numpy as np
import pytest
from src.ccw.mechanisms import HolographicDarkEnergy


def test_holographic_hubble_cutoff_produces_finite_density():
    """Verify that hubble cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="hubble")
    z_values = np.array([0.0, 0.5, 1.0, 2.0])
    rho_de = hde.evaluate(z_values)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_hubble_cutoff_density_increases_with_z():
    """Verify that hubble cutoff density increases with increasing z (for ΛCDM background)."""
    hde = HolographicDarkEnergy(cutoff_type="hubble")
    z_values = np.array([0.0, 0.5, 1.0, 2.0])
    rho_de = hde.evaluate(z_values)

    # For L = c/H(z) and H increasing with z, L decreases, so ρ ~ 1/L^2 increases
    # (This is specific to Hubble cutoff with ΛCDM background)
    assert rho_de[0] < rho_de[1] < rho_de[2] < rho_de[3], (
        "Hubble cutoff density should increase with z"
    )


def test_holographic_particle_horizon_cutoff_produces_finite_density():
    """Verify that particle horizon cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="particle_horizon")
    z_values = np.array([0.0, 0.5, 1.0])
    rho_de = hde.evaluate(z_values)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_event_horizon_cutoff_produces_finite_density():
    """Verify that event horizon cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="event_horizon")
    z_values = np.array([0.0, 0.5, 1.0])
    rho_de = hde.evaluate(z_values)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_c_factor_scales_density():
    """Verify that c_factor scales the density (inversely with L^2)."""
    hde1 = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.0)
    hde2 = HolographicDarkEnergy(cutoff_type="hubble", c_factor=2.0)

    z = 0.5
    rho1 = hde1.evaluate(np.array([z]))[0]
    rho2 = hde2.evaluate(np.array([z]))[0]

    # L2 = 2 * L1 → ρ2 = ρ1 / 4 (since ρ ~ 1/L^2)
    expected_ratio = 1.0 / 4.0
    actual_ratio = rho2 / rho1

    np.testing.assert_allclose(
        actual_ratio,
        expected_ratio,
        rtol=1e-6,
        err_msg="c_factor should scale density as 1/c_factor^2",
    )


def test_holographic_unknown_cutoff_type_raises():
    """Verify that unknown cutoff type raises ValueError."""
    hde = HolographicDarkEnergy(cutoff_type="unknown")  # type: ignore

    with pytest.raises(ValueError, match="Unknown cutoff_type"):
        hde.evaluate(np.array([0.0]))


def test_holographic_describe_assumptions():
    """Verify that describe_assumptions returns a non-empty string."""
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.5)
    desc = hde.describe_assumptions()

    assert isinstance(desc, str), "Description should be a string"
    assert len(desc) > 0, "Description should be non-empty"
    assert "hubble" in desc, "Description should mention cutoff type"
    assert "1.5" in desc, "Description should mention c_factor"


def test_holographic_density_order_of_magnitude():
    """Verify that hubble cutoff density is reasonable order of magnitude."""
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.0)
    rho_de_z0 = hde.evaluate(np.array([0.0]))[0]

    # Observed dark energy density ~ 5-6 × 10^-10 J/m^3
    # Hubble cutoff with c_factor=1 should give same order of magnitude
    # (For ΛCDM background, L ~ c/H0 ~ 10^26 m → ρ ~ 10^-9 J/m^3)
    assert 1e-11 < rho_de_z0 < 1e-8, (
        f"Density order of magnitude unexpected: {rho_de_z0:.3e} J/m^3"
    )

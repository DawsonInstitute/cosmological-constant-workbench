"""Tests for holographic dark energy mechanism."""

import numpy as np
import pytest

from ccw.mechanisms import CosmologyBackground, HolographicDarkEnergy


def test_holographic_hubble_cutoff_produces_finite_density():
    """Verify that hubble cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    z_values = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    rho_de = np.array([hde.evaluate(float(z), bg).result.rho_de_j_m3 for z in z_values], dtype=float)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_hubble_cutoff_density_increases_with_z():
    """Verify that hubble cutoff density increases with increasing z (for ΛCDM background)."""
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    z_values = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    rho_de = np.array([hde.evaluate(float(z), bg).result.rho_de_j_m3 for z in z_values], dtype=float)

    # For L = c/H(z) and H increasing with z, L decreases, so ρ ~ 1/L^2 increases
    # (This is specific to Hubble cutoff with ΛCDM background)
    assert rho_de[0] < rho_de[1] < rho_de[2] < rho_de[3], (
        "Hubble cutoff density should increase with z"
    )


def test_holographic_particle_horizon_cutoff_produces_finite_density():
    """Verify that particle horizon cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="particle_horizon")
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    z_values = np.array([0.0, 0.5, 1.0], dtype=float)
    rho_de = np.array([hde.evaluate(float(z), bg).result.rho_de_j_m3 for z in z_values], dtype=float)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_event_horizon_cutoff_produces_finite_density():
    """Verify that event horizon cutoff produces finite, positive density."""
    hde = HolographicDarkEnergy(cutoff_type="event_horizon")
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    z_values = np.array([0.0, 0.5, 1.0], dtype=float)
    rho_de = np.array([hde.evaluate(float(z), bg).result.rho_de_j_m3 for z in z_values], dtype=float)

    assert np.all(np.isfinite(rho_de)), "Density should be finite"
    assert np.all(rho_de > 0), "Density should be positive"


def test_holographic_c_factor_scales_density():
    """Verify that c_factor scales the density (inversely with L^2)."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    hde1 = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    hde2 = HolographicDarkEnergy(cutoff_type="hubble", c_factor=2.4)

    z = 0.5
    rho1 = hde1.evaluate(z, bg).result.rho_de_j_m3
    rho2 = hde2.evaluate(z, bg).result.rho_de_j_m3

    assert rho2 < rho1


def test_holographic_unknown_cutoff_type_raises():
    """Verify that unknown cutoff type raises ValueError."""
    hde = HolographicDarkEnergy(cutoff_type="unknown")  # type: ignore
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    with pytest.raises(ValueError, match="Unknown cutoff_type"):
        hde.evaluate(0.0, bg)


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
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    rho_de_z0 = hde.evaluate(0.0, bg).result.rho_de_j_m3

    # Observed dark energy density ~ 5-6 × 10^-10 J/m^3
    # Hubble cutoff with c_factor=1 should give same order of magnitude
    # (For ΛCDM background, L ~ c/H0 ~ 10^26 m → ρ ~ 10^-9 J/m^3)
    assert 1e-11 < rho_de_z0 < 1e-8, (
        f"Density order of magnitude unexpected: {rho_de_z0:.3e} J/m^3"
    )

"""
Tests for CMB and BAO observables.
"""

import numpy as np
import pytest
from ccw.cmb_bao_observables import (
    angular_diameter_distance_mpc,
    cmb_acoustic_scale_ell_a,
    dilation_scale_dv,
    get_planck_cmb_observable,
    get_boss_bao_observables,
    get_desi_bao_observables,
)
from ccw.mechanisms import CosmologyBackground
from ccw.frw import h_z_lcdm_s_inv


def test_angular_diameter_distance_positive():
    """Verify D_A(z) is positive and finite."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_callable(z):
        return h_z_lcdm_s_inv(z, bg)
    
    d_a = angular_diameter_distance_mpc(1.0, hz_callable)
    
    assert d_a > 0, "D_A should be positive"
    assert np.isfinite(d_a), "D_A should be finite"


def test_angular_diameter_distance_cmb():
    """Verify D_A at CMB redshift has reasonable value."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_callable(z):
        return h_z_lcdm_s_inv(z, bg)
    
    d_a_cmb = angular_diameter_distance_mpc(1090, hz_callable)
    
    # D_A(z_CMB) ~ 13 Mpc for Planck ΛCDM (proper distance, very small!)
    # This is correct - angular diameter distance decreases at high z
    assert 10 < d_a_cmb < 15, f"D_A(CMB) = {d_a_cmb:.1f} Mpc"


def test_cmb_acoustic_scale_order_of_magnitude():
    """Verify ℓ_A has correct order of magnitude."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_callable(z):
        return h_z_lcdm_s_inv(z, bg)
    
    from ccw.cmb_bao_observables import cmb_acoustic_scale_ell_a
    ell_a = cmb_acoustic_scale_ell_a(1090, hz_callable, r_s_mpc=147.0)
    
    # ℓ_A = π D_C / r_s ~ π * 14000 / 147 ~ 299
    # Planck 2018: ℓ_A = 301.63 ± 0.15
    assert 250 < ell_a < 350, f"ℓ_A = {ell_a:.2f} (expected ~300)"


def test_dilation_scale_positive():
    """Verify D_V is positive and finite."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_callable(z):
        return h_z_lcdm_s_inv(z, bg)
    
    d_v = dilation_scale_dv(0.5, hz_callable)
    
    assert d_v > 0, "D_V should be positive"
    assert np.isfinite(d_v), "D_V should be finite"


def test_dilation_scale_increases_with_z():
    """Verify D_V increases with z (for standard cosmology)."""
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_callable(z):
        return h_z_lcdm_s_inv(z, bg)
    
    d_v_0p3 = dilation_scale_dv(0.3, hz_callable)
    d_v_0p5 = dilation_scale_dv(0.5, hz_callable)
    d_v_0p7 = dilation_scale_dv(0.7, hz_callable)
    
    assert d_v_0p3 < d_v_0p5 < d_v_0p7, "D_V should increase with z"


def test_planck_cmb_observable():
    """Verify Planck CMB observable has correct values."""
    cmb_obs = get_planck_cmb_observable()
    
    assert cmb_obs.z_star > 1000, "z_* should be > 1000"
    assert 250 < cmb_obs.ell_a < 350, f"ℓ_A ~ 301.63 (got {cmb_obs.ell_a})"
    assert cmb_obs.sigma_ell_a > 0, "Uncertainty should be positive"
    assert cmb_obs.r_s_mpc > 100, "r_s should be ~ 147 Mpc"


def test_boss_bao_observables():
    """Verify BOSS BAO observables load correctly."""
    bao_obs = get_boss_bao_observables()
    
    assert len(bao_obs) > 0, "Should have BAO measurements"
    assert all(obs.measurement_type == "DV" for obs in bao_obs), "BOSS provides D_V"
    assert all(obs.sigma > 0 for obs in bao_obs), "Uncertainties should be positive"
    assert all(obs.value > 1000 for obs in bao_obs), "D_V should be > 1000 Mpc"


def test_desi_bao_observables():
    """Verify DESI mock BAO observables load correctly."""
    bao_obs = get_desi_bao_observables()
    
    assert len(bao_obs) > 0, "Should have BAO measurements"
    assert all(obs.sigma > 0 for obs in bao_obs), "Uncertainties should be positive"
    # DESI should have better precision than BOSS
    assert all(obs.sigma < 50 for obs in bao_obs), "DESI precision should be < 50 Mpc"

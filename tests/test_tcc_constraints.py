"""
Tests for TCC constraints.
"""

import numpy as np
import pytest
from src.ccw.tcc_constraints import (
    check_tcc_lcdm,
    check_tcc_inflation,
    check_tcc_hubble_bound,
    h0_km_s_mpc_to_gev,
    LAMBDA_TCC_GEV,
)


def test_h0_conversion():
    """Verify H0 conversion from km/s/Mpc to GeV."""
    h0_km_s_mpc = 67.4
    h0_gev = h0_km_s_mpc_to_gev(h0_km_s_mpc)
    
    # H0 ~ 67 km/s/Mpc ~ 1.4×10^-42 GeV
    assert 1e-43 < h0_gev < 1e-41, f"H0 conversion: {h0_gev:.3e} GeV"
    assert h0_gev > 0, "H0 should be positive"


def test_tcc_lcdm_satisfies():
    """Verify that ΛCDM satisfies TCC (late-time evolution)."""
    result = check_tcc_lcdm(h0_km_s_mpc=67.4, omega_m=0.3, z_max=1100.0)
    
    assert result.satisfies_tcc, "ΛCDM should satisfy TCC for late-time evolution"
    assert result.h_max_gev < LAMBDA_TCC_GEV, "H_max should be below TCC bound"
    assert result.margin < 1.0, "Margin should be < 1 (satisfies bound)"


def test_tcc_lcdm_h_increases_with_z():
    """Verify that H(z) increases with z for ΛCDM."""
    h0_gev = h0_km_s_mpc_to_gev(67.4)
    omega_m = 0.3
    omega_lambda = 0.7
    
    # H(z=0)
    h_z0 = h0_gev * np.sqrt(omega_m + omega_lambda)
    
    # H(z=1)
    h_z1 = h0_gev * np.sqrt(omega_m * 2**3 + omega_lambda)
    
    assert h_z1 > h_z0, "H(z) should increase with z"


def test_tcc_standard_inflation_violates():
    """Verify that standard high-scale inflation violates TCC."""
    h_inf_gev = 1e14  # GUT-scale inflation
    result = check_tcc_inflation(h_inf_gev, n_efolds=60.0)
    
    assert not result.satisfies_tcc, "Standard inflation should violate TCC"
    assert result.h_max_gev > LAMBDA_TCC_GEV, "H_inf should exceed TCC bound"
    assert result.margin > 1.0, "Margin should be >> 1 (violates bound)"


def test_tcc_low_scale_inflation_may_satisfy():
    """Verify that low-scale inflation may satisfy TCC."""
    h_inf_gev = 1e-13  # Very low-scale inflation
    result = check_tcc_inflation(h_inf_gev, n_efolds=60.0)
    
    assert result.satisfies_tcc, "Low-scale inflation may satisfy TCC"
    assert result.h_max_gev <= LAMBDA_TCC_GEV


def test_tcc_hubble_bound_custom():
    """Verify TCC check with custom H(z) function."""
    # Constant H(z) = 10^-13 GeV (below TCC)
    def hz_gev(z):
        return 1e-13
    
    z_values = np.array([0.0, 1.0, 10.0])
    result = check_tcc_hubble_bound(hz_gev, z_values, lambda_tcc_gev=LAMBDA_TCC_GEV)
    
    assert result.satisfies_tcc, "Constant H below TCC should satisfy"
    assert result.h_max_gev == 1e-13


def test_tcc_hubble_bound_violation():
    """Verify TCC check detects violations."""
    # H(z) = 10^-11 GeV (above TCC)
    def hz_gev(z):
        return 1e-11
    
    z_values = np.array([0.0, 1.0, 10.0])
    result = check_tcc_hubble_bound(hz_gev, z_values, lambda_tcc_gev=LAMBDA_TCC_GEV)
    
    assert not result.satisfies_tcc, "H above TCC should violate"
    assert result.h_max_gev == 1e-11
    assert result.margin > 1.0

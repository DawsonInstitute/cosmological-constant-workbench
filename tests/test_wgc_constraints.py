"""
Tests for WGC constraints.
"""

import numpy as np
import pytest
from src.ccw.wgc_constraints import (
    check_wgc_scalar,
    check_wgc_quintessence_exponential,
    swampland_distance_conjecture_mass,
    check_sdc_tower,
    M_PLANCK_GEV,
)


def test_wgc_scalar_satisfies():
    """Verify WGC check passes for m < g M_Pl."""
    mass_gev = 1e10  # 10 GeV
    coupling_g = 1e-8  # Weak coupling
    
    result = check_wgc_scalar(mass_gev, coupling_g)
    
    # g M_Pl = 1e-8 × 1.22e19 = 1.22e11 GeV > 1e10 GeV
    assert result.satisfies_wgc, "Should satisfy WGC for m < g M_Pl"
    assert result.bound_ratio < 1.0


def test_wgc_scalar_violates():
    """Verify WGC check fails for m > g M_Pl."""
    mass_gev = 1e15  # Heavy scalar
    coupling_g = 1e-10  # Very weak coupling
    
    result = check_wgc_scalar(mass_gev, coupling_g)
    
    # g M_Pl = 1e-10 × 1.22e19 = 1.22e9 GeV < 1e15 GeV
    assert not result.satisfies_wgc, "Should violate WGC for m > g M_Pl"
    assert result.bound_ratio > 1.0


def test_wgc_quintessence_steep_potential():
    """Verify steep exponential potential satisfies WGC."""
    lam = 2.0  # Steep (swampland-favored)
    v0_gev4 = 1e-47 * (1.22e19)**4  # Dark energy scale in GeV^4
    coupling_g = 1e-5
    
    result = check_wgc_quintessence_exponential(lam, v0_gev4, coupling_g)
    
    # Steep potentials → small effective mass → likely satisfies WGC
    assert result.mass_gev > 0, "Mass should be positive"
    # WGC likely satisfied for reasonable coupling
    assert result.satisfies_wgc, "Steep potential with reasonable coupling should satisfy WGC"


def test_wgc_quintessence_flat_potential():
    """Verify flat potential (small λ) with weak coupling."""
    lam = 0.1  # Flat (swampland-disfavored)
    v0_gev4 = 1e-47 * (1.22e19)**4
    coupling_g = 1e-10  # Weak coupling
    
    result = check_wgc_quintessence_exponential(lam, v0_gev4, coupling_g)
    
    # Even flat potentials satisfy WGC if coupling is not too tiny
    assert result.mass_gev > 0, "Mass should be positive"
    # Check that result is computed (may pass or fail depending on parameters)
    assert result.bound_ratio > 0, "Bound ratio should be positive"


def test_sdc_tower_mass_decreases_exponentially():
    """Verify SDC tower mass decreases exponentially with field excursion."""
    m_tower_1 = swampland_distance_conjecture_mass(delta_phi_mpl=1.0)
    m_tower_2 = swampland_distance_conjecture_mass(delta_phi_mpl=2.0)
    m_tower_3 = swampland_distance_conjecture_mass(delta_phi_mpl=3.0)
    
    assert m_tower_1 > m_tower_2 > m_tower_3, "Tower mass should decrease with Δφ"
    
    # Check exponential scaling
    ratio = m_tower_1 / m_tower_2
    assert np.isclose(ratio, np.exp(1.0), rtol=0.01), "Should scale as exp(Δφ)"


def test_sdc_tower_trans_planckian():
    """Verify trans-Planckian excursion (Δφ > M_Pl) produces light tower."""
    delta_phi_mpl = 5.0  # 5 M_Pl excursion
    m_tower = swampland_distance_conjecture_mass(delta_phi_mpl)
    
    # m_tower ~ M_Pl exp(-5) ~ M_Pl / 148 ~ 8×10^16 GeV
    assert m_tower < M_PLANCK_GEV, "Tower should be lighter than M_Pl for Δφ > 1"
    assert m_tower > M_PLANCK_GEV / 200, "Order of magnitude check"


def test_sdc_eft_validity_check():
    """Verify SDC EFT validity checker."""
    # Sub-Planckian excursion with high cutoff → EFT valid
    result_valid = check_sdc_tower(delta_phi_mpl=0.5, cutoff_gev=1e10)
    assert result_valid["eft_valid"], "EFT should be valid for small Δφ"
    
    # Trans-Planckian excursion with low cutoff → EFT breaks down
    result_invalid = check_sdc_tower(delta_phi_mpl=10.0, cutoff_gev=1e18)
    assert not result_invalid["eft_valid"], "EFT should break down for large Δφ"


def test_wgc_zero_mass_requires_zero_coupling():
    """Verify m→0 requires g→0 (ΛCDM limit)."""
    mass_gev = 1e-20  # Nearly massless
    coupling_g = 1e-40  # Must be extremely tiny
    
    result = check_wgc_scalar(mass_gev, coupling_g)
    
    # g M_Pl = 1e-40 × 1.22e19 = 1.22e-21 GeV > 1e-20 GeV
    # This should fail! m > g M_Pl for m ~ 1e-20, g ~ 1e-40
    assert not result.satisfies_wgc, "Very small mass with tiny coupling should violate"
    
    # To satisfy, need g > m / M_Pl = 1e-20 / 1.22e19 ~ 8e-40
    coupling_g_needed = 1e-39
    result_pass = check_wgc_scalar(mass_gev, coupling_g_needed)
    assert result_pass.satisfies_wgc, "Larger coupling should satisfy"

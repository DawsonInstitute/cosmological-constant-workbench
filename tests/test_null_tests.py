"""
Tests for null test harness.
"""

import numpy as np
import pytest
from src.ccw.null_tests import (
    NullTestHarness,
    NullTestBound,
    evaluate_mechanism_with_null_tests,
)
from src.ccw.mechanisms import HolographicDarkEnergy, CosmologyBackground


def test_null_test_harness_initialization():
    """Verify harness initializes with default bounds."""
    harness = NullTestHarness()
    
    assert len(harness.bounds) > 0, "Should have default bounds"
    bound_names = [b.name for b in harness.bounds]
    assert "w_today_approx_minus_one" in bound_names
    assert "rho_de_positive" in bound_names
    assert "hz_positive_finite" in bound_names


def test_null_test_passes_reasonable_values():
    """Verify null tests pass for reasonable mechanism results."""
    harness = NullTestHarness()
    
    # Reasonable ΛCDM-like results
    mechanism_result = {
        "rho_de_z0": 5.3e-10,  # Observed value
        "w_de_z0": -1.0,  # Cosmological constant
        "rho_de_values": [5.3e-10, 5.3e-10, 5.3e-10],
        "hz_values": [1.0, 1.2, 1.5],  # Increasing with z
        "z_values": [0.0, 0.5, 1.0],
    }
    
    result = harness.run_tests(mechanism_result)
    
    assert result.passed, "Should pass all tests for reasonable values"
    assert len(result.failed_bounds) == 0


def test_null_test_fails_extreme_w():
    """Verify null tests fail for extreme w values."""
    harness = NullTestHarness()
    
    # Extreme w value
    mechanism_result = {
        "rho_de_z0": 5.3e-10,
        "w_de_z0": -2.5,  # Too negative
        "rho_de_values": [5.3e-10] * 3,
        "hz_values": [1.0, 1.2, 1.5],
        "z_values": [0.0, 0.5, 1.0],
    }
    
    result = harness.run_tests(mechanism_result)
    
    assert not result.passed, "Should fail for extreme w"
    assert "w_today_approx_minus_one" in result.failed_bounds


def test_null_test_fails_negative_rho():
    """Verify null tests fail for negative ρ_DE."""
    harness = NullTestHarness()
    
    # Negative density
    mechanism_result = {
        "rho_de_z0": 5.3e-10,
        "w_de_z0": -1.0,
        "rho_de_values": [5.3e-10, -1e-10, 5.3e-10],  # Negative at z=0.5
        "hz_values": [1.0, 1.2, 1.5],
        "z_values": [0.0, 0.5, 1.0],
    }
    
    result = harness.run_tests(mechanism_result)
    
    assert not result.passed, "Should fail for negative ρ_DE"
    assert "rho_de_positive" in result.failed_bounds


def test_null_test_fails_nonmonotonic_hz():
    """Verify null tests fail for non-monotonic H(z)."""
    harness = NullTestHarness()
    
    # Non-monotonic H(z)
    mechanism_result = {
        "rho_de_z0": 5.3e-10,
        "w_de_z0": -1.0,
        "rho_de_values": [5.3e-10] * 3,
        "hz_values": [1.0, 1.5, 1.2],  # Decreases from z=0.5 to z=1.0
        "z_values": [0.0, 0.5, 1.0],
    }
    
    result = harness.run_tests(mechanism_result)
    
    assert not result.passed, "Should fail for non-monotonic H(z)"
    assert "hz_monotonic_increasing" in result.failed_bounds


def test_null_test_fails_extreme_rho_today():
    """Verify null tests fail for extreme ρ_DE(z=0)."""
    harness = NullTestHarness()
    
    # ρ_DE >> observed
    mechanism_result = {
        "rho_de_z0": 5.3e-5,  # 10^5 × observed
        "w_de_z0": -1.0,
        "rho_de_values": [5.3e-5] * 3,
        "hz_values": [1.0, 1.2, 1.5],
        "z_values": [0.0, 0.5, 1.0],
    }
    
    result = harness.run_tests(mechanism_result)
    
    assert not result.passed, "Should fail for extreme ρ_DE(z=0)"
    assert "rho_today_order_of_magnitude" in result.failed_bounds


def test_null_test_custom_bound():
    """Verify custom bounds can be added."""
    harness = NullTestHarness()
    
    # Add custom bound
    def check_custom(mechanism_result):
        return mechanism_result.get("rho_de_z0", 0) > 1e-20
    
    custom_bound = NullTestBound(
        name="custom_test",
        description="Custom test description",
        check_function=check_custom,
        failure_message="Custom test failed",
    )
    
    harness.add_custom_bound(custom_bound)
    
    # Test passes
    result = harness.run_tests({"rho_de_z0": 5.3e-10})
    assert "custom_test" in result.details
    assert result.details["custom_test"]["passed"]


def test_evaluate_mechanism_with_null_tests():
    """Verify end-to-end mechanism evaluation with null tests."""
    hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    z_values = np.array([0.0, 0.5, 1.0])
    
    result = evaluate_mechanism_with_null_tests(hde, bg, z_values)
    
    assert "mechanism_results" in result
    assert "null_tests" in result
    assert "passed" in result["null_tests"]
    assert isinstance(result["null_tests"]["passed"], bool)

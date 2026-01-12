"""
Tests for data loader.
"""

import pytest
from ccw.data_loader import load_pantheon_plus_subset, DistanceModulusPoint


def test_load_builtin_dataset():
    """Verify that builtin dataset loads successfully."""
    data = load_pantheon_plus_subset(max_points=50)
    
    assert len(data) > 0, "Dataset should not be empty"
    assert len(data) <= 50, "Should respect max_points limit"
    assert all(isinstance(p, DistanceModulusPoint) for p in data), "All points should be DistanceModulusPoint"


def test_builtin_dataset_has_reasonable_redshifts():
    """Verify redshifts are in reasonable cosmological range."""
    data = load_pantheon_plus_subset()
    
    z_values = [p.z for p in data]
    assert min(z_values) > 0, "All redshifts should be positive"
    assert max(z_values) <= 2.5, "Max redshift should be < 2.5 (typical SNe Ia range)"
    assert z_values == sorted(z_values), "Redshifts should be sorted"


def test_builtin_dataset_distance_modulus_increases():
    """Verify distance modulus increases with redshift."""
    data = load_pantheon_plus_subset()
    
    mu_values = [p.mu for p in data]
    # Distance modulus should be monotonically increasing for expanding universe
    for i in range(len(mu_values) - 1):
        assert mu_values[i] < mu_values[i + 1], f"μ should increase: μ({data[i].z}) < μ({data[i+1].z})"


def test_builtin_dataset_uncertainties_positive():
    """Verify all uncertainties are positive."""
    data = load_pantheon_plus_subset()
    
    for point in data:
        assert point.sigma_mu > 0, f"Uncertainty should be positive at z={point.z}"


def test_max_points_limit():
    """Verify max_points parameter limits dataset size."""
    data_10 = load_pantheon_plus_subset(max_points=10)
    data_20 = load_pantheon_plus_subset(max_points=20)
    
    assert len(data_10) == 10, "Should return exactly 10 points"
    assert len(data_20) == 20, "Should return exactly 20 points"
    
    # First 10 points should be identical
    for p1, p2 in zip(data_10, data_20[:10]):
        assert p1.z == p2.z
        assert p1.mu == p2.mu

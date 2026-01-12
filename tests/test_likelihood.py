"""
Tests for likelihood functions.
"""

import numpy as np
import pytest
from src.ccw.data_loader import load_pantheon_plus_subset, DistanceModulusPoint
from src.ccw.likelihood import distance_modulus_likelihood, fit_mechanism_parameters
from src.ccw.mechanisms import HolographicDarkEnergy


def test_distance_modulus_likelihood_returns_finite():
    """Verify likelihood computation returns finite values."""
    data = load_pantheon_plus_subset(max_points=10)
    
    # Simple constant density evaluator (ΛCDM-like)
    def evaluator(z):
        return 5.3e-10  # Observed dark energy density
    
    result = distance_modulus_likelihood(data, evaluator, h0_fiducial=70.0)
    
    assert np.isfinite(result.log_likelihood), "Log-likelihood should be finite"
    assert np.isfinite(result.chi_squared), "Chi-squared should be finite"
    assert result.dof > 0, "Degrees of freedom should be positive"


def test_distance_modulus_likelihood_chi_squared_positive():
    """Verify chi-squared is non-negative."""
    data = load_pantheon_plus_subset(max_points=10)
    
    def evaluator(z):
        return 5.3e-10
    
    result = distance_modulus_likelihood(data, evaluator, h0_fiducial=70.0)
    
    assert result.chi_squared >= 0, "Chi-squared should be non-negative"


def test_distance_modulus_likelihood_dof_equals_ndata():
    """Verify degrees of freedom equals number of data points (no fitted params)."""
    data = load_pantheon_plus_subset(max_points=15)
    
    def evaluator(z):
        return 5.3e-10
    
    result = distance_modulus_likelihood(data, evaluator, h0_fiducial=70.0)
    
    assert result.dof == 15, "DOF should equal number of data points"


def test_fit_mechanism_parameters_holographic():
    """Verify parameter fitting for holographic mechanism."""
    data = load_pantheon_plus_subset(max_points=20)
    
    # Factory function for holographic mechanism
    def factory(params):
        return HolographicDarkEnergy(
            cutoff_type="hubble",
            c_factor=params["c_factor"],
        )
    
    initial_params = {"c_factor": 1.0}
    param_bounds = {"c_factor": (0.1, 10.0)}
    
    result = fit_mechanism_parameters(
        data=data,
        mechanism_factory=factory,
        initial_params=initial_params,
        param_bounds=param_bounds,
        h0_fiducial=70.0,
    )
    
    assert result.best_fit_params is not None, "Should return best-fit parameters"
    assert "c_factor" in result.best_fit_params, "Should include c_factor"
    assert 0.1 <= result.best_fit_params["c_factor"] <= 10.0, "c_factor should be within bounds"
    assert np.isfinite(result.chi_squared), "Chi-squared should be finite"


def test_fit_mechanism_parameters_improves_likelihood():
    """Verify fitting improves likelihood compared to initial guess."""
    data = load_pantheon_plus_subset(max_points=15)
    
    def factory(params):
        return HolographicDarkEnergy(
            cutoff_type="hubble",
            c_factor=params["c_factor"],
        )
    
    # Initial guess (deliberately poor)
    initial_params = {"c_factor": 5.0}
    
    # Compute initial likelihood
    mech_initial = factory(initial_params)
    def evaluator_initial(z):
        return mech_initial.evaluate(np.array([z]))[0]
    initial_result = distance_modulus_likelihood(data, evaluator_initial, h0_fiducial=70.0)
    
    # Fit parameters
    fit_result = fit_mechanism_parameters(
        data=data,
        mechanism_factory=factory,
        initial_params=initial_params,
        param_bounds={"c_factor": (0.1, 10.0)},
        h0_fiducial=70.0,
    )
    
    # Fitted likelihood should be better (higher log-likelihood, lower chi-squared)
    assert fit_result.log_likelihood >= initial_result.log_likelihood, (
        "Fitting should improve log-likelihood"
    )
    assert fit_result.chi_squared <= initial_result.chi_squared, (
        "Fitting should reduce chi-squared"
    )

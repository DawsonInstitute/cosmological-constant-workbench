"""Tests for LQG-constrained mechanisms."""

import pytest

from ccw.integrations.lqg_constrained import (
    holographic_constrained_by_lqg,
    sequestering_constrained_by_lqg,
)
from ccw.integrations.lqg_predictor import lqg_predictor_available
from ccw.mechanisms import CosmologyBackground


@pytest.mark.skipif(not lqg_predictor_available(), reason="LQG predictor not available")
def test_holographic_lqg_constrained_returns_result():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_lambda=0.6889, omega_m=0.3111)
    result = holographic_constrained_by_lqg(bg)
    assert result.mechanism_name == "holographic_lqg_constrained"
    assert result.lqg_target_rho_j_m3 > 0
    assert "c_factor" in result.best_fit_params
    assert result.achieved_rho_j_m3 > 0


@pytest.mark.skipif(not lqg_predictor_available(), reason="LQG predictor not available")
def test_holographic_lqg_constrained_c_factor_reasonable():
    bg = CosmologyBackground()
    result = holographic_constrained_by_lqg(bg)
    c_factor = result.best_fit_params["c_factor"]
    # c_factor should be within a reasonable range (not极端)
    assert 0.1 <= c_factor <= 10.0


@pytest.mark.skipif(not lqg_predictor_available(), reason="LQG predictor not available")
def test_sequestering_lqg_constrained_returns_result():
    bg = CosmologyBackground()
    result = sequestering_constrained_by_lqg(bg)
    assert result.mechanism_name == "sequestering_lqg_constrained"
    assert result.lqg_target_rho_j_m3 > 0
    assert "f_cancel" in result.best_fit_params
    assert result.achieved_rho_j_m3 > 0


@pytest.mark.skipif(not lqg_predictor_available(), reason="LQG predictor not available")
def test_sequestering_lqg_constrained_extreme_tuning():
    bg = CosmologyBackground()
    result = sequestering_constrained_by_lqg(bg, rho_vac_j_m3=1e113)
    # Sequestering with Planck-scale vacuum requires extreme fine-tuning
    # (should be ~120 orders of magnitude)
    assert result.residual_tuning > 100.0
    assert not result.success  # Should not claim success with such tuning


def test_holographic_lqg_constrained_graceful_failure_without_predictor():
    """Test graceful degradation when LQG predictor is unavailable."""
    bg = CosmologyBackground()
    result = holographic_constrained_by_lqg(bg)
    # Should return a result even if predictor is missing
    assert result.mechanism_name == "holographic_lqg_constrained"
    if not lqg_predictor_available():
        assert not result.success
        assert "not available" in result.notes


def test_sequestering_lqg_constrained_graceful_failure_without_predictor():
    """Test graceful degradation when LQG predictor is unavailable."""
    bg = CosmologyBackground()
    result = sequestering_constrained_by_lqg(bg)
    assert result.mechanism_name == "sequestering_lqg_constrained"
    if not lqg_predictor_available():
        assert not result.success
        assert "not available" in result.notes

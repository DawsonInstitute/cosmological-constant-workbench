"""Integration tests for optional LQG predictor adapter.

These tests skip gracefully if lqg-cosmological-constant-predictor is unavailable.
"""

from __future__ import annotations

import pytest

from ccw.integrations import lqg_predictor_available


@pytest.mark.skipif(not lqg_predictor_available(), reason="lqg-cosmological-constant-predictor not available")
def test_lqg_predictor_comparison_runs() -> None:
    """Validate that the LQG predictor comparison adapter runs without errors."""

    from ccw.integrations.lqg_predictor import compare_baseline

    result = compare_baseline(h0_km_s_mpc=67.4, omega_lambda=0.6889)
    assert result.lqg_lambda_m_minus2 > 0
    assert result.lqg_rho_j_m3 > 0
    assert result.ccw_lambda_m_minus2 > 0
    assert result.ccw_rho_j_m3 > 0
    assert result.ratio_lambda > 0
    assert result.ratio_rho > 0


@pytest.mark.skipif(not lqg_predictor_available(), reason="lqg-cosmological-constant-predictor not available")
def test_lqg_predictor_comparison_returns_finite_ratios() -> None:
    """Validate that ratios are finite and physically reasonable."""

    import math

    from ccw.integrations.lqg_predictor import compare_baseline

    result = compare_baseline()
    assert math.isfinite(result.ratio_lambda)
    assert math.isfinite(result.ratio_rho)
    # Expect LQG first-principles to differ substantially from observed; ratio can be >> 1 or << 1
    assert result.ratio_lambda > 1e-200 and result.ratio_lambda < 1e200


def test_lqg_predictor_unavailable_raises() -> None:
    """Validate that compare_baseline raises ImportError if LQG predictor is unavailable."""

    if lqg_predictor_available():
        pytest.skip("LQG predictor is available; cannot test unavailable path")

    from ccw.integrations.lqg_predictor import compare_baseline

    with pytest.raises(ImportError, match="lqg-cosmological-constant-predictor not available"):
        compare_baseline()

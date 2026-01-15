"""Tests for predictor sweep logic (runner-injected, no external dependency)."""

import math

from ccw.integrations.lqg_predictor import LQGPredictorFirstPrinciplesConfig, LQGPredictorFirstPrinciplesResult
from ccw.integrations.lqg_predictor_sweep import PredictorSweepPoint, scan_points
from ccw.mechanisms import CosmologyBackground


def _dummy_runner(cfg: LQGPredictorFirstPrinciplesConfig) -> LQGPredictorFirstPrinciplesResult:
    # Deterministic toy mapping: rho scales with a param override.
    # The scan logic should pick the point closest to rho_obs.
    mu = float((cfg.params_overrides or {}).get("mu_polymer", 1.0))
    rho = 1e-9 * mu  # J/m^3
    return LQGPredictorFirstPrinciplesResult(
        lambda_effective_m_minus2=1.0,
        vacuum_energy_density_j_m3=rho,
        mu_scale=None,
        enhancement_factor=None,
        scale_correction=None,
        notes="dummy",
    )


def test_scan_points_sorts_by_abs_log10_ratio():
    bg = CosmologyBackground()
    rho_obs = bg.rho_lambda0_j_m3
    assert rho_obs > 0

    points = [
        PredictorSweepPoint(params_overrides={"mu_polymer": 0.1}, target_scale_m=1e-15),
        PredictorSweepPoint(params_overrides={"mu_polymer": 1.0}, target_scale_m=1e-15),
        PredictorSweepPoint(params_overrides={"mu_polymer": 10.0}, target_scale_m=1e-15),
    ]

    best = scan_points(points, bg=bg, runner=_dummy_runner, top_k=3)
    assert len(best) == 3

    # Ensure ordering is non-decreasing in |log10_ratio|.
    abs_logs = [abs(ev.log10_ratio) for ev in best]
    assert abs_logs == sorted(abs_logs)

    # Sanity: computed ratios are finite.
    for ev in best:
        assert math.isfinite(ev.log10_ratio)
        assert ev.rho_pred_j_m3 > 0

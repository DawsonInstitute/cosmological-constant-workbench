"""Hybrid integration step: scan LQG predictor knobs for an observed-Λ match.

This module is intentionally lightweight and testable:
- core scan logic can be tested with a dummy runner (no external dependency)
- when the external predictor is available, it can be used as the runner.

Primary question:
  Do *bounded, natural-ish* predictor inputs produce ρ_vac ~ ρ_Λ,obs?

If not, the scan gives an empirical no-go style constraint:
  min over scan |log10(ρ_pred/ρ_obs)| is still huge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ..mechanisms import CosmologyBackground
from .lqg_predictor import LQGPredictorFirstPrinciplesConfig, LQGPredictorFirstPrinciplesResult, run_first_principles


@dataclass(frozen=True)
class PredictorSweepPoint:
    params_overrides: dict
    target_scale_m: float


@dataclass(frozen=True)
class PredictorSweepEvaluation:
    point: PredictorSweepPoint
    rho_pred_j_m3: float
    rho_obs_j_m3: float
    ratio_pred_to_obs: float
    log10_ratio: float


def _evaluation_from_rho(
    *,
    point: PredictorSweepPoint,
    rho_pred_j_m3: float,
    rho_obs_j_m3: float,
) -> PredictorSweepEvaluation:
    if rho_pred_j_m3 <= 0 or not math.isfinite(rho_pred_j_m3):
        raise ValueError("rho_pred_j_m3 must be finite and > 0")
    if rho_obs_j_m3 <= 0 or not math.isfinite(rho_obs_j_m3):
        raise ValueError("rho_obs_j_m3 must be finite and > 0")

    ratio = rho_pred_j_m3 / rho_obs_j_m3
    return PredictorSweepEvaluation(
        point=point,
        rho_pred_j_m3=rho_pred_j_m3,
        rho_obs_j_m3=rho_obs_j_m3,
        ratio_pred_to_obs=ratio,
        log10_ratio=math.log10(ratio),
    )


def evaluate_point(
    point: PredictorSweepPoint,
    *,
    bg: CosmologyBackground,
    runner: Callable[[LQGPredictorFirstPrinciplesConfig], LQGPredictorFirstPrinciplesResult] = run_first_principles,
    include_uncertainty: bool = False,
) -> PredictorSweepEvaluation:
    cfg = LQGPredictorFirstPrinciplesConfig(
        target_scale_m=point.target_scale_m,
        include_uncertainty=include_uncertainty,
        params_overrides=point.params_overrides,
    )
    res = runner(cfg)

    return _evaluation_from_rho(
        point=point,
        rho_pred_j_m3=float(res.vacuum_energy_density_j_m3),
        rho_obs_j_m3=float(bg.rho_lambda0_j_m3),
    )


def scan_points(
    points: Iterable[PredictorSweepPoint],
    *,
    bg: CosmologyBackground,
    runner: Callable[[LQGPredictorFirstPrinciplesConfig], LQGPredictorFirstPrinciplesResult] = run_first_principles,
    include_uncertainty: bool = False,
    top_k: int = 10,
) -> list[PredictorSweepEvaluation]:
    """Evaluate scan points and return the closest-to-observed candidates.

    Sorting key is |log10(ρ_pred/ρ_obs)|.
    """

    evals: list[PredictorSweepEvaluation] = []
    for point in points:
        evals.append(evaluate_point(point, bg=bg, runner=runner, include_uncertainty=include_uncertainty))

    evals.sort(key=lambda e: abs(e.log10_ratio))
    return evals[: max(0, int(top_k))]


def make_default_points(
    *,
    target_scale_m: float = 1e-15,
    mu_polymer_values: Optional[list[float]] = None,
    gamma_coefficient_values: Optional[list[float]] = None,
    alpha_scaling_values: Optional[list[float]] = None,
    volume_eigenvalue_cutoff_values: Optional[list[float]] = None,
) -> list[PredictorSweepPoint]:
    """Construct a bounded grid of predictor parameter overrides.

    Defaults are chosen to be:
    - small enough to run quickly
    - broad enough to flag gross mismatches
    """

    mu_polymer_values = mu_polymer_values or [0.05, 0.10, 0.15, 0.20]
    gamma_coefficient_values = gamma_coefficient_values or [0.3, 1.0, 3.0]
    alpha_scaling_values = alpha_scaling_values or [0.05, 0.10, 0.20]
    volume_eigenvalue_cutoff_values = volume_eigenvalue_cutoff_values or [5.0, 10.0, 20.0]

    points: list[PredictorSweepPoint] = []
    for mu in mu_polymer_values:
        for gamma_coeff in gamma_coefficient_values:
            for alpha in alpha_scaling_values:
                for jmax in volume_eigenvalue_cutoff_values:
                    points.append(
                        PredictorSweepPoint(
                            target_scale_m=target_scale_m,
                            params_overrides={
                                "mu_polymer": mu,
                                "gamma_coefficient": gamma_coeff,
                                "alpha_scaling": alpha,
                                "volume_eigenvalue_cutoff": jmax,
                            },
                        )
                    )

    return points

"""LQG-constrained mechanisms: use first-principles LQG Λ as a target.

This module extends the basic LQG adapter to provide mechanisms that are
*constrained* by the LQG prediction rather than just compared against it.

The idea: if LQG predicts Λ_LQG from first principles, can we use this as
a target to constrain free parameters in other mechanisms (e.g., holographic
c_factor, sequestering f_cancel)?

This tests whether combining LQG predictions with phenomenological mechanisms
can reduce or eliminate tuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..mechanisms import CosmologyBackground, HolographicDarkEnergy, MechanismOutput, SequesteringToy
from .lqg_predictor import LQGPredictorComparison, compare_baseline, lqg_predictor_available


@dataclass(frozen=True)
class LQGConstrainedResult:
    """Result from LQG-constrained mechanism optimization.

    Attributes
    ----------
    mechanism_name:
        Name of the mechanism being constrained.
    lqg_target_rho_j_m3:
        LQG-predicted vacuum energy density (target).
    best_fit_params:
        Best-fit parameter values for the mechanism.
    achieved_rho_j_m3:
        Achieved ρ_DE(z=0) with best-fit params.
    residual_tuning:
        Remaining tuning: log10(|achieved - target| / target).
    success:
        True if optimization converged.
    notes:
        Human-readable summary.
    """

    mechanism_name: str
    lqg_target_rho_j_m3: float
    best_fit_params: dict
    achieved_rho_j_m3: float
    residual_tuning: float
    success: bool
    notes: str


def holographic_constrained_by_lqg(
    bg: CosmologyBackground,
    h0_km_s_mpc: float = 67.4,
    omega_lambda: float = 0.6889,
) -> LQGConstrainedResult:
    """Find holographic c_factor that matches LQG prediction.

    Parameters
    ----------
    bg:
        Background cosmology.
    h0_km_s_mpc, omega_lambda:
        Observational inputs for LQG predictor.

    Returns
    -------
    LQGConstrainedResult
        Optimization result.

    Notes
    -----
    Holographic DE has one free parameter c_factor. We solve:

      ρ_HDE(z=0; c_factor) = ρ_LQG

    to find the required c_factor. If c_factor ~ O(1), no tuning.
    If c_factor ≪ 1 or ≫ 1, residual tuning remains.
    """
    if not lqg_predictor_available():
        return LQGConstrainedResult(
            mechanism_name="holographic_lqg_constrained",
            lqg_target_rho_j_m3=0.0,
            best_fit_params={},
            achieved_rho_j_m3=0.0,
            residual_tuning=float("inf"),
            success=False,
            notes="LQG predictor not available",
        )

    # Get LQG target
    comparison = compare_baseline(h0_km_s_mpc, omega_lambda)
    rho_target = comparison.lqg_rho_j_m3

    # Holographic mechanism: ρ_DE ~ c² ρ_crit ~ c² H²
    # We need to find c_factor such that ρ_HDE(z=0) ~ rho_target
    # Approximate: ρ_HDE ~ (3 c² / 8πG) H0²
    # For simplicity, scan c_factor and pick the closest match
    c_factors = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    best_c = 1.0
    best_residual = float("inf")

    for c in c_factors:
        mech = HolographicDarkEnergy(c_factor=c, cutoff_type="hubble")
        out = mech.evaluate(z=0.0, bg=bg)
        rho_achieved = out.result.rho_de_j_m3
        residual = abs(rho_achieved - rho_target)
        if residual < best_residual:
            best_residual = residual
            best_c = c

    # Achieved density
    mech_best = HolographicDarkEnergy(c_factor=best_c, cutoff_type="hubble")
    rho_achieved = mech_best.evaluate(z=0.0, bg=bg).result.rho_de_j_m3

    # Residual tuning metric
    if rho_target > 0:
        residual_tuning = math.log10(abs(rho_achieved - rho_target) / rho_target)
    else:
        residual_tuning = float("inf")

    # Natural-ness check: is c ~ O(1)?
    is_natural = 0.3 <= best_c <= 3.0
    success = is_natural and residual_tuning < 0.0  # <10x residual mismatch

    notes = (
        f"LQG target: {rho_target:.3e} J/m³. "
        f"Best c_factor: {best_c:.2f}. "
        f"Achieved: {rho_achieved:.3e} J/m³. "
    )
    if is_natural:
        notes += "c_factor is O(1) → no additional tuning beyond LQG prediction."
    else:
        notes += f"c_factor = {best_c:.2f} is not O(1) → residual tuning remains."

    return LQGConstrainedResult(
        mechanism_name="holographic_lqg_constrained",
        lqg_target_rho_j_m3=rho_target,
        best_fit_params={"c_factor": best_c},
        achieved_rho_j_m3=rho_achieved,
        residual_tuning=residual_tuning,
        success=success,
        notes=notes,
    )


def sequestering_constrained_by_lqg(
    bg: CosmologyBackground,
    h0_km_s_mpc: float = 67.4,
    omega_lambda: float = 0.6889,
    rho_vac_j_m3: float = 1e113,
) -> LQGConstrainedResult:
    """Find sequestering f_cancel that matches LQG prediction.

    Parameters
    ----------
    bg:
        Background cosmology.
    h0_km_s_mpc, omega_lambda:
        Observational inputs for LQG predictor.
    rho_vac_j_m3:
        Assumed bare vacuum energy (default: Planck-scale cutoff).

    Returns
    -------
    LQGConstrainedResult
        Optimization result.

    Notes
    -----
    Sequestering has one tunable parameter f_cancel (cancellation fraction).
    We solve:

      ρ_residual = ρ_vac * (1 - f_cancel) = ρ_LQG

    so f_cancel = 1 - ρ_LQG / ρ_vac.

    If ρ_vac ~ 10^113 and ρ_LQG ~ 10^-10, then f_cancel ~ 1 - 10^-123,
    which is still extreme fine-tuning (~120 orders of magnitude).
    """
    if not lqg_predictor_available():
        return LQGConstrainedResult(
            mechanism_name="sequestering_lqg_constrained",
            lqg_target_rho_j_m3=0.0,
            best_fit_params={},
            achieved_rho_j_m3=0.0,
            residual_tuning=float("inf"),
            success=False,
            notes="LQG predictor not available",
        )

    # Get LQG target
    comparison = compare_baseline(h0_km_s_mpc, omega_lambda)
    rho_target = comparison.lqg_rho_j_m3

    # Solve for f_cancel
    if rho_vac_j_m3 > 0:
        f_cancel_required = 1.0 - (rho_target / rho_vac_j_m3)
    else:
        f_cancel_required = 0.0

    # Construct mechanism
    mech = SequesteringToy(rho_vac_j_m3=rho_vac_j_m3, rho_pt_j_m3=0.0, f_cancel=f_cancel_required)
    rho_achieved = mech.evaluate(z=0.0, bg=bg).result.rho_de_j_m3

    # Residual tuning: how much cancellation is required?
    # Tuning ~ log10(1 / (1 - f_cancel)) = log10(ρ_vac / ρ_LQG)
    if rho_target > 0 and rho_vac_j_m3 > 0:
        residual_tuning = math.log10(rho_vac_j_m3 / rho_target)
    else:
        residual_tuning = float("inf")

    success = residual_tuning < 2.0  # <100x tuning (extremely optimistic threshold)

    notes = (
        f"LQG target: {rho_target:.3e} J/m³. "
        f"Bare vacuum: {rho_vac_j_m3:.3e} J/m³. "
        f"Required f_cancel: {f_cancel_required:.15f}. "
        f"Tuning: ~10^{residual_tuning:.0f}. "
    )
    if success:
        notes += "Low tuning achieved."
    else:
        notes += "Extreme fine-tuning remains (sequestering alone does not explain Λ)."

    return LQGConstrainedResult(
        mechanism_name="sequestering_lqg_constrained",
        lqg_target_rho_j_m3=rho_target,
        best_fit_params={"f_cancel": f_cancel_required, "rho_vac_j_m3": rho_vac_j_m3},
        achieved_rho_j_m3=rho_achieved,
        residual_tuning=residual_tuning,
        success=success,
        notes=notes,
    )

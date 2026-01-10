"""Optional adapter for comparing against lqg-cosmological-constant-predictor.

This adapter is guarded: all imports are lazy and all functions handle import
failures gracefully, returning availability metadata.

Usage:
    from ccw.integrations.lqg_predictor import lqg_predictor_available, compare_baseline

    if lqg_predictor_available():
        result = compare_baseline(h0, omega_lambda)
        print(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LQGPredictorComparison:
    """Comparison result between CCW baseline and LQG predictor."""

    lqg_lambda_m_minus2: float
    lqg_rho_j_m3: float
    ccw_lambda_m_minus2: float
    ccw_rho_j_m3: float
    ratio_lambda: float
    ratio_rho: float
    notes: str


def lqg_predictor_available() -> bool:
    """Check if lqg-cosmological-constant-predictor is importable."""

    try:
        import cosmological_constant_predictor  # noqa: F401

        return True
    except ImportError:
        return False


def compare_baseline(
    h0_km_s_mpc: float = 67.4, omega_lambda: float = 0.6889
) -> LQGPredictorComparison:
    """Compare CCW baseline (H0, ΩΛ) conversion with LQG predictor first-principles result.

    Args:
        h0_km_s_mpc: Hubble constant in km/s/Mpc
        omega_lambda: Dark-energy density fraction today

    Returns:
        Comparison result.

    Raises:
        ImportError: if lqg-cosmological-constant-predictor is not available.
    """

    if not lqg_predictor_available():
        raise ImportError(
            "lqg-cosmological-constant-predictor not available. "
            "Ensure it is installed or in the Python path."
        )

    # Import CCW baseline
    from ..cosmology import observed_lambda_from_h0_omega

    # Lazy import of LQG predictor
    from cosmological_constant_predictor import CosmologicalConstantPredictor

    # CCW baseline (observational conversion)
    ccw_result = observed_lambda_from_h0_omega(h0_km_s_mpc, omega_lambda)

    # LQG predictor first-principles result
    predictor = CosmologicalConstantPredictor()
    # Note: The LQG predictor returns a complex result object; we extract the key fields.
    # This may require adjustment if the LQG API changes.
    lqg_full_result = predictor.predict_lambda_from_first_principles(
        include_uncertainty=False
    )

    # Extract LQG predictions
    lqg_lambda = lqg_full_result.lambda_effective
    lqg_rho = lqg_full_result.vacuum_energy_density

    # Compute ratios
    ratio_lambda = lqg_lambda / ccw_result.lambda_m_minus2
    ratio_rho = lqg_rho / ccw_result.rho_lambda_j_m3

    notes = (
        "LQG predictor uses first-principles polymer quantization + SU(2) 3nj corrections; "
        "CCW baseline uses standard ΛCDM conversion from (H0, ΩΛ)."
    )

    return LQGPredictorComparison(
        lqg_lambda_m_minus2=lqg_lambda,
        lqg_rho_j_m3=lqg_rho,
        ccw_lambda_m_minus2=ccw_result.lambda_m_minus2,
        ccw_rho_j_m3=ccw_result.rho_lambda_j_m3,
        ratio_lambda=ratio_lambda,
        ratio_rho=ratio_rho,
        notes=notes,
    )

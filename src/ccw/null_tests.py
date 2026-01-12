"""
Null test harness for observational sanity bounds.

Provides automated checks to exclude unphysical parameter regions based on observational constraints.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass
class NullTestBound:
    """Single null test bound."""
    name: str
    description: str
    check_function: Callable[[Any], bool]
    failure_message: str


@dataclass
class NullTestResult:
    """Result from null test evaluation."""
    passed: bool
    failed_bounds: List[str]
    details: Dict[str, Any]


class NullTestHarness:
    """
    Harness for running observational sanity checks on mechanisms.

    Null tests are designed to *exclude* parameter regions that are
    inconsistent with observational bounds, without requiring precise
    data fitting.

    Examples:
    - w(z=0) ≈ -1 today (dark energy equation of state)
    - H(z) monotonically increasing with z
    - ρ_DE(z) > 0 for all z in tested range
    - No singularities or discontinuities
    """

    def __init__(self):
        self.bounds: List[NullTestBound] = []
        self._register_default_bounds()

    def _register_default_bounds(self):
        """Register standard observational sanity bounds."""

        # Bound 1: w(z=0) should be close to -1
        def check_w_today(mechanism_result):
            w_today = mechanism_result.get("w_de_z0")
            if w_today is None:
                return True  # Pass if w not provided
            return -1.5 <= w_today <= -0.5

        self.bounds.append(
            NullTestBound(
                name="w_today_approx_minus_one",
                description="w(z=0) ≈ -1 within reasonable range",
                check_function=check_w_today,
                failure_message="w(z=0) is not approximately -1 (observed: -1.03 ± 0.03)",
            )
        )

        # Bound 2: ρ_DE > 0 for all z
        def check_rho_positive(mechanism_result):
            rho_values = mechanism_result.get("rho_de_values")
            if rho_values is None:
                return True
            return np.all(np.array(rho_values) > 0)

        self.bounds.append(
            NullTestBound(
                name="rho_de_positive",
                description="ρ_DE(z) > 0 for all tested redshifts",
                check_function=check_rho_positive,
                failure_message="ρ_DE is negative at some redshift (unphysical)",
            )
        )

        # Bound 3: H(z) should be finite and positive
        def check_hz_positive(mechanism_result):
            hz_values = mechanism_result.get("hz_values")
            if hz_values is None:
                return True
            hz_arr = np.array(hz_values)
            return np.all(np.isfinite(hz_arr)) and np.all(hz_arr > 0)

        self.bounds.append(
            NullTestBound(
                name="hz_positive_finite",
                description="H(z) > 0 and finite for all tested redshifts",
                check_function=check_hz_positive,
                failure_message="H(z) is non-positive or infinite (unphysical)",
            )
        )

        # Bound 4: H(z) should be monotonically increasing with z (for ΛCDM-like)
        def check_hz_monotonic(mechanism_result):
            hz_values = mechanism_result.get("hz_values")
            z_values = mechanism_result.get("z_values")
            if hz_values is None or z_values is None:
                return True
            
            # Check that H(z) increases with z (for standard cosmology)
            for i in range(len(hz_values) - 1):
                if hz_values[i] >= hz_values[i + 1]:
                    return False
            return True

        self.bounds.append(
            NullTestBound(
                name="hz_monotonic_increasing",
                description="H(z) monotonically increases with z",
                check_function=check_hz_monotonic,
                failure_message="H(z) is not monotonically increasing (unusual cosmology)",
            )
        )

        # Bound 5: ρ_DE(z=0) should be within factor of 10 of observed
        def check_rho_today_order_of_magnitude(mechanism_result):
            rho_today = mechanism_result.get("rho_de_z0")
            if rho_today is None:
                return True
            
            # Observed: ρ_Λ ~ 5.3×10^-10 J/m^3
            rho_obs = 5.3e-10
            ratio = rho_today / rho_obs
            
            # Allow factor of 10 (order-of-magnitude sanity check)
            return 0.1 <= ratio <= 10.0

        self.bounds.append(
            NullTestBound(
                name="rho_today_order_of_magnitude",
                description="ρ_DE(z=0) within factor of 10 of observed",
                check_function=check_rho_today_order_of_magnitude,
                failure_message="ρ_DE(z=0) is >10× or <0.1× observed value",
            )
        )

    def add_custom_bound(self, bound: NullTestBound):
        """Add a custom null test bound."""
        self.bounds.append(bound)

    def run_tests(self, mechanism_result: Dict[str, Any]) -> NullTestResult:
        """
        Run all null tests on mechanism result.

        Parameters
        ----------
        mechanism_result : Dict[str, Any]
            Dictionary containing mechanism evaluation results.
            Expected keys:
            - rho_de_z0: ρ_DE(z=0) in J/m^3
            - w_de_z0: w(z=0) (optional)
            - rho_de_values: List of ρ_DE(z) values
            - hz_values: List of H(z) values
            - z_values: List of redshifts

        Returns
        -------
        NullTestResult
            Test results with pass/fail status and details.
        """
        failed_bounds = []
        details = {}

        for bound in self.bounds:
            passed = bound.check_function(mechanism_result)
            details[bound.name] = {
                "passed": passed,
                "description": bound.description,
            }
            
            if not passed:
                failed_bounds.append(bound.name)
                details[bound.name]["failure_message"] = bound.failure_message

        return NullTestResult(
            passed=len(failed_bounds) == 0,
            failed_bounds=failed_bounds,
            details=details,
        )


def evaluate_mechanism_with_null_tests(
    mechanism: Any,
    bg: Any,
    z_values: np.ndarray,
    harness: Optional[NullTestHarness] = None,
) -> Dict[str, Any]:
    """
    Evaluate mechanism and package results for null testing.

    Parameters
    ----------
    mechanism : Any
        Mechanism instance with evaluate() method.
    bg : Any
        CosmologyBackground instance.
    z_values : np.ndarray
        Redshifts to evaluate.
    harness : NullTestHarness, optional
        Harness to use for null tests. If None, creates default harness.

    Returns
    -------
    Dict[str, Any]
        Dictionary with mechanism results and null test outcomes.
    """
    if harness is None:
        harness = NullTestHarness()

    # Evaluate mechanism at all redshifts
    rho_de_values = []
    w_de_values = []
    hz_values = []

    for z in z_values:
        # Check if mechanism has standard evaluate(z, bg) interface or simplified evaluate(z)
        try:
            result = mechanism.evaluate(z, bg).result
            rho_de_values.append(result.rho_de_j_m3)
            w_de_values.append(result.w_de if result.w_de is not None else -1.0)
        except TypeError:
            # Simplified interface (e.g., HolographicDarkEnergy)
            rho_de = mechanism.evaluate(np.array([z]))[0]
            rho_de_values.append(rho_de)
            w_de_values.append(-1.0)  # Default to ΛCDM-like
        
        # Compute H(z) (approximate with ΛCDM for now)
        from .frw import h_z_lcdm_s_inv
        hz_values.append(h_z_lcdm_s_inv(z, bg))

    # Package results
    mechanism_result = {
        "z_values": z_values.tolist(),
        "rho_de_values": rho_de_values,
        "rho_de_z0": rho_de_values[0] if len(rho_de_values) > 0 else None,
        "w_de_values": w_de_values,
        "w_de_z0": w_de_values[0] if len(w_de_values) > 0 else None,
        "hz_values": hz_values,
    }

    # Run null tests
    null_test_result = harness.run_tests(mechanism_result)

    return {
        "mechanism_results": mechanism_result,
        "null_tests": {
            "passed": null_test_result.passed,
            "failed_bounds": null_test_result.failed_bounds,
            "details": null_test_result.details,
        },
    }

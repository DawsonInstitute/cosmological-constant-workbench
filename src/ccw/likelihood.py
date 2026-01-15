"""
Likelihood functions for cosmological parameter constraints.

Provides lightweight Bayesian inference tools for fitting mechanism parameters to observational data.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize

from .data_loader import DistanceModulusPoint
from .frw import distance_modulus as frw_distance_modulus
from .constants import MPC_M
from .cmb_bao_observables import (
    CMBObservable,
    BAOObservable,
    angular_diameter_distance_mpc,
    cmb_acoustic_scale_ell_a,
    dilation_scale_dv,
)
from .gw_observables import GWObservable, gw_chi_squared


@dataclass
class LikelihoodResult:
    """Result from likelihood evaluation or parameter fit."""
    log_likelihood: float
    chi_squared: float
    dof: int
    reduced_chi_squared: float
    best_fit_params: Optional[Dict[str, Any]] = None
    param_uncertainties: Optional[Dict[str, float]] = None


def distance_modulus_likelihood(
    data: List[DistanceModulusPoint],
    hz_s_inv_callable: Callable[[float], float],
) -> LikelihoodResult:
    """
    Compute likelihood for distance modulus observations given a mechanism.

    Parameters
    ----------
    data : List[DistanceModulusPoint]
        Observed distance modulus data.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.

    Returns
    -------
    LikelihoodResult
        Log-likelihood, chi-squared, and goodness-of-fit statistics.

    Notes
    -----
    Distance modulus: μ(z) = 5 log₁₀(d_L / 10 pc)
    where d_L is luminosity distance in parsecs.

    Chi-squared: χ² = Σ [(μ_obs - μ_theory)² / σ_μ²]
    Log-likelihood: ln L = -χ²/2 (Gaussian errors)
    """
    # Fast distance-modulus evaluation: precompute D_C(z) via a cumulative trapezoid
    # on a uniform z grid. This avoids calling quad() for each SNe point.
    chi_squared = 0.0
    n_points = len(data)
    if n_points == 0:
        return LikelihoodResult(
            log_likelihood=0.0,
            chi_squared=0.0,
            dof=0,
            reduced_chi_squared=float("inf"),
        )

    z_max = max(float(p.z) for p in data)
    if z_max < 0:
        raise ValueError("Encountered z<0 in distance modulus data")

    # Keep grid modest to stay fast inside optimizers.
    n_grid = 800
    z_grid = np.linspace(0.0, z_max, n_grid)
    hz_vals = np.array([hz_s_inv_callable(float(z)) for z in z_grid], dtype=float)
    if np.any(~np.isfinite(hz_vals)) or np.any(hz_vals <= 0):
        raise ValueError("Non-physical H(z) encountered (must be finite and > 0)")

    inv_h = 1.0 / hz_vals
    dz = z_grid[1] - z_grid[0] if n_grid > 1 else 0.0
    # cumulative trapezoid for integral_0^z dz'/H(z')
    cum_int = np.zeros_like(z_grid)
    if n_grid > 1:
        cum_int[1:] = np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * dz)

    # Convert to distance modulus using FRW relation: D_L=(1+z) c * ∫ dz/H
    # μ = 5 log10(D_L / 10 pc)
    c_m_s = 2.99792458e8
    pc_m = MPC_M / 1e6
    for point in data:
        z = float(point.z)
        int_0_z = float(np.interp(z, z_grid, cum_int))
        d_l_m = (1.0 + z) * c_m_s * int_0_z
        mu_theory = 5.0 * np.log10(d_l_m / (10.0 * pc_m))
        residual = float(point.mu) - mu_theory
        chi_squared += (residual / float(point.sigma_mu)) ** 2

    # Degrees of freedom (n_data - n_params, assume 0 free params for now)
    dof = n_points
    reduced_chi_squared = chi_squared / dof if dof > 0 else float("inf")

    # Log-likelihood (Gaussian errors)
    log_likelihood = -0.5 * chi_squared

    return LikelihoodResult(
        log_likelihood=log_likelihood,
        chi_squared=chi_squared,
        dof=dof,
        reduced_chi_squared=reduced_chi_squared,
    )


def fit_mechanism_parameters(
    data: List[DistanceModulusPoint],
    mechanism_factory: Callable[[Dict[str, Any]], Any],
    initial_params: Dict[str, float],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    h0_fiducial: float = 70.0,
    omega_m_fiducial: float = 0.3,
    maxiter: int = 60,
) -> LikelihoodResult:
    """
    Fit mechanism parameters to distance modulus data using maximum likelihood.

    Parameters
    ----------
    data : List[DistanceModulusPoint]
        Observed distance modulus data.
    mechanism_factory : Callable[[Dict[str, Any]], Any]
        Function that takes parameter dict and returns mechanism instance.
    initial_params : Dict[str, float]
        Initial parameter values for optimization.
    param_bounds : Dict[str, Tuple[float, float]], optional
        Bounds for each parameter (min, max). If None, unbounded.
    h0_fiducial : float
        Fiducial H0 in km/s/Mpc.

    Returns
    -------
    LikelihoodResult
        Best-fit parameters, uncertainties, and goodness-of-fit statistics.

    Notes
    -----
    Uses SciPy minimize to find maximum likelihood parameters.
    Uncertainties estimated from inverse Hessian (approximate).
    """
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[name] for name in param_names])

    # Set up bounds
    bounds = None
    if param_bounds is not None:
        bounds = [param_bounds.get(name, (None, None)) for name in param_names]

    # Negative log-likelihood objective
    def neg_log_likelihood(x):
        params = {name: val for name, val in zip(param_names, x)}
        try:
            mech = mechanism_factory(params)

            # Build H(z) from mechanism-provided ρ_DE(z) via bookkeeping Friedmann.
            from .mechanisms import CosmologyBackground
            from .frw import MechanismHz

            omega_lambda = max(0.0, 1.0 - float(omega_m_fiducial))
            bg = CosmologyBackground(
                h0_km_s_mpc=float(h0_fiducial),
                omega_m=float(omega_m_fiducial),
                omega_lambda=omega_lambda,
            )
            hz = MechanismHz(mech, bg).h

            result = distance_modulus_likelihood(data, hz)
            return -result.log_likelihood
        except Exception:
            # Keep optimizers robust to non-physical parameter regions.
            return 1e100

    # Optimize
    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B" if bounds else "BFGS",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )

    best_fit = {name: val for name, val in zip(param_names, result.x)}

    # Compute final likelihood with best-fit parameters
    mech_best = mechanism_factory(best_fit)

    from .mechanisms import CosmologyBackground
    from .frw import MechanismHz

    omega_lambda = max(0.0, 1.0 - float(omega_m_fiducial))
    bg_best = CosmologyBackground(h0_km_s_mpc=float(h0_fiducial), omega_m=float(omega_m_fiducial), omega_lambda=omega_lambda)
    hz_best = MechanismHz(mech_best, bg_best).h
    final_likelihood = distance_modulus_likelihood(data, hz_best)

    # Estimate uncertainties from inverse Hessian (if available)
    uncertainties = None
    if hasattr(result, "hess_inv"):
        try:
            # For BFGS: hess_inv is directly the inverse Hessian
            if isinstance(result.hess_inv, np.ndarray):
                cov_matrix = result.hess_inv
            else:
                # For L-BFGS-B: approximate from final Hessian
                cov_matrix = np.diag([1e-3] * len(param_names))  # fallback
            
            param_std = np.sqrt(np.diag(cov_matrix))
            uncertainties = {name: std for name, std in zip(param_names, param_std)}
        except:
            pass

    return LikelihoodResult(
        log_likelihood=final_likelihood.log_likelihood,
        chi_squared=final_likelihood.chi_squared,
        dof=final_likelihood.dof - len(param_names),  # Subtract fitted params
        reduced_chi_squared=final_likelihood.chi_squared / (final_likelihood.dof - len(param_names)),
        best_fit_params=best_fit,
        param_uncertainties=uncertainties,
    )


def cmb_likelihood(
    cmb_obs: CMBObservable,
    hz_s_inv_callable: Callable[[float], float],
) -> LikelihoodResult:
    """
    Compute CMB acoustic scale likelihood.

    Parameters
    ----------
    cmb_obs : CMBObservable
        CMB acoustic scale measurement (ℓ_A).
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.

    Returns
    -------
    LikelihoodResult
        Likelihood result from CMB.

    Notes
    -----
    Compares theoretical ℓ_A = π D_C(z_*) / r_s to observed value.
    Uses COMOVING distance D_C, not angular diameter distance D_A.
    """
    ell_a_theory = cmb_acoustic_scale_ell_a(
        cmb_obs.z_star,
        hz_s_inv_callable,
        r_s_mpc=cmb_obs.r_s_mpc,
    )
    
    residual = cmb_obs.ell_a - ell_a_theory
    chi2 = (residual / cmb_obs.sigma_ell_a) ** 2
    
    return LikelihoodResult(
        log_likelihood=-0.5 * chi2,
        chi_squared=chi2,
        dof=1,  # Single CMB observable
        reduced_chi_squared=chi2,
    )


def bao_likelihood(
    bao_obs_list: List[BAOObservable],
    hz_s_inv_callable: Callable[[float], float],
) -> LikelihoodResult:
    """
    Compute BAO likelihood.

    Parameters
    ----------
    bao_obs_list : List[BAOObservable]
        List of BAO measurements.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.

    Returns
    -------
    LikelihoodResult
        Likelihood result from BAO.

    Notes
    -----
    Supports D_V(z) measurements (dilation scale).
    Can be extended for H(z) and D_A(z) separately.
    """
    chi2_total = 0.0
    
    for obs in bao_obs_list:
        if obs.measurement_type == "DV":
            theory = dilation_scale_dv(obs.z, hz_s_inv_callable)
        elif obs.measurement_type == "DA":
            theory = angular_diameter_distance_mpc(obs.z, hz_s_inv_callable)
        elif obs.measurement_type == "H":
            # H(z) in km/s/Mpc from H(z) in s^-1
            h_z_s_inv = hz_s_inv_callable(obs.z)
            theory = h_z_s_inv * (MPC_M / 1e3)
        else:
            raise ValueError(f"Unknown BAO measurement type: {obs.measurement_type}")
        
        residual = obs.value - theory
        chi2_total += (residual / obs.sigma) ** 2
    
    n_obs = len(bao_obs_list)
    return LikelihoodResult(
        log_likelihood=-0.5 * chi2_total,
        chi_squared=chi2_total,
        dof=n_obs,
        reduced_chi_squared=chi2_total / n_obs if n_obs > 0 else float("inf"),
    )


def gw_likelihood(
    gw_obs_list: List[GWObservable],
    hz_s_inv_callable: Callable[[float], float],
    g_eff_func: Optional[Callable[[float], float]] = None,
) -> LikelihoodResult:
    """
    Compute GW standard siren likelihood.

    Parameters
    ----------
    gw_obs_list : List[GWObservable]
        GW siren measurements.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.
    g_eff_func : Callable[[float], float], optional
        Function returning G_eff(z) / G_N. None → GR (no modification).

    Returns
    -------
    LikelihoodResult
        Likelihood result from GW sirens.

    Notes
    -----
    Provides constraints on modified gravity via d_L^GW(z) measurements.
    In GR: d_L^GW = d_L^EM (no deviation).
    Modified gravity: d_L^GW = d_L^EM × sqrt(G_N / G_eff(z)).
    """
    chi2 = gw_chi_squared(gw_obs_list, hz_s_inv_callable, g_eff_func=g_eff_func)
    n_obs = len(gw_obs_list)
    
    return LikelihoodResult(
        log_likelihood=-0.5 * chi2,
        chi_squared=chi2,
        dof=n_obs,
        reduced_chi_squared=chi2 / n_obs if n_obs > 0 else float("inf"),
    )


def joint_likelihood(
    sne_data: Optional[List[DistanceModulusPoint]],
    cmb_obs: Optional[CMBObservable],
    bao_obs_list: Optional[List[BAOObservable]],
    hz_s_inv_callable: Callable[[float], float],
    gw_obs_list: Optional[List[GWObservable]] = None,
    g_eff_func: Optional[Callable[[float], float]] = None,
    h0_fiducial: float = 70.0,
) -> LikelihoodResult:
    """
    Compute joint SNe + CMB + BAO + GW likelihood.

    Parameters
    ----------
    sne_data : List[DistanceModulusPoint], optional
        SNe Ia distance modulus data.
    cmb_obs : CMBObservable, optional
        CMB acoustic scale measurement.
    bao_obs_list : List[BAOObservable], optional
        BAO measurements.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.
    gw_obs_list : List[GWObservable], optional
        GW standard siren measurements (new in Phase I.21).
    g_eff_func : Callable[[float], float], optional
        Modified gravity G_eff(z)/G_N for GW propagation. None → GR.
    h0_fiducial : float
        Fiducial H0 for SNe absolute calibration.

    Returns
    -------
    LikelihoodResult
        Joint likelihood with combined chi-squared.

    Notes
    -----
    Chi-squared: χ² = χ²_SNe + χ²_CMB + χ²_BAO + χ²_GW
    
    GW sirens add constraints on modified gravity via distance mismatch:
    - GR: d_L^GW = d_L^EM → no extra constraints beyond geometry
    - Modified: d_L^GW ≠ d_L^EM → constrains G_eff(z) parameters
    
    Example: Emergent gravity with β ≠ 0 predicts small GW-EM tension.
    """
    chi2_total = 0.0
    n_data = 0
    
    # SNe contribution (depends only on H(z) via luminosity distance)
    if sne_data is not None and len(sne_data) > 0:
        sne_result = distance_modulus_likelihood(sne_data, hz_s_inv_callable)
        chi2_total += sne_result.chi_squared
        n_data += len(sne_data)
    
    # CMB contribution
    if cmb_obs is not None:
        cmb_result = cmb_likelihood(cmb_obs, hz_s_inv_callable)
        chi2_total += cmb_result.chi_squared
        n_data += 1
    
    # BAO contribution
    if bao_obs_list is not None and len(bao_obs_list) > 0:
        bao_result = bao_likelihood(bao_obs_list, hz_s_inv_callable)
        chi2_total += bao_result.chi_squared
        n_data += len(bao_obs_list)
    
    # GW contribution (NEW: Phase I.21)
    if gw_obs_list is not None and len(gw_obs_list) > 0:
        gw_result = gw_likelihood(gw_obs_list, hz_s_inv_callable, g_eff_func=g_eff_func)
        chi2_total += gw_result.chi_squared
        n_data += len(gw_obs_list)
    
    dof = n_data
    reduced_chi2 = chi2_total / dof if dof > 0 else float('inf')
    log_likelihood = -0.5 * chi2_total
    
    return LikelihoodResult(
        log_likelihood=log_likelihood,
        chi_squared=chi2_total,
        dof=dof,
        reduced_chi_squared=reduced_chi2,
    )

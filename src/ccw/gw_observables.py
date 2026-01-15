"""
Gravitational wave standard siren observables for modified gravity constraints.

Provides tools to compute GW luminosity distances with modified propagation
(e.g., running effective gravitational constant G_eff(z) or modified tensor modes)
and constrain mechanisms via d_L^GW ≠ d_L^EM tension.

Key physics:
- In GR: d_L^GW(z) = d_L^EM(z) (equivalence principle for massless gravitons)
- Modified gravity: d_L^GW(z) = d_L^EM(z) × sqrt(G_N / G_eff(z)) or similar corrections
- Emergent gravity example: G_eff(z) = G_N × [1 + β·H(z)/H_0] from horizon entropy

Observable: GW+EM counterpart events (sirens) measure d_L^GW directly
Constraints: χ²_GW = Σ [(d_L^GW,obs - d_L^GW,model)² / σ²]
"""

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
from scipy.integrate import cumulative_trapezoid

from .constants import MPC_M

# Speed of light in m/s (exact SI definition)
SPEED_OF_LIGHT_M_S = 299792458.0


@dataclass
class GWObservable:
    """
    Gravitational wave standard siren measurement.
    
    Attributes
    ----------
    z : float
        Redshift from EM counterpart (e.g., GRB, kilonova).
    dL_gw_mpc : float
        GW-inferred luminosity distance in Mpc.
    sigma_dL_mpc : float
        Uncertainty on GW distance (typically 10-20% for LIGO/Virgo, improving for ET/LISA).
    event_name : str
        Event identifier (e.g., 'GW170817', 'mock_ET_001').
    """
    z: float
    dL_gw_mpc: float
    sigma_dL_mpc: float
    event_name: str = "unknown"


def gw_luminosity_distance_mpc(
    z: float,
    hz_s_inv_callable: Callable[[float], float],
    g_eff_func: Optional[Callable[[float], float]] = None,
    n_grid: int = 400,
) -> float:
    """
    Compute modified GW luminosity distance d_L^GW(z).
    
    Parameters
    ----------
    z : float
        Redshift.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1 (SI units).
    g_eff_func : Callable[[float], float], optional
        Function returning G_eff(z) / G_N (dimensionless).
        Default None → GR (G_eff = G_N everywhere → factor = 1.0).
    n_grid : int
        Integration grid resolution.
    
    Returns
    -------
    float
        GW luminosity distance in Mpc.
    
    Notes
    -----
    Standard EM luminosity distance:
        d_L^EM(z) = (1+z) × c × ∫₀^z dz'/H(z')
    
    Modified GW distance (toy model):
        d_L^GW(z) = d_L^EM(z) × √[G_N / G_eff(z)]
    
    More general: integrate sqrt(G_N/G_eff(z')) along path for running G.
    Here we use endpoint approximation (sufficient for small deviations).
    
    Physical interpretation:
    - G_eff > G_N: weaker GW coupling → larger inferred distance
    - G_eff < G_N: stronger GW coupling → smaller inferred distance
    
    Example mechanisms:
    - Emergent gravity: G_eff(z) = G_N × [1 + β·H(z)/H_0]
    - Scalar-tensor: G_eff(z) = G_N / [1 + α·φ(z)]
    - Brans-Dicke: G_eff = G_N / ω_BD
    """
    if z < 0:
        raise ValueError(f"Redshift must be non-negative, got z={z}")
    
    # Build integration grid
    z_grid = np.linspace(0.0, float(z), n_grid)
    
    # Compute H(z) on grid
    hz_vals = np.array([hz_s_inv_callable(float(zz)) for zz in z_grid])
    if np.any(~np.isfinite(hz_vals)) or np.any(hz_vals <= 0):
        raise ValueError("Non-physical H(z) encountered in GW distance computation")
    
    # Comoving distance integral: ∫ c dz / H(z)
    integrand = SPEED_OF_LIGHT_M_S / hz_vals
    if n_grid > 1:
        comoving_m = cumulative_trapezoid(integrand, z_grid, initial=0.0)[-1]
    else:
        comoving_m = 0.0
    
    # EM luminosity distance (GR baseline)
    d_l_em_m = (1.0 + z) * comoving_m
    d_l_em_mpc = d_l_em_m / MPC_M
    
    # Modified GW propagation correction
    if g_eff_func is None:
        # GR: no correction
        gw_correction = 1.0
    else:
        # Endpoint approximation: sqrt(G_N / G_eff(z))
        g_ratio = g_eff_func(float(z))
        if g_ratio <= 0:
            raise ValueError(f"G_eff/G_N must be positive, got {g_ratio} at z={z}")
        gw_correction = 1.0 / np.sqrt(g_ratio)
    
    d_l_gw_mpc = d_l_em_mpc * gw_correction
    
    return float(d_l_gw_mpc)


def gw_chi_squared(
    gw_data: List[GWObservable],
    hz_s_inv_callable: Callable[[float], float],
    g_eff_func: Optional[Callable[[float], float]] = None,
) -> float:
    """
    Compute chi-squared for GW standard siren data.
    
    Parameters
    ----------
    gw_data : List[GWObservable]
        GW siren measurements.
    hz_s_inv_callable : Callable[[float], float]
        Function returning H(z) in s^-1.
    g_eff_func : Callable[[float], float], optional
        Function returning G_eff(z) / G_N. None → GR.
    
    Returns
    -------
    float
        Chi-squared: χ² = Σ [(d_L^GW,obs - d_L^GW,model)² / σ²]
    
    Notes
    -----
    Gaussian likelihood: ln L = -χ²/2
    """
    chi2 = 0.0
    
    for obs in gw_data:
        d_l_model = gw_luminosity_distance_mpc(
            obs.z,
            hz_s_inv_callable,
            g_eff_func=g_eff_func,
        )
        
        residual = obs.dL_gw_mpc - d_l_model
        chi2 += (residual / obs.sigma_dL_mpc) ** 2
    
    return chi2


def generate_mock_gw_data(
    z_events: List[float],
    hz_fiducial: Callable[[float], float],
    g_eff_fiducial: Optional[Callable[[float], float]] = None,
    fractional_error: float = 0.15,
    seed: Optional[int] = None,
) -> List[GWObservable]:
    """
    Generate mock GW siren data for testing.
    
    Parameters
    ----------
    z_events : List[float]
        Redshifts of mock siren events.
    hz_fiducial : Callable[[float], float]
        Fiducial H(z) in s^-1 (e.g., ΛCDM or emergent gravity).
    g_eff_fiducial : Callable[[float], float], optional
        Fiducial G_eff(z)/G_N for injected signal. None → GR.
    fractional_error : float
        Fractional uncertainty on d_L (e.g., 0.15 = 15%).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    List[GWObservable]
        Mock GW measurements with Gaussian noise.
    
    Examples
    --------
    >>> # LIGO/Virgo-like: z ~ 0.01-0.5, σ ~ 15%
    >>> mock_lv = generate_mock_gw_data(
    ...     [0.01, 0.1, 0.3, 0.5],
    ...     lambda z: 70e3 * np.sqrt(0.3*(1+z)**3 + 0.7) / MPC_M,
    ...     fractional_error=0.15,
    ... )
    
    >>> # Einstein Telescope: z ~ 0.5-2.0, σ ~ 5-10%
    >>> mock_et = generate_mock_gw_data(
    ...     [0.5, 1.0, 1.5, 2.0],
    ...     lambda z: 70e3 * np.sqrt(0.3*(1+z)**3 + 0.7) / MPC_M,
    ...     fractional_error=0.08,
    ... )
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    mock_data = []
    
    for i, z in enumerate(z_events):
        # True distance from fiducial cosmology
        d_l_true = gw_luminosity_distance_mpc(
            z,
            hz_fiducial,
            g_eff_func=g_eff_fiducial,
        )
        
        # Add Gaussian noise
        sigma = fractional_error * d_l_true
        d_l_obs = d_l_true + rng.normal(0, sigma)
        
        mock_data.append(
            GWObservable(
                z=z,
                dL_gw_mpc=d_l_obs,
                sigma_dL_mpc=sigma,
                event_name=f"mock_{i+1:03d}",
            )
        )
    
    return mock_data


# Example G_eff models for common modified gravity scenarios

def g_eff_emergent_gravity(z: float, beta: float = 0.0, h0_km_s_mpc: float = 70.0, omega_m: float = 0.3) -> float:
    """
    Emergent gravity toy model: G_eff/G_N = 1 + β·H(z)/H_0.
    
    Parameters
    ----------
    z : float
        Redshift.
    beta : float
        Coupling parameter (dimensionless). β=0 → GR.
        Typical range: |β| < 0.1 from current constraints.
    h0_km_s_mpc : float
        Present-day Hubble in km/s/Mpc.
    omega_m : float
        Matter density parameter.
    
    Returns
    -------
    float
        G_eff(z) / G_N (dimensionless).
    
    Notes
    -----
    Motivation: Horizon entropy S ~ A/(4 l_P²) → entropic force F ~ T dS
    gives modified Friedmann with H-dependent effective G.
    
    Physical bounds:
    - G_eff > 0 everywhere (causality)
    - |β| << 1 from Solar System tests (Cassini: |β| < 10^-5)
    - Cosmological: |β| < 0.1 from CMB/BAO (weaker)
    """
    omega_lambda = 1.0 - omega_m
    h_z_over_h0 = np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    
    g_ratio = 1.0 + beta * h_z_over_h0
    
    if g_ratio <= 0:
        raise ValueError(f"G_eff < 0 at z={z} with beta={beta}: unphysical")
    
    return float(g_ratio)


def g_eff_scalar_tensor(z: float, alpha: float = 0.0, phi_z_func: Optional[Callable[[float], float]] = None) -> float:
    """
    Scalar-tensor gravity: G_eff/G_N = 1 / (1 + α·φ(z)).
    
    Parameters
    ----------
    z : float
        Redshift.
    alpha : float
        Coupling strength (dimensionless). α=0 → GR.
    phi_z_func : Callable[[float], float], optional
        Scalar field φ(z) (dimensionless, normalized to φ=1 today).
        Default: φ(z) = 1 + 0.1·z (toy linear evolution).
    
    Returns
    -------
    float
        G_eff(z) / G_N.
    
    Notes
    -----
    Brans-Dicke: α = 1/ω_BD with ω_BD > 40000 from Cassini.
    Solar System: |α| < 10^-5.
    Cosmological: |α| < 0.01 from structure formation.
    """
    if phi_z_func is None:
        # Toy model: linear evolution φ(z) = 1 + 0.1·z
        phi = 1.0 + 0.1 * z
    else:
        phi = phi_z_func(z)
    
    g_ratio = 1.0 / (1.0 + alpha * phi)
    
    if g_ratio <= 0:
        raise ValueError(f"G_eff < 0 at z={z} with alpha={alpha}, phi={phi}")
    
    return float(g_ratio)

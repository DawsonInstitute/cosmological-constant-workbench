"""
Trans-Planckian Censorship Conjecture (TCC) constraints.

The TCC (Bedroya & Vafa 2019) posits that quantum gravity forbids eternal inflation
and constrains the expansion history of the universe:

    H(t) ≲ Λ_TCC ~ 10^-12 GeV

This bound arises from requiring that trans-Planckian modes (wavelengths initially
smaller than the Planck length) never become observable.

For cosmology, TCC implies:
- De Sitter space is inconsistent with quantum gravity (eternal inflation forbidden).
- The Hubble parameter must be bounded throughout cosmic history.
- Dark energy models with large H(z) at late times may violate TCC.

References:
- Bedroya A., Vafa C. (2019), "Trans-Planckian Censorship and the Swampland", arXiv:1909.11063
- Brahma S. (2020), "Trans-Planckian Censorship and Inflation in Grand Unified Theories", arXiv:2001.09536

Physical constants:
- Reduced Planck mass: M_Pl = 1.22×10^19 GeV
- TCC scale: Λ_TCC ~ 10^-12 GeV (order-of-magnitude estimate)
- Observed H_0 ~ 67.4 km/s/Mpc ~ 1.44×10^-42 GeV (well below TCC)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np


# Physical constants
M_PLANCK_GEV = 1.22e19  # Reduced Planck mass (GeV)
H_BAR_GEV_S = 6.582119569e-25  # ℏ in GeV·s
C_M_S = 2.99792458e8  # Speed of light (m/s)

# TCC scale (order-of-magnitude estimate)
LAMBDA_TCC_GEV = 1e-12  # GeV

# Conversion: H in km/s/Mpc to GeV
# H [GeV] = H [km/s/Mpc] × (1000 m/km) / (3.086e22 m/Mpc) / (ℏc)
# ℏc ~ 197 MeV·fm = 1.973e-7 GeV·m
H_BAR_C_GEV_M = 1.97326980e-7  # ℏc in GeV·m


def h0_km_s_mpc_to_gev(h0_km_s_mpc: float) -> float:
    """Convert Hubble parameter from km/s/Mpc to GeV."""
    h0_m_s_per_m = h0_km_s_mpc * 1e3 / 3.085677581e22  # km/s/Mpc to s^-1
    h0_gev = h0_m_s_per_m * H_BAR_GEV_S
    return h0_gev


@dataclass
class TCCConstraintCheck:
    """Result from TCC constraint check."""
    satisfies_tcc: bool
    h_max_gev: float
    lambda_tcc_gev: float
    margin: float  # h_max / lambda_tcc
    details: str


def check_tcc_hubble_bound(
    hz_gev_callable: Callable[[float], float],
    z_values: np.ndarray,
    lambda_tcc_gev: float = LAMBDA_TCC_GEV,
) -> TCCConstraintCheck:
    """
    Check if Hubble parameter H(z) satisfies TCC bound throughout cosmic history.

    Parameters
    ----------
    hz_gev_callable : Callable[[float], float]
        Function that returns H(z) in GeV for given redshift z.
    z_values : np.ndarray
        Redshifts to check (should span 0 to high-z).
    lambda_tcc_gev : float
        TCC scale in GeV (default 10^-12 GeV).

    Returns
    -------
    TCCConstraintCheck
        Whether TCC is satisfied and diagnostic details.

    Notes
    -----
    TCC bound: H(z) ≲ Λ_TCC ~ 10^-12 GeV for all z.
    """
    h_values_gev = np.array([hz_gev_callable(z) for z in z_values])
    h_max_gev = np.max(h_values_gev)
    
    satisfies_tcc = h_max_gev <= lambda_tcc_gev
    margin = h_max_gev / lambda_tcc_gev
    
    if satisfies_tcc:
        details = (
            f"TCC satisfied: H_max = {h_max_gev:.3e} GeV ≤ Λ_TCC = {lambda_tcc_gev:.3e} GeV "
            f"(margin: {margin:.3e})"
        )
    else:
        details = (
            f"TCC violated: H_max = {h_max_gev:.3e} GeV > Λ_TCC = {lambda_tcc_gev:.3e} GeV "
            f"(excess factor: {margin:.3e})"
        )
    
    return TCCConstraintCheck(
        satisfies_tcc=satisfies_tcc,
        h_max_gev=h_max_gev,
        lambda_tcc_gev=lambda_tcc_gev,
        margin=margin,
        details=details,
    )


def check_tcc_lcdm(h0_km_s_mpc: float, omega_m: float, z_max: float = 1100.0) -> TCCConstraintCheck:
    """
    Check if ΛCDM satisfies TCC bound.

    Parameters
    ----------
    h0_km_s_mpc : float
        Hubble constant today (km/s/Mpc).
    omega_m : float
        Matter density parameter today.
    z_max : float
        Maximum redshift to check (default 1100 for CMB).

    Returns
    -------
    TCCConstraintCheck
        TCC satisfaction for ΛCDM.

    Notes
    -----
    ΛCDM: H(z) = H0 sqrt(Ω_m (1+z)^3 + Ω_Λ)
    H is maximum at highest redshift (radiation domination if included).
    """
    h0_gev = h0_km_s_mpc_to_gev(h0_km_s_mpc)
    omega_lambda = 1.0 - omega_m
    
    def hz_gev(z):
        e2 = omega_m * (1 + z) ** 3 + omega_lambda
        return h0_gev * np.sqrt(e2)
    
    z_values = np.logspace(-3, np.log10(z_max), 100)
    
    return check_tcc_hubble_bound(hz_gev, z_values)


def check_tcc_inflation(
    h_inf_gev: float,
    n_efolds: float = 60.0,
    lambda_tcc_gev: float = LAMBDA_TCC_GEV,
) -> TCCConstraintCheck:
    """
    Check if inflationary epoch satisfies TCC.

    Parameters
    ----------
    h_inf_gev : float
        Hubble parameter during inflation (GeV).
    n_efolds : float
        Number of e-folds of inflation (default 60).
    lambda_tcc_gev : float
        TCC scale in GeV.

    Returns
    -------
    TCCConstraintCheck
        TCC satisfaction for inflation.

    Notes
    -----
    TCC tension with inflation:
    - Standard slow-roll inflation: H_inf ~ 10^13-10^14 GeV (GUT scale).
    - TCC bound: H ≲ 10^-12 GeV.
    - Conflict: H_inf >> Λ_TCC by 25 orders of magnitude!
    
    This is a major challenge for TCC if inflation is correct.
    """
    satisfies_tcc = h_inf_gev <= lambda_tcc_gev
    margin = h_inf_gev / lambda_tcc_gev
    
    if satisfies_tcc:
        details = (
            f"Inflation satisfies TCC: H_inf = {h_inf_gev:.3e} GeV ≤ Λ_TCC = {lambda_tcc_gev:.3e} GeV"
        )
    else:
        details = (
            f"Inflation violates TCC: H_inf = {h_inf_gev:.3e} GeV >> Λ_TCC = {lambda_tcc_gev:.3e} GeV "
            f"(excess: {margin:.3e}). This is the TCC-inflation tension."
        )
    
    return TCCConstraintCheck(
        satisfies_tcc=satisfies_tcc,
        h_max_gev=h_inf_gev,
        lambda_tcc_gev=lambda_tcc_gev,
        margin=margin,
        details=details,
    )


def describe_tcc_conjecture() -> str:
    """Return human-readable description of TCC."""
    desc = [
        "Trans-Planckian Censorship Conjecture (TCC):",
        "",
        "Principle:",
        "  - Quantum gravity forbids modes with wavelengths initially < Planck length",
        "    from crossing the Hubble horizon and becoming observable.",
        "  - This censors trans-Planckian physics from low-energy observations.",
        "",
        "Cosmological implications:",
        "  - Eternal inflation is forbidden (would produce infinite trans-Planckian modes).",
        f"  - Hubble parameter bounded: H(t) ≲ Λ_TCC ~ {LAMBDA_TCC_GEV:.0e} GeV for all time.",
        "  - De Sitter space is inconsistent with quantum gravity.",
        "",
        "Observational status:",
        f"  - Current H_0 ~ 67 km/s/Mpc ~ {h0_km_s_mpc_to_gev(67.0):.2e} GeV << Λ_TCC ✓",
        "  - ΛCDM satisfies TCC for late-time evolution ✓",
        "  - Standard inflation (H_inf ~ 10^13 GeV) violates TCC ✗",
        "",
        "Tension with inflation:",
        "  - If TCC is correct, standard high-scale inflation must be modified.",
        "  - Alternatives: low-scale inflation, bouncing cosmology, emergent inflation.",
        "",
        "Reference: Bedroya & Vafa (2019), arXiv:1909.11063",
    ]
    return "\n".join(desc)

"""
Weak Gravity Conjecture (WGC) constraints for scalar field mechanisms.

The WGC (Arkani-Hamed et al. 2007, Vafa et al.) posits that quantum gravity
requires gravity to be the weakest force, which for scalar fields with charge g
under some U(1) gauge field implies:

    m_φ ≤ g M_Pl

where m_φ is the scalar mass and M_Pl the reduced Planck mass.

For quintessence/dark energy scalars coupled to gravity, this translates to
constraints on the potential steepness. Combined with the swampland distance
conjecture (SDC), which states that a scalar traveling a distance Δφ in field
space encounters a tower of states with mass:

    m_tower ∼ M_Pl exp(-Δφ/M_Pl)

the WGC further restricts viable potentials.

Key implications:
- Flat potentials (small m) require extremely weak couplings (tiny g).
- Steep potentials (large |V'|/V) are favored (consistent with swampland gradient bound).
- Eternal de Sitter is disfavored (tower of states becomes light).

References:
- Arkani-Hamed N. et al. (2007), "The String Landscape, Black Holes and Gravity as the Weakest Force", JHEP 0706:060
- Ooguri H., Vafa C. (2007), "On the Geometry of the String Landscape and the Swampland", Nucl.Phys.B766:21-33
- Palti E. (2019), "The Swampland: Introduction and Review", arXiv:1903.06239

Physical constants:
- Reduced Planck mass: M_Pl = 1.22×10^19 GeV = 2.435×10^18 kg
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


# Physical constants
M_PLANCK_GEV = 1.22e19  # Reduced Planck mass (GeV)
M_PLANCK_KG = 2.435e18  # Reduced Planck mass (kg)


@dataclass
class WGCConstraintCheck:
    """Result from WGC constraint check."""
    satisfies_wgc: bool
    mass_gev: float
    coupling_g: float
    m_planck_gev: float
    bound_ratio: float  # m / (g M_Pl)
    details: str


def check_wgc_scalar(
    mass_gev: float,
    coupling_g: float,
    m_planck_gev: float = M_PLANCK_GEV,
) -> WGCConstraintCheck:
    """
    Check if scalar field satisfies Weak Gravity Conjecture.

    Parameters
    ----------
    mass_gev : float
        Scalar mass in GeV.
    coupling_g : float
        Dimensionless coupling strength (e.g., gauge coupling or Yukawa).
    m_planck_gev : float
        Reduced Planck mass in GeV (default 1.22×10^19).

    Returns
    -------
    WGCConstraintCheck
        Whether WGC is satisfied and diagnostic details.

    Notes
    -----
    WGC bound: m_φ ≤ g M_Pl

    For quintessence with V(φ) ∝ exp(-λφ), the effective mass is:
        m² ~ V''(φ) ~ λ² V(φ) / M_Pl²

    And the coupling to matter/gauge fields determines g.
    """
    wgc_bound = coupling_g * m_planck_gev
    satisfies_wgc = mass_gev <= wgc_bound
    bound_ratio = mass_gev / wgc_bound if wgc_bound > 0 else float('inf')

    if satisfies_wgc:
        details = (
            f"WGC satisfied: m = {mass_gev:.3e} GeV ≤ g M_Pl = {wgc_bound:.3e} GeV "
            f"(ratio: {bound_ratio:.3e})"
        )
    else:
        details = (
            f"WGC violated: m = {mass_gev:.3e} GeV > g M_Pl = {wgc_bound:.3e} GeV "
            f"(excess factor: {bound_ratio:.3e})"
        )

    return WGCConstraintCheck(
        satisfies_wgc=satisfies_wgc,
        mass_gev=mass_gev,
        coupling_g=coupling_g,
        m_planck_gev=m_planck_gev,
        bound_ratio=bound_ratio,
        details=details,
    )


def check_wgc_quintessence_exponential(
    lam: float,
    v0_gev4: float,
    coupling_g: float,
    m_planck_gev: float = M_PLANCK_GEV,
) -> WGCConstraintCheck:
    """
    Check WGC for quintessence with exponential potential V = V0 exp(-λφ).

    Parameters
    ----------
    lam : float
        Exponential steepness parameter (dimensionless, assuming φ/M_Pl).
    v0_gev4 : float
        Potential scale in GeV^4.
    coupling_g : float
        Coupling to other fields (dimensionless).
    m_planck_gev : float
        Reduced Planck mass in GeV.

    Returns
    -------
    WGCConstraintCheck
        WGC satisfaction for exponential quintessence.

    Notes
    -----
    For V = V0 exp(-λφ/M_Pl), the effective mass is:
        m² = V'' ~ λ² V / M_Pl²

    At typical field values φ ~ M_Pl, this gives:
        m ~ λ sqrt(V0) / M_Pl

    WGC requires m ≤ g M_Pl, so:
        λ sqrt(V0) / M_Pl ≤ g M_Pl
        λ ≤ g M_Pl² / sqrt(V0)
    """
    # Effective mass at φ ~ M_Pl
    mass_squared_gev2 = lam**2 * v0_gev4 / m_planck_gev**2
    mass_gev = np.sqrt(mass_squared_gev2) if mass_squared_gev2 > 0 else 0.0

    return check_wgc_scalar(mass_gev, coupling_g, m_planck_gev)


def swampland_distance_conjecture_mass(
    delta_phi_mpl: float,
    m_planck_gev: float = M_PLANCK_GEV,
) -> float:
    """
    Compute tower of states mass from swampland distance conjecture.

    Parameters
    ----------
    delta_phi_mpl : float
        Field excursion in units of M_Pl (Δφ / M_Pl).
    m_planck_gev : float
        Reduced Planck mass in GeV.

    Returns
    -------
    float
        Tower mass in GeV: m_tower ~ M_Pl exp(-Δφ/M_Pl).

    Notes
    -----
    SDC: For Δφ > M_Pl (trans-Planckian excursion), a tower of states
    with exponentially decreasing mass appears, signaling breakdown of
    effective field theory.
    """
    m_tower_gev = m_planck_gev * np.exp(-delta_phi_mpl)
    return m_tower_gev


def check_sdc_tower(
    delta_phi_mpl: float,
    cutoff_gev: float,
    m_planck_gev: float = M_PLANCK_GEV,
) -> Dict[str, Any]:
    """
    Check if swampland distance conjecture tower becomes relevant.

    Parameters
    ----------
    delta_phi_mpl : float
        Field excursion in units of M_Pl.
    cutoff_gev : float
        Energy cutoff of effective theory (GeV).
    m_planck_gev : float
        Reduced Planck mass in GeV.

    Returns
    -------
    Dict[str, Any]
        Dictionary with tower mass, validity flag, and details.

    Notes
    -----
    If m_tower < cutoff, the effective theory is no longer valid
    (tower states must be included).
    """
    m_tower = swampland_distance_conjecture_mass(delta_phi_mpl, m_planck_gev)
    eft_valid = m_tower >= cutoff_gev

    return {
        "tower_mass_gev": m_tower,
        "cutoff_gev": cutoff_gev,
        "eft_valid": eft_valid,
        "delta_phi_mpl": delta_phi_mpl,
        "details": (
            f"SDC tower mass: {m_tower:.3e} GeV "
            f"{'≥' if eft_valid else '<'} cutoff {cutoff_gev:.3e} GeV. "
            f"EFT {'valid' if eft_valid else 'breaks down'}."
        ),
    }


def describe_wgc_conjecture() -> str:
    """Return human-readable description of WGC."""
    desc = [
        "Weak Gravity Conjecture (WGC):",
        "",
        "Principle:",
        "  - Gravity must be the weakest force in any consistent quantum gravity theory.",
        "  - For scalars with gauge charge g: m_φ ≤ g M_Pl (gravity weaker than gauge force).",
        "  - Prevents stable black hole remnants (extremal black holes must decay).",
        "",
        "Quintessence implications:",
        "  - Flat potentials (small m) require tiny couplings g << 1.",
        "  - Steep potentials (large |V'|/V) are favored (aligns with swampland).",
        "  - Cosmological constant (m→0) requires g→0 (decoupled scalar, unnatural).",
        "",
        "Swampland Distance Conjecture (SDC):",
        "  - Trans-Planckian field excursions (Δφ > M_Pl) trigger tower of states.",
        f"  - Tower mass: m_tower ~ M_Pl exp(-Δφ/M_Pl) (EFT breaks down).",
        "  - Combined with WGC: restricts viable quintessence parameter space.",
        "",
        "Observational status:",
        "  - ΛCDM (m=0, g→0): WGC technically satisfied but unnatural.",
        "  - Quintessence (steep potentials): WGC+SDC favor λ > O(1) (swampland-consistent).",
        "  - Flat quintessence (λ ~ 0): Ruled out by swampland, WGC suggests also problematic.",
        "",
        "Reference: Arkani-Hamed et al. (2007), Ooguri & Vafa (2007), Palti (2019)",
    ]
    return "\n".join(desc)

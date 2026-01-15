"""UV completion checks for scalar field mechanisms.

Wilsonian effective field theory (EFT) requires a consistent UV completion.
For cosmological scalar fields (quintessence, etc.), this means:

1. **Field excursions**: Δφ should remain sub-Planckian (Δφ < M_Pl) to avoid
   breakdown of EFT. Trans-Planckian excursions signal missing UV physics.

2. **Operator dimension**: Higher-dimension operators like φ^n/M_Pl^(n-4) become
   important at φ ~ M_Pl. Non-renormalizable operators require UV completion.

3. **Cutoff consistency**: The UV cutoff Λ_UV must be self-consistent with the
   theory's mass scales (e.g., Λ_UV ≳ m_φ for reliable perturbation theory).

4. **Strong coupling**: The effective coupling g_eff = φ/Λ_UV should remain
   g_eff ≪ 1 to avoid strong-coupling breakdown of perturbation theory.

References:
- Weinberg S. (1979), "Phenomenological Lagrangians", Physica A 96:327
- Burgess C.P. (2007), "Quantum Gravity in Everyday Life", Living Rev.Rel. 7:5-56
- Rudelius T. (2015), "On the Possibility of Large Axion Moduli Spaces", JCAP 1504:049
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .constants import C_M_S, G_M3_KG_S2, HBAR_J_S


def planck_mass_gev() -> float:
    """Planck mass in GeV.

    M_Pl = sqrt(ħc/G) ~ 1.22×10^19 GeV (reduced Planck mass)
    """
    m_pl_kg = math.sqrt((HBAR_J_S * C_M_S) / G_M3_KG_S2)
    joules_to_gev = 6.242e9
    return float((m_pl_kg * C_M_S**2) * joules_to_gev)


@dataclass(frozen=True)
class UVCompletionCheck:
    """Result of UV completion consistency check.

    Attributes
    ----------
    field_excursion_ok:
        True if Δφ < M_Pl (sub-Planckian).
    delta_phi_over_m_pl:
        Field excursion in Planck units.
    operators_ok:
        True if non-renormalizable operators are under control.
    max_operator_dim:
        Highest dimension operator that is important (e.g., 6 for φ^6/M_Pl^2).
    cutoff_consistent:
        True if UV cutoff is self-consistent with mass scales.
    cutoff_gev:
        UV cutoff scale in GeV.
    strong_coupling_ok:
        True if effective coupling g_eff = φ/Λ_UV ≪ 1.
    g_eff:
        Effective coupling strength.
    uv_complete:
        True if all checks pass.
    details:
        Human-readable summary.
    """

    field_excursion_ok: bool
    delta_phi_over_m_pl: float
    operators_ok: bool
    max_operator_dim: Optional[int]
    cutoff_consistent: bool
    cutoff_gev: float
    strong_coupling_ok: bool
    g_eff: float
    uv_complete: bool
    details: str


def check_uv_completion_scalar_field(
    *,
    delta_phi_gev: float,
    mass_gev: float,
    cutoff_gev: float,
    m_pl_gev: Optional[float] = None,
    g_eff_threshold: float = 0.3,
) -> UVCompletionCheck:
    """Check UV completion for a scalar field mechanism.

    Parameters
    ----------
    delta_phi_gev:
        Total field excursion in GeV (from cosmological evolution z_max → 0).
    mass_gev:
        Scalar field mass in GeV.
    cutoff_gev:
        UV cutoff scale in GeV (where new physics enters).
    m_pl_gev:
        Planck mass in GeV (default: computed from constants).
    g_eff_threshold:
        Threshold for strong coupling (default 0.3).

    Returns
    -------
    UVCompletionCheck
        UV completion consistency check result.

    Notes
    -----
    Field excursion check:
      - Δφ < M_Pl: safe (sub-Planckian)
      - Δφ ~ M_Pl: marginal (EFT validity questionable)
      - Δφ ≫ M_Pl: trans-Planckian (requires UV completion, e.g., string theory)

    Operator check:
      - Dimension-4 operators (renormalizable): always OK
      - Dimension-5,6 operators: OK if suppressed by Λ_UV^(n-4)
      - Dimension-8+ operators: signal missing UV physics if φ ~ Λ_UV

    Cutoff consistency:
      - Λ_UV ≳ m_φ: perturbation theory reliable
      - Λ_UV ≲ m_φ: cutoff too low, theory is non-predictive

    Strong coupling:
      - g_eff = φ/Λ_UV ≪ 1: weak coupling, perturbation theory valid
      - g_eff ~ 1: strong coupling, perturbative expansion breaks down
    """
    if delta_phi_gev < 0:
        raise ValueError("delta_phi_gev must be non-negative")
    if mass_gev <= 0:
        raise ValueError("mass_gev must be positive")
    if cutoff_gev <= 0:
        raise ValueError("cutoff_gev must be positive")
    if g_eff_threshold <= 0 or g_eff_threshold >= 1:
        raise ValueError("g_eff_threshold must be in (0, 1)")

    if m_pl_gev is None:
        m_pl_gev = planck_mass_gev()

    # 1. Field excursion check
    delta_phi_over_m_pl = delta_phi_gev / m_pl_gev
    field_excursion_ok = delta_phi_over_m_pl < 1.0

    # 2. Operator dimension check
    # Estimate which dimension operators are important
    # Dimension-n operator: φ^n / Λ^(n-4)
    # Important when φ ~ Λ
    phi_over_cutoff = delta_phi_gev / cutoff_gev
    if phi_over_cutoff < 0.1:
        # Only renormalizable operators matter
        max_operator_dim = 4
        operators_ok = True
    elif phi_over_cutoff < 1.0:
        # Dimension-5,6 operators become relevant
        max_operator_dim = 6
        operators_ok = True
    else:
        # Higher-dimension operators dominate (UV completion required)
        max_operator_dim = int(4 + math.ceil(math.log10(phi_over_cutoff) / math.log10(2)) * 2)
        operators_ok = False

    # 3. Cutoff consistency check
    cutoff_consistent = cutoff_gev >= mass_gev

    # 4. Strong coupling check
    g_eff = delta_phi_gev / cutoff_gev
    strong_coupling_ok = g_eff < g_eff_threshold

    # Overall verdict
    uv_complete = field_excursion_ok and operators_ok and cutoff_consistent and strong_coupling_ok

    # Build details
    details_lines = []
    details_lines.append(f"Field excursion: Δφ = {delta_phi_gev:.2e} GeV = {delta_phi_over_m_pl:.2f} M_Pl")
    if field_excursion_ok:
        details_lines.append("  ✓ Sub-Planckian (EFT valid)")
    else:
        details_lines.append("  ✗ Trans-Planckian (requires UV completion)")

    details_lines.append(f"Operator dimension: max dim-{max_operator_dim}")
    if operators_ok:
        details_lines.append("  ✓ Higher-dim operators under control")
    else:
        details_lines.append("  ✗ Non-renormalizable operators dominate")

    details_lines.append(f"Cutoff: Λ_UV = {cutoff_gev:.2e} GeV vs m_φ = {mass_gev:.2e} GeV")
    if cutoff_consistent:
        details_lines.append("  ✓ Cutoff above mass scale")
    else:
        details_lines.append("  ✗ Cutoff below mass scale (theory non-predictive)")

    details_lines.append(f"Strong coupling: g_eff = φ/Λ = {g_eff:.2f}")
    if strong_coupling_ok:
        details_lines.append("  ✓ Weak coupling (perturbation theory valid)")
    else:
        details_lines.append("  ✗ Strong coupling (perturbation theory breaks down)")

    if uv_complete:
        details_lines.append("Overall: ✓ UV completion checks pass")
    else:
        details_lines.append("Overall: ✗ UV completion required")

    details = "\n".join(details_lines)

    return UVCompletionCheck(
        field_excursion_ok=field_excursion_ok,
        delta_phi_over_m_pl=delta_phi_over_m_pl,
        operators_ok=operators_ok,
        max_operator_dim=max_operator_dim,
        cutoff_consistent=cutoff_consistent,
        cutoff_gev=cutoff_gev,
        strong_coupling_ok=strong_coupling_ok,
        g_eff=g_eff,
        uv_complete=uv_complete,
        details=details,
    )


def quintessence_uv_check(
    *,
    lambda_coupling: float,
    z_max: float = 5.0,
    cutoff_gev: float = 1e16,
    m_pl_gev: Optional[float] = None,
) -> UVCompletionCheck:
    """Convenience: UV check for exponential quintessence.

    For V ~ exp(-λφ/M_Pl), the field excursion from z=z_max to z=0 is
    roughly Δφ ~ λ M_Pl ln(1+z_max).

    Parameters
    ----------
    lambda_coupling:
        λ parameter in exponential potential.
    z_max:
        Maximum redshift of cosmological evolution.
    cutoff_gev:
        UV cutoff scale (default: GUT scale 10^16 GeV).
    m_pl_gev:
        Planck mass in GeV.

    Returns
    -------
    UVCompletionCheck
        UV completion check result.

    Notes
    -----
    For λ ~ O(1) and z_max ~ 5:
      - Δφ ~ 1.8 M_Pl (trans-Planckian!)
      - Requires string theory UV completion or similar.

    For λ ~ 0.1:
      - Δφ ~ 0.18 M_Pl (sub-Planckian, EFT valid).
    """
    if m_pl_gev is None:
        m_pl_gev = planck_mass_gev()

    # Approximate field excursion
    delta_phi_over_m_pl = lambda_coupling * math.log(1.0 + z_max)
    delta_phi_gev = delta_phi_over_m_pl * m_pl_gev

    # Mass scale ~ H0 ~ 1.5e-33 eV
    h0_gev = 1.5e-33

    return check_uv_completion_scalar_field(
        delta_phi_gev=delta_phi_gev,
        mass_gev=h0_gev,
        cutoff_gev=cutoff_gev,
        m_pl_gev=m_pl_gev,
    )

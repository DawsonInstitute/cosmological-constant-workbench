"""SUSY-breaking vacuum energy toy mechanism.

This mechanism models the residual vacuum energy after softly-broken supersymmetry,
where tree-level cancellations leave a loop-suppressed contribution:

  ρ_vac ≈ (m_SUSY^4 / (16π²)) × log(M_Pl / m_SUSY)

Key features:
- Ties vacuum energy to a particle physics scale (m_SUSY)
- Explicit experimental lower bounds (e.g., m_SUSY > 1 TeV from LHC)
- Logarithmic enhancement from running between m_SUSY and M_Pl
- Transparent "tuning pressure" diagnostic

This is a toy model for constraint exploration, not a complete SUSY framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import C_M_S, G_M3_KG_S2, PI
from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


# Reduced Planck mass in GeV
M_PLANCK_GEV = 1.22e19

# GeV^4 to J/m^3 conversion factor
# 1 GeV = 1.60218e-10 J
# Energy density: (GeV)^4 -> J^4 / (hbar c)^3 in natural units
# Simplified: (GeV^4) × (1.60218e-10 J/GeV)^4 / (ħc)^3
GEV4_TO_J_M3 = (1.60218e-10) ** 4 / ((1.054571817e-34) ** 3 * (2.99792458e8) ** 3)


@dataclass(frozen=True)
class SUSYBreaking:
    """Toy SUSY-breaking vacuum energy mechanism.

    Assumptions:
    1. Minimal SUSY with soft breaking at scale m_SUSY
    2. Tree-level vacuum energy cancels (ρ_bare = 0 by construction)
    3. Loop corrections yield ρ_vac ~ m_SUSY^4 / (16π²) × log(M_Pl / m_SUSY)
    4. Constant dark energy (w = -1) — no dynamical evolution

    Explicit constraints:
    - LHC bounds: m_SUSY ≳ 1 TeV (conservative; varies by model)
    - Perturbativity: m_SUSY ≪ M_Pl (required for loop expansion)
    - Observed Λ: Requires m_SUSY ~ 10^{-3} eV (extreme fine-tuning)
    """

    name: str = "susy_breaking"
    m_susy_gev: float = 1e3  # SUSY breaking scale in GeV (default: 1 TeV)
    loop_factor: float = 1.0 / (16.0 * PI**2)  # 1/(16π²)
    log_enhancement: bool = True  # Include log(M_Pl / m_SUSY) factor

    def describe_assumptions(self) -> str:
        log_info = "with log(M_Pl/m_SUSY) enhancement" if self.log_enhancement else "no log enhancement"
        return (
            f"Toy SUSY-breaking vacuum energy: m_SUSY = {self.m_susy_gev:.2e} GeV, "
            f"ρ_vac = m_SUSY^4 / (16π²) {log_info}. "
            "Assumes tree-level cancellation; constant w=-1. "
            f"**Tuning diagnostic**: m_SUSY required for observed Λ ~ 10^-3 eV (factor ~10^12 below LHC bounds)."
        )

    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        ensure_z_nonnegative(z)

        if self.m_susy_gev <= 0:
            raise ValueError("m_SUSY must be > 0")
        if self.m_susy_gev >= M_PLANCK_GEV:
            raise ValueError("m_SUSY must be ≪ M_Pl for perturbativity")

        # Compute ρ_vac in GeV^4
        rho_gev4 = self.loop_factor * (self.m_susy_gev**4)

        if self.log_enhancement:
            if self.m_susy_gev >= M_PLANCK_GEV:
                raise ValueError("log(M_Pl/m_SUSY) undefined for m_SUSY >= M_Pl")
            log_factor = math.log(M_PLANCK_GEV / self.m_susy_gev)
            rho_gev4 *= log_factor

        # Convert to SI units (J/m³)
        rho_j_m3 = rho_gev4 * GEV4_TO_J_M3

        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_j_m3, w_de=-1.0),
            assumptions=self.describe_assumptions(),
        )


def required_m_susy_for_observed_lambda(
    rho_lambda_obs_j_m3: float,
    *,
    loop_factor: float = 1.0 / (16.0 * PI**2),
    log_enhancement: bool = True,
) -> float:
    """Invert the SUSY formula to find m_SUSY required to match observed ρ_Λ.

    This is a diagnostic tool to quantify fine-tuning pressure.

    Returns:
        m_SUSY in GeV required to produce rho_lambda_obs_j_m3
    """

    if rho_lambda_obs_j_m3 <= 0:
        raise ValueError("ρ_Λ must be > 0")

    # Convert observed density to GeV^4
    rho_obs_gev4 = rho_lambda_obs_j_m3 / GEV4_TO_J_M3

    # Without log: rho = loop_factor * m^4 => m = (rho / loop_factor)^(1/4)
    # With log: rho = loop_factor * m^4 * log(M_Pl / m)
    #   Requires iterative solve; use Newton's method or approximation

    if not log_enhancement:
        return (rho_obs_gev4 / loop_factor) ** 0.25

    # Approximate solution for log case via iteration
    # Start with no-log estimate
    m_guess = (rho_obs_gev4 / loop_factor) ** 0.25

    for _ in range(10):
        if m_guess <= 0 or m_guess >= M_PLANCK_GEV:
            raise ValueError("Failed to converge to physical m_SUSY")
        log_factor = math.log(M_PLANCK_GEV / m_guess)
        m_new = (rho_obs_gev4 / (loop_factor * log_factor)) ** 0.25
        if abs(m_new - m_guess) / m_guess < 1e-6:
            return m_new
        m_guess = m_new

    return m_guess

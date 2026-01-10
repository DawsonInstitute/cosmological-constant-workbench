from __future__ import annotations

from dataclasses import dataclass

from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


@dataclass(frozen=True)
class SequesteringToy:
    """Explicit vacuum energy sequestering cancellation bookkeeping.

    Real sequestering proposals (Kaloper–Padilla, etc.) modify how vacuum energy
    gravitates via global constraints or auxiliary fields.

    This toy models the *effective* result after cancellation using transparent
    parameters:
      1. Assumed bare vacuum density ρ_vac (user-controlled)
      2. Assumed GUT/electroweak phase transition contribution ρ_PT (user-controlled)
      3. Cancellation leaves residual:
         ρ_residual = (ρ_vac + ρ_PT) * f_cancel

    where f_cancel is a small residual fraction (e.g., 1e-120 to match observed ρ_Λ).

    No hidden tuning: the parameters are explicit and the residual is derived.
    """

    name: str = "sequestering_toy"
    rho_vac_j_m3: float = 1e113  # Example naive Planck-scale vacuum ~ (M_pl c^2)^4
    rho_pt_j_m3: float = 1e80   # Example GUT/EW phase transition scale ~ (100 GeV)^4
    f_cancel: float = 1e-120    # Residual fraction post-cancellation

    def describe_assumptions(self) -> str:
        return (
            f"Explicit sequestering cancellation: ρ_vac={self.rho_vac_j_m3:.2e} J/m³, "
            f"ρ_PT={self.rho_pt_j_m3:.2e} J/m³, residual fraction f_cancel={self.f_cancel:.2e}. "
            "Effective residual: ρ_DE = (ρ_vac + ρ_PT) * f_cancel. No hidden tuning; all parameters explicit."
        )

    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        ensure_z_nonnegative(z)
        rho_residual = (self.rho_vac_j_m3 + self.rho_pt_j_m3) * self.f_cancel
        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_residual, w_de=-1.0),
            assumptions=self.describe_assumptions(),
        )

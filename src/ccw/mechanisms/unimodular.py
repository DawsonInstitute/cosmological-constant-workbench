from __future__ import annotations

from dataclasses import dataclass

from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


@dataclass(frozen=True)
class UnimodularBookkeeping:
    """Explicit unimodular gravity integration-constant parameterization.

    In unimodular gravity, det(g) is fixed and Λ appears as an integration constant
    in the field equations rather than a parameter in the action.

    This toy models the residual DE density after assuming:
      1. Bare cosmological constant Λ_bare (integration constant)
      2. Quantum vacuum contribution ρ_vac_quantum
      3. Unimodular constraint enforces det(g) = -1, yielding effective:
         ρ_DE = Λ_bare/(8πG) + ρ_vac_quantum * α_grav

    where α_grav is a suppression factor (e.g., from modified gravitational coupling
    of vacuum energy in the unimodular formalism).

    No hidden tuning: Λ_bare, ρ_vac_quantum, and α_grav are explicit parameters.
    """

    name: str = "unimodular_bookkeeping"
    lambda_bare_m_minus2: float = 1e-52  # Integration constant ~ observed Λ
    rho_vac_quantum_j_m3: float = 1e113  # Assumed quantum vacuum ~ Planck scale
    alpha_grav: float = 0.0              # Vacuum energy gravitational coupling suppression

    def describe_assumptions(self) -> str:
        return (
            f"Explicit unimodular bookkeeping: Λ_bare={self.lambda_bare_m_minus2:.2e} m⁻², "
            f"ρ_vac_quantum={self.rho_vac_quantum_j_m3:.2e} J/m³, α_grav={self.alpha_grav:.2e}. "
            "Effective ρ_DE = Λ_bare c²/(8πG) + ρ_vac_quantum * α_grav. No hidden tuning; all parameters explicit."
        )

    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        ensure_z_nonnegative(z)
        from ..constants import C_M_S, G_M3_KG_S2
        import math
        
        rho_from_lambda = self.lambda_bare_m_minus2 * (C_M_S ** 2) / (8.0 * math.pi * G_M3_KG_S2)
        rho_from_quantum = self.rho_vac_quantum_j_m3 * self.alpha_grav
        rho_eff = rho_from_lambda + rho_from_quantum
        
        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_eff, w_de=-1.0),
            assumptions=self.describe_assumptions(),
        )

"""
Emergent gravity mechanism based on Verlinde's entropic force hypothesis.

Derives dark energy from holographic entanglement entropy on the Hubble horizon,
providing a parameter-free alternative to Λ.

References:
- Verlinde (2011): "On the Origin of Gravity and the Laws of Newton"
- Emergent gravity approach: F = T ∇S on cosmological horizons

Math:
  Modified Friedmann: H² = (8πG/3)ρ - α·H·(kʙT/ℏc)·(dS/dt)
  Where:
    - T = Hawking temperature ~ ℏc³/(2πkʙR_H) with R_H = c/H
    - S = holographic entropy ~ A/(4l_P²) with A = 4π(c/H)²
    - α = phenomenological coupling (α=1 for pure emergent, free otherwise)

For this simplified implementation:
  ρ_DE = α · (1 - Ω_m - Ω_r) · ρ_crit,0
  
which is approximately constant (Λ-like). Parameter-free version has α=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


@dataclass(frozen=True)
class EmergentGravity:
    """Emergent gravity dark energy from entropic force.
    
    Parameters
    ----------
    alpha : float
        Entropic coupling strength (α=1 for pure Verlinde theory).
        Free parameter: measures deviation from strict holographic bound.
    name : str
        Mechanism identifier.
    """
    alpha: float = 1.0
    name: str = "emergent_gravity"
    
    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
    
    def describe_assumptions(self) -> str:
        return (
            f"Emergent gravity via Verlinde entropic force with α={self.alpha:.2f}. "
            "Assumes holographic entropy on Hubble horizon generates effective dark energy. "
            "Not a fundamental field; gravity emerges from thermodynamic entropy."
        )
    
    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        """Evaluate dark energy density and equation of state.
        
        Parameters
        ----------
        z : float
            Redshift.
        bg : CosmologyBackground
            Background cosmology.
        
        Returns
        -------
        MechanismOutput
            Contains rho_DE(z) and w(z).
        
        Notes
        -----
        For this simplified model:
        - ρ_DE = α · Ω_Λ,eff · ρ_crit,0 (approximately constant)
        - Ω_Λ,eff = 1 - Ω_m - Ω_r (from entropic constraint)
        - w ≈ -1 (Λ-like)
        """
        ensure_z_nonnegative(z)
        
        # Effective dark energy density parameter from entropic argument
        # For α=1 (parameter-free): use all "missing" energy as emergent DE
        omega_Lambda_eff = self.alpha * (1.0 - bg.omega_m - bg.omega_r)
        
        # Dark energy density (approximately constant, independent of z)
        rho_de = omega_Lambda_eff * bg.rho_lambda0_j_m3 / bg.omega_lambda
        
        # Equation of state (cosmological constant-like)
        w_de = -1.0
        
        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_de, w_de=w_de),
            assumptions=self.describe_assumptions()
        )

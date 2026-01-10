from __future__ import annotations

from dataclasses import dataclass

from .constants import C_M_S, G_M3_KG_S2, MPC_M, PI


def h0_km_s_mpc_to_s_inv(h0_km_s_mpc: float) -> float:
    return (h0_km_s_mpc * 1000.0) / MPC_M


@dataclass(frozen=True)
class ObservedLambdaResult:
    h0_km_s_mpc: float
    omega_lambda: float
    h0_s_inv: float
    rho_crit_mass_kg_m3: float
    rho_crit_energy_j_m3: float
    rho_lambda_j_m3: float
    lambda_m_inv2: float


def observed_lambda_from_h0_omega(h0_km_s_mpc: float, omega_lambda: float) -> ObservedLambdaResult:
    """Compute observed Λ and ρ_Λ from (H0, Ω_Λ) in a flat ΛCDM bookkeeping sense.

    Formulas:
      ρ_c = 3 H0^2 / (8 π G)
      ρ_{c,E} = ρ_c c^2
      ρ_Λ = Ω_Λ ρ_{c,E}
      Ω_Λ = Λ c^2 / (3 H0^2)  -> Λ = 3 Ω_Λ H0^2 / c^2
    """

    if h0_km_s_mpc <= 0:
        raise ValueError("H0 must be positive")
    if not (0.0 <= omega_lambda <= 2.0):
        raise ValueError("Ω_Λ is expected to be in a reasonable range")

    h0_s_inv = h0_km_s_mpc_to_s_inv(h0_km_s_mpc)
    rho_crit_mass = 3.0 * (h0_s_inv**2) / (8.0 * PI * G_M3_KG_S2)
    rho_crit_energy = rho_crit_mass * (C_M_S**2)
    rho_lambda = omega_lambda * rho_crit_energy
    lambda_m_inv2 = 3.0 * omega_lambda * (h0_s_inv**2) / (C_M_S**2)

    return ObservedLambdaResult(
        h0_km_s_mpc=h0_km_s_mpc,
        omega_lambda=omega_lambda,
        h0_s_inv=h0_s_inv,
        rho_crit_mass_kg_m3=rho_crit_mass,
        rho_crit_energy_j_m3=rho_crit_energy,
        rho_lambda_j_m3=rho_lambda,
        lambda_m_inv2=lambda_m_inv2,
    )

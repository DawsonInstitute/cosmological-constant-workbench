from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from scipy import integrate

from .constants import C_M_S, MPC_M
from .cosmology import h0_km_s_mpc_to_s_inv
from .mechanisms import CosmologyBackground, Mechanism


def e2_lcdm(z: float, bg: CosmologyBackground) -> float:
    """Dimensionless E(z)^2 = (H(z)/H0)^2 for ΛCDM-style bookkeeping."""

    if z < 0:
        raise ValueError("z must be >= 0")
    zp1 = 1.0 + z
    return bg.omega_m * zp1**3 + bg.omega_r * zp1**4 + bg.omega_k * zp1**2 + bg.omega_lambda


def h_z_lcdm_s_inv(z: float, bg: CosmologyBackground) -> float:
    h0 = h0_km_s_mpc_to_s_inv(bg.h0_km_s_mpc)
    return h0 * math.sqrt(e2_lcdm(z, bg))


def h_z_from_rho_de_s_inv(z: float, bg: CosmologyBackground, rho_de_j_m3: float) -> float:
    """Compute H(z) given a background + a supplied DE energy density u_DE(z).

    Uses a bookkeeping Friedmann form in terms of today's critical *energy* density.

      H(z)^2 = H0^2 * [Ωm(1+z)^3 + Ωr(1+z)^4 + Ωk(1+z)^2 + u_DE(z)/u_crit0]

    where u_crit0 = ρ_c c^2 = 3 H0^2 c^2 / (8πG).

    This ensures that constant u_DE(z)=ΩΛ u_crit0 reproduces ΛCDM.
    """

    if z < 0:
        raise ValueError("z must be >= 0")
    if rho_de_j_m3 < 0:
        raise ValueError("rho_de_j_m3 must be >= 0")

    h0 = h0_km_s_mpc_to_s_inv(bg.h0_km_s_mpc)

    # u_crit0 = ρ_c c^2. We can compute via ΩΛ = u_Λ0/u_crit0.
    if bg.omega_lambda <= 0:
        raise ValueError("omega_lambda must be > 0 to infer u_crit0 from u_Λ0")
    u_crit0 = bg.rho_lambda0_j_m3 / bg.omega_lambda

    zp1 = 1.0 + z
    e2 = bg.omega_m * zp1**3 + bg.omega_r * zp1**4 + bg.omega_k * zp1**2 + (rho_de_j_m3 / u_crit0)
    if e2 <= 0:
        raise ValueError("Non-physical E(z)^2")
    return h0 * math.sqrt(e2)


def comoving_distance_m(z: float, hz_s_inv: Callable[[float], float]) -> float:
    """Line-of-sight comoving distance for flat FRW: D_C(z)=c∫_0^z dz'/H(z')."""

    if z < 0:
        raise ValueError("z must be >= 0")

    def integrand(zp: float) -> float:
        return 1.0 / hz_s_inv(zp)

    val, err = integrate.quad(integrand, 0.0, z, epsabs=0.0, epsrel=1e-9, limit=200)
    return C_M_S * val


def luminosity_distance_m(z: float, hz_s_inv: Callable[[float], float]) -> float:
    """For flat FRW: D_L(z)=(1+z) D_C(z)."""

    return (1.0 + z) * comoving_distance_m(z, hz_s_inv)


def distance_modulus(z: float, hz_s_inv: Callable[[float], float]) -> float:
    """μ = 5 log10(D_L / 10 pc), with D_L in meters."""

    dl_m = luminosity_distance_m(z, hz_s_inv)
    pc_m = MPC_M / 1e6
    return 5.0 * math.log10(dl_m / (10.0 * pc_m))


@dataclass(frozen=True)
class MechanismHz:
    mechanism: Mechanism
    bg: CosmologyBackground

    def rho_de(self, z: float) -> float:
        return self.mechanism.evaluate(z, self.bg).result.rho_de_j_m3

    def h(self, z: float) -> float:
        return h_z_from_rho_de_s_inv(z, self.bg, self.rho_de(z))

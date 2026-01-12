"""Holographic dark energy (HDE) mechanism.

This file implements a toy HDE-style mechanism behind the common ansatz

  ρ_DE(z) = 3 c^4 / (8π G L(z)^2)

with explicit IR cutoff choices L(z).

Scope and intent:
- This is scaffolding for constraint exploration.
- For the Hubble cutoff we use a simple *self-consistent algebraic closure*
  (because L depends on H).
- For the horizon-integral cutoffs we keep a ΛCDM background in the integral
  for tractability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import integrate

from ..constants import C_M_S, G_M3_KG_S2, PI
from ..cosmology import h0_km_s_mpc_to_s_inv
from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


def _h_z_lcdm_s_inv(z: float, bg: CosmologyBackground) -> float:
    """Local ΛCDM H(z) helper to avoid circular imports (mechanisms <-> frw)."""

    if z < 0:
        raise ValueError("z must be >= 0")
    zp1 = 1.0 + z
    e2 = bg.omega_m * (zp1**3) + bg.omega_r * (zp1**4) + bg.omega_k * (zp1**2) + bg.omega_lambda
    if e2 <= 0.0 or not math.isfinite(e2):
        raise ValueError("Non-physical E(z)^2")
    h0_s_inv = h0_km_s_mpc_to_s_inv(bg.h0_km_s_mpc)
    return h0_s_inv * math.sqrt(e2)


CutoffType = Literal["hubble", "particle_horizon", "event_horizon"]


@dataclass(frozen=True)
class HolographicDarkEnergy:
    """Toy holographic dark energy mechanism with explicit IR cutoffs.

    Parameters
    ----------
    cutoff_type:
        IR cutoff choice.
    c_factor:
        Numerical prefactor applied to L: L → c_factor * L.
        Note: for the Hubble cutoff this maps directly to an algebraic DE fraction.
    z_max_horizon:
        Upper integration limit for the particle-horizon approximation.
    """

    name: str = "holographic_dark_energy"
    cutoff_type: CutoffType = "hubble"
    c_factor: float = 1.0
    z_max_horizon: float = 10.0

    def describe_assumptions(self) -> str:
        return (
            "Toy holographic dark energy with explicit IR cutoffs. "
            f"cutoff_type={self.cutoff_type}, c_factor={self.c_factor}. "
            "Uses ρ_DE = 3 c^4/(8π G L^2) with L chosen as either the Hubble scale or a horizon integral. "
            "Hubble cutoff uses a simple algebraic self-consistency closure; horizon-integral cutoffs use ΛCDM H(z) inside the integral."
        )

    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        ensure_z_nonnegative(z)
        if self.c_factor <= 0:
            raise ValueError("c_factor must be > 0")

        if self.cutoff_type == "hubble":
            # Self-consistent algebraic closure for L = c_factor * c/H.
            # With ρ_DE ∝ 1/L^2, this implies ρ_DE(z) = u_crit(z)/c_factor^2.
            # In the bookkeeping Friedmann form: E^2 = other + E^2/c_factor^2.
            alpha = 1.0 / (self.c_factor * self.c_factor)
            denom = 1.0 - alpha
            if denom <= 0.0:
                raise ValueError("Non-physical: require c_factor > 1 for Hubble cutoff in this toy closure")

            zp1 = 1.0 + z
            other = bg.omega_m * (zp1**3) + bg.omega_r * (zp1**4) + bg.omega_k * (zp1**2)
            ez2 = other / denom
            if ez2 <= 0.0 or not math.isfinite(ez2):
                raise ValueError("Non-physical E(z)^2 in holographic Hubble-cutoff closure")

            h0_s_inv = h0_km_s_mpc_to_s_inv(bg.h0_km_s_mpc)
            u_crit0 = 3.0 * (h0_s_inv**2) * (C_M_S**2) / (8.0 * PI * G_M3_KG_S2)
            rho_de = alpha * u_crit0 * ez2
            return MechanismOutput(
                result=MechanismResult(z=z, rho_de_j_m3=rho_de, w_de=None),
                assumptions=self.describe_assumptions(),
            )

        # Horizon-integral cutoffs (use ΛCDM background H(z) in the integral for tractability).
        a_z = 1.0 / (1.0 + z)

        def inv_h(zp: float) -> float:
            return 1.0 / _h_z_lcdm_s_inv(zp, bg)

        if self.cutoff_type == "particle_horizon":
            z_max = float(self.z_max_horizon)
            if z_max <= z:
                raise ValueError("z_max_horizon must exceed z for particle_horizon cutoff")
            integral, _ = integrate.quad(inv_h, z, z_max, epsabs=0.0, epsrel=1e-9, limit=200)
            l_base_m = a_z * C_M_S * integral
        elif self.cutoff_type == "event_horizon":
            if z <= 0.0:
                # Regularize the z=0 limit.
                l_base_m = (C_M_S / _h_z_lcdm_s_inv(0.0, bg)) * 1e-3
            else:
                integral, _ = integrate.quad(inv_h, 0.0, z, epsabs=0.0, epsrel=1e-9, limit=200)
                l_base_m = a_z * C_M_S * integral
        else:
            raise ValueError(f"Unknown cutoff_type: {self.cutoff_type}")

        l_m = self.c_factor * l_base_m
        if l_m <= 0.0 or not math.isfinite(l_m):
            raise ValueError("Non-physical IR cutoff length")

        rho_de = 3.0 * (C_M_S**4) / (8.0 * PI * G_M3_KG_S2 * (l_m**2))
        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_de, w_de=None),
            assumptions=self.describe_assumptions(),
        )

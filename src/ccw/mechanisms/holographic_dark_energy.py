"""
Holographic dark energy (HDE) mechanism.

Toy implementation of holographic/entropic gravity-inspired dark energy with explicit IR cutoff choices.

Physical assumptions:
1. Holographic principle constrains maximal entropy in a region: S_max ~ (Area/l_P^2) ~ L^2/l_P^2.
2. Identifying the system size L with an IR cutoff yields ρ_DE ~ 3c^2/(8πG L^2).
3. The IR cutoff L is parameterized:
   - "hubble": L(z) = c/H(z) (future horizon)
   - "particle_horizon": L(z) = a(z) * ∫ c dz'/H(z') from z to ∞ (particle horizon)
   - "event_horizon": L(z) = a(z) * ∫ c dz'/H(z') from 0 to z (event horizon approximation)

Reference:
- Li M. (2004), "A Model of Holographic Dark Energy", Phys. Lett. B 603, 1-5.
- Hsu S. D. H. (2004), "Entropy Bounds and Dark Energy", Phys. Lett. B 594, 13-16.

Explicit scale ties:
- L is constructed from FRW observables (H(z), a(z)).
- No hidden tuning; ρ_DE is derived from the chosen IR cutoff.

Limitations:
- Toy model — does not address backreaction or solve for full self-consistent cosmology.
- Particle/event horizon integrals assume ΛCDM background for tractability.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy import integrate

# Physical constants
C_LIGHT_M_S = 2.99792458e8  # speed of light (m/s)
G_SI = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
H_BAR_SI = 1.054571817e-34  # reduced Planck constant (J·s)
M_PLANCK_M = np.sqrt(H_BAR_SI * C_LIGHT_M_S / G_SI)  # Planck length (m)


@dataclass
class HolographicDarkEnergy:
    """
    Holographic dark energy mechanism with explicit IR cutoff parameterization.

    Parameters
    ----------
    cutoff_type : {"hubble", "particle_horizon", "event_horizon"}
        Choice of IR cutoff scale.
        - "hubble": L = c/H(z)
        - "particle_horizon": L = a(z) * ∫ c dz'/H(z') (z to ∞)
        - "event_horizon": L = a(z) * ∫ c dz'/H(z') (0 to z)
    c_factor : float, optional
        Numerical prefactor for IR cutoff (default 1.0).
        L → c_factor * L (allows tuning without changing cutoff type).
    background_h0 : float, optional
        Hubble constant today (km/s/Mpc) for background evolution (default 67.4).
    background_omega_m : float, optional
        Matter density parameter today for background (default 0.3).

    Attributes
    ----------
    cutoff_type : str
        IR cutoff parameterization.
    c_factor : float
        Numerical prefactor.
    background_h0 : float
        Background H0.
    background_omega_m : float
        Background Ω_m.

    Notes
    -----
    The energy density is computed as:
        ρ_DE(z) = 3 c^4 / (8π G L(z)^2)
    where L(z) is the chosen IR cutoff scaled by c_factor.

    For particle/event horizon integrals, we assume a ΛCDM background with
    H(z) = H0 * sqrt(Ω_m * (1+z)^3 + (1 - Ω_m)) for tractability.
    """

    cutoff_type: Literal["hubble", "particle_horizon", "event_horizon"]
    c_factor: float = 1.0
    background_h0: float = 67.4  # km/s/Mpc
    background_omega_m: float = 0.3

    def evaluate(self, z_values: np.ndarray) -> np.ndarray:
        """
        Compute holographic dark energy density at redshifts z.

        Parameters
        ----------
        z_values : np.ndarray
            Redshift values.

        Returns
        -------
        np.ndarray
            Energy density in J/m^3.
        """
        z_arr = np.atleast_1d(z_values)
        rho_de = np.zeros_like(z_arr)

        for i, z in enumerate(z_arr):
            L_m = self._compute_cutoff_m(z)
            # ρ_DE = 3 c^4 / (8π G L^2)
            rho_de[i] = 3 * C_LIGHT_M_S**4 / (8 * np.pi * G_SI * L_m**2)

        return rho_de

    def _compute_cutoff_m(self, z: float) -> float:
        """
        Compute IR cutoff L(z) in meters.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            IR cutoff in meters.
        """
        if self.cutoff_type == "hubble":
            # L = c / H(z)
            H_z_s_inv = self._background_hubble(z)
            L_m = C_LIGHT_M_S / H_z_s_inv

        elif self.cutoff_type == "particle_horizon":
            # L = a(z) * ∫ c dz' / H(z') from z to ∞
            # Approximation: integrate from z to z_max (default 10)
            a_z = 1.0 / (1.0 + z)
            z_max = 10.0

            def integrand(zp):
                return C_LIGHT_M_S / self._background_hubble(zp)

            integral, _ = integrate.quad(integrand, z, z_max)
            L_m = a_z * integral

        elif self.cutoff_type == "event_horizon":
            # L = a(z) * ∫ c dz' / H(z') from 0 to z
            a_z = 1.0 / (1.0 + z)

            if z > 0:
                def integrand(zp):
                    return C_LIGHT_M_S / self._background_hubble(zp)

                integral, _ = integrate.quad(integrand, 0, z)
                L_m = a_z * integral
            else:
                # At z=0, event horizon integral is zero; use small regularization
                L_m = C_LIGHT_M_S / self._background_hubble(0) * 1e-3

        else:
            raise ValueError(f"Unknown cutoff_type: {self.cutoff_type}")

        return self.c_factor * L_m

    def _background_hubble(self, z: float) -> float:
        """
        Background Hubble parameter H(z) in s^-1.

        Assumes ΛCDM: H(z) = H0 * sqrt(Ω_m (1+z)^3 + Ω_Λ).

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            H(z) in s^-1.
        """
        # Convert H0 from km/s/Mpc to s^-1
        H0_s_inv = self.background_h0 * 1e3 / (3.085677581e22)  # 1 Mpc = 3.086e22 m
        omega_lambda = 1.0 - self.background_omega_m

        H_z = H0_s_inv * np.sqrt(
            self.background_omega_m * (1 + z) ** 3 + omega_lambda
        )
        return H_z

    def describe_assumptions(self) -> str:
        """
        Return a human-readable description of model assumptions.

        Returns
        -------
        str
            Description string.
        """
        desc = [
            "Holographic dark energy mechanism:",
            f"  - IR cutoff: {self.cutoff_type}",
            f"  - Numerical factor: c_factor = {self.c_factor}",
            f"  - Background: ΛCDM with H0={self.background_h0} km/s/Mpc, Ω_m={self.background_omega_m}",
            "  - Assumptions:",
            "    * Holographic principle: S_max ~ L^2 / l_P^2",
            "    * Energy density: ρ_DE ~ 3c^2 / (8πG L^2)",
            "    * Background for horizon integrals: ΛCDM (no self-consistency)",
            "  - Limitations:",
            "    * Toy model — no backreaction or full self-consistent evolution",
            "    * Particle/event horizon integrals assume fixed ΛCDM background",
        ]
        return "\n".join(desc)

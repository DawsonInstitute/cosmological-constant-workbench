from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy import integrate

from .base import CosmologyBackground, MechanismOutput, MechanismResult, ensure_z_nonnegative


PotentialKind = Literal["exponential", "inverse_power"]


def _lambda_for_potential(kind: PotentialKind, *, lam: float, alpha: float, phi: float) -> float:
    if kind == "exponential":
        return lam
    if kind == "inverse_power":
        if phi <= 0:
            raise ValueError("phi must remain > 0 for inverse-power potential")
        return alpha / phi
    raise ValueError(f"Unknown potential kind: {kind}")


@dataclass
class ScalarFieldQuintessence:
    """Explicit (toy) scalar-field quintessence evolution in flat FRW.

    Implements the standard autonomous system in terms of dimensionless variables
    (see e.g. Copeland, Liddle & Wands 1998), specialized to:
    - spatially flat FRW (Ω_k = 0)
    - matter (w=0) and optional radiation (w=1/3)

    Variables:
      x = φdot/(√6 H M_pl)
      y = √V/(√3 H M_pl)
      Ω_r = ρ_r/(3 M_pl^2 H^2)

    Then Ω_φ = x^2 + y^2, w_φ = (x^2 - y^2)/(x^2 + y^2).

    Notes / scope:
    - This is a *numerical scaffolding* mechanism for constraint exploration.
    - It is not a claim of realism; potentials are simplistic and parameters are
      not fit to data here.
    """

    name: str = "scalar_field_quintessence"
    potential: PotentialKind = "exponential"

    # Exponential potential: V ∝ exp(-λ φ/M_pl)
    lam: float = 0.0

    # Inverse power potential: V ∝ (φ/M_pl)^(-α). Requires phi0 > 0.
    alpha: float = 1.0
    phi0: float = 1.0

    # Integration controls
    z_max: float = 5.0
    n_eval: int = 800
    rtol: float = 1e-8
    atol: float = 1e-10

    # Initial conditions at z=0 (today)
    # Choose small x0 by default (w ~ -1 today). y0 is set from ΩΛ.
    x0: float = 0.0

    _cache_bg: Optional[CosmologyBackground] = field(default=None, init=False, repr=False)
    _cache_n: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cache_state: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def describe_assumptions(self) -> str:
        pot = (
            f"exponential V∝exp(-λ φ/M_pl) with λ={self.lam}" if self.potential == "exponential" else f"inverse-power V∝(φ/M_pl)^(-α) with α={self.alpha}"
        )
        return (
            "Toy canonical scalar-field quintessence evolved via an autonomous ODE system in flat FRW. "
            f"Potential: {pot}. "
            "Uses (x,y,Ω_r) variables; sets y(z=0)=sqrt(ΩΛ) and x(z=0)=x0, and evolves backward to higher z. "
            "Not a data fit; intended for numerical constraint scaffolding."
        )

    def _ensure_cache(self, bg: CosmologyBackground) -> None:
        if self._cache_bg == bg and self._cache_n is not None and self._cache_state is not None:
            return

        if bg.omega_k != 0.0:
            raise ValueError("ScalarFieldQuintessence currently assumes flat FRW (omega_k=0)")
        if bg.omega_lambda <= 0:
            raise ValueError("omega_lambda must be > 0")
        if self.z_max <= 0:
            raise ValueError("z_max must be > 0")
        if self.n_eval < 50:
            raise ValueError("n_eval too small")

        n0 = 0.0
        nmin = -math.log1p(self.z_max)

        y0 = math.sqrt(max(bg.omega_lambda, 0.0))
        if not math.isfinite(y0) or y0 <= 0:
            raise ValueError("Non-physical y0 derived from omega_lambda")

        omega_r0 = max(bg.omega_r, 0.0)
        if omega_r0 >= 1.0:
            raise ValueError("omega_r must be < 1")

        if self.x0 * self.x0 + y0 * y0 + omega_r0 >= 1.0:
            raise ValueError("Initial conditions violate closure: x0^2+y0^2+Ωr0 must be < 1")

        if self.potential == "inverse_power" and self.phi0 <= 0:
            raise ValueError("phi0 must be > 0 for inverse_power")

        # state: [x, y, omega_r, phi]
        # phi is only used for inverse_power; for exponential we still carry it (constant) for simplicity.
        y_init = np.array([self.x0, y0, omega_r0, self.phi0], dtype=float)

        def rhs(n: float, y: np.ndarray) -> np.ndarray:
            x, yv, omega_r, phi = float(y[0]), float(y[1]), float(y[2]), float(y[3])

            # Guard against numerical drift.
            yv = max(yv, 0.0)
            omega_r = max(omega_r, 0.0)
            omega_phi = x * x + yv * yv
            omega_m = 1.0 - omega_phi - omega_r
            if omega_m <= 0.0:
                raise ValueError("Encountered Ω_m <= 0 during integration")

            lam_eff = _lambda_for_potential(self.potential, lam=self.lam, alpha=self.alpha, phi=phi)
            w_eff = (x * x - yv * yv) + omega_r / 3.0
            hdot_over_h2 = -1.5 * (1.0 + w_eff)

            dx = -3.0 * x + math.sqrt(1.5) * lam_eff * (yv * yv) - x * hdot_over_h2
            dy = -math.sqrt(1.5) * lam_eff * x * yv - yv * hdot_over_h2

            # Ω_r = ρ_r/(3 Mpl^2 H^2), with ρ_r∝a^-4 => dΩ_r/dN = Ω_r(-4 - d ln H^2/dN)
            domega_r = omega_r * (-4.0 - 2.0 * hdot_over_h2)

            dphi = math.sqrt(6.0) * x
            return np.array([dx, dy, domega_r, dphi], dtype=float)

        n_eval = np.linspace(n0, nmin, self.n_eval)
        sol = integrate.solve_ivp(
            rhs,
            t_span=(n0, nmin),
            y0=y_init,
            t_eval=n_eval,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success or sol.y is None:
            raise RuntimeError(f"Scalar-field integration failed: {sol.message}")

        # Cache (ensure ascending N for interpolation).
        t = sol.t
        y = sol.y
        if t[0] > t[-1]:
            t = t[::-1]
            y = y[:, ::-1]

        self._cache_bg = bg
        self._cache_n = t
        self._cache_state = y.T  # shape (n_eval, 4)

    def _interp_state(self, n: float) -> tuple[float, float, float, float]:
        assert self._cache_n is not None and self._cache_state is not None
        n_grid = self._cache_n
        state = self._cache_state

        if n < n_grid.min() or n > n_grid.max():
            raise ValueError("Requested redshift outside cached integration range")

        x = float(np.interp(n, n_grid, state[:, 0]))
        yv = float(np.interp(n, n_grid, state[:, 1]))
        omega_r = float(np.interp(n, n_grid, state[:, 2]))
        phi = float(np.interp(n, n_grid, state[:, 3]))
        return x, yv, omega_r, phi

    def evaluate(self, z: float, bg: CosmologyBackground) -> MechanismOutput:
        ensure_z_nonnegative(z)
        if z > self.z_max:
            raise ValueError(f"z={z} exceeds z_max={self.z_max}; increase z_max for this mechanism")

        self._ensure_cache(bg)
        n = -math.log1p(z)
        x, yv, omega_r, _phi = self._interp_state(n)

        omega_phi = max(0.0, x * x + yv * yv)
        omega_r = max(0.0, omega_r)
        omega_m = 1.0 - omega_phi - omega_r
        if omega_m <= 0.0:
            raise ValueError("Non-physical Ω_m <= 0")

        # Convert the dimensionless fractions to an SI energy density.
        # Use Ω_m(z) = Ω_m0 (1+z)^3 / E(z)^2 => E(z)^2 = Ω_m0 (1+z)^3 / Ω_m(z)
        ez2 = bg.omega_m * (1.0 + z) ** 3 / omega_m

        # u_crit0 inferred from ΩΛ = u_Λ0/u_crit0.
        u_crit0 = bg.rho_lambda0_j_m3 / bg.omega_lambda
        rho_de = omega_phi * u_crit0 * ez2

        denom = omega_phi
        w_phi = None
        if denom > 0:
            w_phi = (x * x - yv * yv) / denom

        return MechanismOutput(
            result=MechanismResult(z=z, rho_de_j_m3=rho_de, w_de=w_phi),
            assumptions=self.describe_assumptions(),
        )

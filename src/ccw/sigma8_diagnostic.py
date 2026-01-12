"""σ8 / fσ8 diagnostic utilities.

This module provides a minimal, self-contained linear-growth calculation to
support a σ8 tension diagnostic and redshift-space distortion (RSD) style
constraints via fσ8(z).

Scope / intent
--------------
- Uses a background expansion H(z) provided by the caller.
- Solves the standard linear growth ODE for matter perturbations in GR:

  D'' + (2 + d ln H / d ln a) D' - 3/2 Ω_m(a) D = 0

  where derivatives are w.r.t. ln a and D is the (unnormalized) growth factor.
- Normalizes D(a=1) = 1.

This is intentionally a diagnostic layer (Phase I.20). It is not a full
Boltzmann-code replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
from scipy import integrate


@dataclass(frozen=True)
class FSigma8Observable:
    """A measurement of fσ8 at redshift z."""

    z: float
    f_sigma8: float
    sigma: float
    source: str = ""


def _h_of_a_s_inv(a: float, hz_s_inv: Callable[[float], float]) -> float:
    if a <= 0:
        raise ValueError("a must be > 0")
    z = (1.0 / a) - 1.0
    return float(hz_s_inv(float(z)))


def _dlnh_dln_a(
    a: float,
    hz_s_inv: Callable[[float], float],
    rel_step: float = 1e-4,
) -> float:
    """Finite-difference derivative d ln H / d ln a at scale factor a."""

    # Work in ln(a) to keep symmetry.
    if a <= 0:
        raise ValueError("a must be > 0")

    x = np.log(a)
    dx = rel_step

    a_minus = float(np.exp(x - dx))
    a_plus = float(np.exp(x + dx))

    # Avoid evaluating z<0 (a>1) for late-time derivatives.
    # Use a one-sided difference near boundaries.
    if a_plus > 1.0:
        h0 = _h_of_a_s_inv(a, hz_s_inv)
        h_minus = _h_of_a_s_inv(a_minus, hz_s_inv)
        return float((np.log(h0) - np.log(h_minus)) / dx)

    if a_minus <= 0.0:
        h_plus = _h_of_a_s_inv(a_plus, hz_s_inv)
        h0 = _h_of_a_s_inv(a, hz_s_inv)
        return float((np.log(h_plus) - np.log(h0)) / dx)

    h_minus = _h_of_a_s_inv(a_minus, hz_s_inv)
    h_plus = _h_of_a_s_inv(a_plus, hz_s_inv)

    # d ln H / d ln a = d ln H / dx
    return float((np.log(h_plus) - np.log(h_minus)) / (2.0 * dx))


def _omega_m_of_a(
    a: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
) -> float:
    """Matter fraction Ω_m(a) for a flat matter+DE cosmology.

    Uses Ω_m(a) = Ω_m0 a^{-3} (H0/H(a))^2.
    """

    if not (0.0 < omega_m0 < 1.0):
        raise ValueError("omega_m0 must be between 0 and 1")

    h0 = _h_of_a_s_inv(1.0, hz_s_inv)
    h_a = _h_of_a_s_inv(a, hz_s_inv)

    return float(omega_m0 * (a ** -3) * (h0 / h_a) ** 2)


@dataclass(frozen=True)
class GrowthSolution:
    """Solution for growth factor D(a) and derivative dD/dln a."""

    a_grid: np.ndarray
    d: np.ndarray
    d_prime: np.ndarray

    def d_normalized(self) -> np.ndarray:
        d1 = float(self.d[-1])
        return self.d / d1

    def dprime_normalized(self) -> np.ndarray:
        d1 = float(self.d[-1])
        return self.d_prime / d1


def solve_linear_growth(
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
    a_min: float = 1e-3,
    a_max: float = 1.0,
    n_steps: int = 800,
) -> GrowthSolution:
    """Solve linear growth ODE from a_min to a_max.

    Parameters
    ----------
    hz_s_inv:
        Callable returning H(z) in s^-1.
    omega_m0:
        Present-day matter fraction.
    a_min:
        Starting scale factor (default 1e-3).
    a_max:
        Final scale factor (default 1.0).
    n_steps:
        Number of sample points for the returned grid.

    Returns
    -------
    GrowthSolution
        D(a) and D'(a)=dD/dln a on an a-grid.
    """

    if not (0.0 < a_min < a_max):
        raise ValueError("Require 0 < a_min < a_max")

    x0 = float(np.log(a_min))
    x1 = float(np.log(a_max))

    # Initial conditions in matter domination: D ~ a so dD/dln a = D.
    d0 = a_min
    dp0 = d0

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        a = float(np.exp(x))
        d, dp = float(y[0]), float(y[1])

        dlnh = _dlnh_dln_a(a, hz_s_inv)
        omega_m_a = _omega_m_of_a(a, hz_s_inv, omega_m0)

        # y = [D, D'] with ' = d/dln a
        dd = dp
        ddp = -(2.0 + dlnh) * dp + 1.5 * omega_m_a * d
        return np.array([dd, ddp], dtype=float)

    a_grid = np.geomspace(a_min, a_max, n_steps)
    x_eval = np.log(a_grid)

    sol = integrate.solve_ivp(
        rhs,
        t_span=(x0, x1),
        y0=np.array([d0, dp0], dtype=float),
        t_eval=x_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.success:
        raise RuntimeError(f"Growth ODE solve failed: {sol.message}")

    d = sol.y[0]
    dp = sol.y[1]

    return GrowthSolution(a_grid=a_grid, d=d, d_prime=dp)


def growth_factor_D(
    z: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
    a_min: float = 1e-3,
) -> float:
    """Return normalized growth factor D(z) with D(z=0)=1."""

    if z < 0:
        raise ValueError("z must be >= 0")

    a = 1.0 / (1.0 + z)
    sol = solve_linear_growth(hz_s_inv=hz_s_inv, omega_m0=omega_m0, a_min=a_min, a_max=1.0)

    # Interpolate D(a) and D'(a) in ln(a)
    x_grid = np.log(sol.a_grid)
    d_norm = sol.d_normalized()

    x = float(np.log(a))
    d_at = float(np.interp(x, x_grid, d_norm))
    return d_at


def growth_rate_f(
    z: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
    a_min: float = 1e-3,
) -> float:
    """Return growth rate f(z) = d ln D / d ln a."""

    if z < 0:
        raise ValueError("z must be >= 0")

    a = 1.0 / (1.0 + z)
    sol = solve_linear_growth(hz_s_inv=hz_s_inv, omega_m0=omega_m0, a_min=a_min, a_max=1.0)

    x_grid = np.log(sol.a_grid)
    d_norm = sol.d_normalized()
    dp_norm = sol.dprime_normalized()

    x = float(np.log(a))
    d_at = float(np.interp(x, x_grid, d_norm))
    dp_at = float(np.interp(x, x_grid, dp_norm))

    # f = D'/D with derivatives in ln a.
    if d_at <= 0:
        raise RuntimeError("Non-positive growth factor encountered")

    return float(dp_at / d_at)


def sigma8_z(
    z: float,
    sigma8_0: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
) -> float:
    """Return σ8(z) assuming σ8(z)=σ8(0) D(z)."""

    d = growth_factor_D(z=z, hz_s_inv=hz_s_inv, omega_m0=omega_m0)
    return float(sigma8_0 * d)


def f_sigma8_z(
    z: float,
    sigma8_0: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
) -> float:
    """Return fσ8(z)."""

    f = growth_rate_f(z=z, hz_s_inv=hz_s_inv, omega_m0=omega_m0)
    s8 = sigma8_z(z=z, sigma8_0=sigma8_0, hz_s_inv=hz_s_inv, omega_m0=omega_m0)
    return float(f * s8)


def get_boss_dr12_fsigma8_observables() -> List[FSigma8Observable]:
    """A small BOSS DR12-like fσ8 dataset (illustrative).

    Notes
    -----
    These are representative values commonly used in forecasting / pedagogy.
    They are not intended to be a definitive compilation.
    """

    return [
        FSigma8Observable(z=0.38, f_sigma8=0.497, sigma=0.045, source="BOSS DR12 (illustrative)"),
        FSigma8Observable(z=0.51, f_sigma8=0.458, sigma=0.038, source="BOSS DR12 (illustrative)"),
        FSigma8Observable(z=0.61, f_sigma8=0.436, sigma=0.034, source="BOSS DR12 (illustrative)"),
    ]


def fsigma8_chi_squared(
    obs: Sequence[FSigma8Observable],
    sigma8_0: float,
    hz_s_inv: Callable[[float], float],
    omega_m0: float,
) -> float:
    chi2 = 0.0
    for o in obs:
        theory = f_sigma8_z(o.z, sigma8_0=sigma8_0, hz_s_inv=hz_s_inv, omega_m0=omega_m0)
        chi2 += ((o.f_sigma8 - theory) / o.sigma) ** 2
    return float(chi2)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .constants import C_M_S, G_M3_KG_S2, PI
from .cosmology import ObservedLambdaResult, observed_lambda_from_h0_omega
from .mechanisms.scalar_field import ScalarFieldQuintessence


@dataclass(frozen=True)
class LambdaObservables:
    """Convenience container for (ΩΛ, H0) <-> (ρΛ, Λ) conversions."""

    h0_km_s_mpc: float
    omega_lambda: float
    rho_lambda_j_m3: float
    lambda_m_inv2: float


def from_h0_omega(h0_km_s_mpc: float, omega_lambda: float) -> LambdaObservables:
    obs: ObservedLambdaResult = observed_lambda_from_h0_omega(h0_km_s_mpc, omega_lambda)
    return LambdaObservables(
        h0_km_s_mpc=h0_km_s_mpc,
        omega_lambda=omega_lambda,
        rho_lambda_j_m3=obs.rho_lambda_j_m3,
        lambda_m_inv2=obs.lambda_m_inv2,
    )


def omega_lambda_from_lambda_h0(lambda_m_inv2: float, h0_s_inv: float) -> float:
    """ΩΛ = Λ c^2 / (3 H0^2)."""

    if lambda_m_inv2 < 0:
        raise ValueError("Λ must be >= 0 for this bookkeeping conversion")
    if h0_s_inv <= 0:
        raise ValueError("H0 must be positive")
    return lambda_m_inv2 * (C_M_S**2) / (3.0 * (h0_s_inv**2))


def lambda_from_rho_lambda(rho_lambda_j_m3: float) -> float:
    """Λ = 8πG ρ / c^4 when ρ is an energy density (J/m^3)."""

    if rho_lambda_j_m3 < 0:
        raise ValueError("ρ_Λ must be >= 0 for this bookkeeping conversion")
    return (8.0 * PI * G_M3_KG_S2 * rho_lambda_j_m3) / (C_M_S**4)


def rho_lambda_from_lambda(lambda_m_inv2: float) -> float:
    """ρ = Λ c^4 / (8πG) when ρ is an energy density (J/m^3)."""

    if lambda_m_inv2 < 0:
        raise ValueError("Λ must be >= 0 for this bookkeeping conversion")
    return (lambda_m_inv2 * (C_M_S**4)) / (8.0 * PI * G_M3_KG_S2)


def sanity_check_background(omega_m: float, omega_lambda: float, omega_r: float = 0.0, omega_k: float = 0.0) -> None:
    """Basic bounds and consistency checks (not a cosmological fit)."""

    for name, val in [("Ωm", omega_m), ("ΩΛ", omega_lambda), ("Ωr", omega_r), ("Ωk", omega_k)]:
        if not (-5.0 <= val <= 5.0):
            raise ValueError(f"{name} outside sanity range")

    total = omega_m + omega_lambda + omega_r + omega_k
    if not (0.0 <= total <= 5.0):
        raise ValueError("Ω total outside sanity range")


@dataclass(frozen=True)
class ConstraintCheck:
    ok: bool
    detail: str


def swampland_ds_gradient_bound_exponential(lam: float, *, c_min: float = 2.0 / (3.0**0.5)) -> ConstraintCheck:
    """Refined dS swampland gradient bound for V ∝ exp(-λ φ).

    In reduced Planck units, the bound is often written as:
      |V'|/V >= c_min

    For exponential potentials, |V'|/V = λ (constant).

    Note: In this workbench's scalar-field implementation, `lam` is already the
    dimensionless quantity M_pl |V'|/V.
    """

    if lam < 0:
        return ConstraintCheck(ok=False, detail="λ must be >= 0")
    if lam + 0.0 < c_min:
        return ConstraintCheck(ok=False, detail=f"Swampland violation: λ={lam:.3g} < c_min={c_min:.3g}")
    return ConstraintCheck(ok=True, detail=f"Passes swampland gradient bound: λ={lam:.3g} >= c_min={c_min:.3g}")


def swampland_ds_gradient_bound_inverse_power(
    alpha: float,
    *,
    phi_values: Iterable[float],
    c_min: float = 2.0 / (3.0**0.5),
) -> ConstraintCheck:
    """Gradient bound check for V ∝ φ^{-α} in reduced Planck units.

    For inverse-power potentials, |V'|/V = α/φ.
    """

    if alpha <= 0:
        return ConstraintCheck(ok=False, detail="α must be > 0")

    min_ratio: Optional[float] = None
    for phi in phi_values:
        if phi <= 0:
            return ConstraintCheck(ok=False, detail="Encountered φ <= 0 (inverse-power requires φ>0)")
        ratio = alpha / phi
        min_ratio = ratio if min_ratio is None else min(min_ratio, ratio)

    if min_ratio is None:
        return ConstraintCheck(ok=False, detail="No φ values provided")
    if min_ratio < c_min:
        return ConstraintCheck(ok=False, detail=f"Swampland violation: min(α/φ)={min_ratio:.3g} < c_min={c_min:.3g}")
    return ConstraintCheck(ok=True, detail=f"Passes swampland gradient bound: min(α/φ)={min_ratio:.3g} >= c_min={c_min:.3g}")


def check_scalar_field_swampland(
    mech: ScalarFieldQuintessence,
    bg,
    *,
    z_values: Iterable[float],
    c_min: float = 2.0 / (3.0**0.5),
) -> ConstraintCheck:
    """Evaluate a scalar-field mechanism trajectory against the gradient bound.

    This uses the mechanism's cached φ(z) trajectory (for inverse-power), making
    it compatible with the explicit ODE evolution already in the workbench.
    """

    # Ensure cache built once.
    mech.evaluate(0.0, bg)

    if mech.potential == "exponential":
        return swampland_ds_gradient_bound_exponential(mech.lam, c_min=c_min)

    if mech.potential == "inverse_power":
        # Pull φ(z) values from the mechanism cache via its private interpolation.
        phi_vals: list[float] = []
        for z in z_values:
            # Evaluate once to enforce z_max etc.
            mech.evaluate(z, bg)
            n = -__import__("math").log1p(z)
            _x, _y, _or, phi = mech._interp_state(n)  # type: ignore[attr-defined]
            phi_vals.append(phi)
        return swampland_ds_gradient_bound_inverse_power(mech.alpha, phi_values=phi_vals, c_min=c_min)

    return ConstraintCheck(ok=False, detail=f"Unsupported potential: {mech.potential}")


def holographic_energy_density_bound_j_m3(L_m: float) -> float:
    """Simple holographic-style bound using Λ ~ 3/L^2.

    If Λ_holo = 3/L^2, then ρ_holo = Λ_holo c^4/(8πG) = 3 c^4/(8πG L^2).

    Returns an *energy* density in J/m^3.
    """

    if L_m <= 0:
        raise ValueError("L must be > 0")
    return (3.0 * (C_M_S**4)) / (8.0 * PI * G_M3_KG_S2 * (L_m**2))


def holographic_bound_from_hz(
    z: float,
    *,
    hz_s_inv: Callable[[float], float],
    rho_de_j_m3: float,
    c_factor: float = 1.0,
) -> ConstraintCheck:
    """Check ρ_DE(z) against a holographic bound with L(z)=c/H(z).

    Uses L(z) = c/(c_factor * H(z)). With c_factor=1, L is the Hubble radius.
    """

    if c_factor <= 0:
        return ConstraintCheck(ok=False, detail="c_factor must be > 0")
    if rho_de_j_m3 < 0:
        return ConstraintCheck(ok=False, detail="ρ_DE must be >= 0")

    hz = hz_s_inv(z)
    if hz <= 0:
        return ConstraintCheck(ok=False, detail="H(z) must be > 0")
    L = C_M_S / (c_factor * hz)
    bound = holographic_energy_density_bound_j_m3(L)
    if rho_de_j_m3 > bound:
        return ConstraintCheck(ok=False, detail=f"Holographic violation: ρ_DE={rho_de_j_m3:.3e} > ρ_bound={bound:.3e}")
    return ConstraintCheck(ok=True, detail=f"Passes holographic bound: ρ_DE={rho_de_j_m3:.3e} <= ρ_bound={bound:.3e}")

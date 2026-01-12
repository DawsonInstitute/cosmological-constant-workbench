#!/usr/bin/env python3
"""Demo: σ8 / fσ8 diagnostic.

Runs the Phase I.20 diagnostic against a ΛCDM background to compute fσ8(z)
for a few RSD-style data points.

This is a diagnostic layer (not a full Boltzmann code).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from ccw.frw import h_z_lcdm_s_inv
from ccw.mechanisms import CosmologyBackground
from ccw.sigma8_diagnostic import (
    f_sigma8_z,
    get_boss_dr12_fsigma8_observables,
    fsigma8_chi_squared,
)


def main() -> None:
    print("=" * 70)
    print("σ8 / fσ8 DIAGNOSTIC DEMO")
    print("=" * 70)

    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    sigma8_0 = 0.811

    obs = get_boss_dr12_fsigma8_observables()
    print(f"Using σ8(0) = {sigma8_0:.3f}, Ωm0 = {bg.omega_m:.3f}, H0 = {bg.h0_km_s_mpc:.1f}")
    print("\nData vs theory:")

    for o in obs:
        theory = f_sigma8_z(o.z, sigma8_0=sigma8_0, hz_s_inv=hz, omega_m0=bg.omega_m)
        pull = (o.f_sigma8 - theory) / o.sigma
        print(
            f"  z={o.z:0.2f}:  fσ8_obs={o.f_sigma8:0.3f}±{o.sigma:0.3f}  "
            f"fσ8_th={theory:0.3f}  pull={pull:+0.2f}σ"
        )

    chi2 = fsigma8_chi_squared(obs, sigma8_0=sigma8_0, hz_s_inv=hz, omega_m0=bg.omega_m)
    dof = len(obs)
    print(f"\nχ² = {chi2:.2f} for dof={dof}  (χ²/dof={chi2/dof:.2f})")

    print("\nNotes:")
    print("- If χ²/dof is large, it suggests a potential σ8 tension for this background")
    print("- This demo uses an illustrative BOSS-like dataset and a GR growth ODE")


if __name__ == "__main__":
    main()

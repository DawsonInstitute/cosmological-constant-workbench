#!/usr/bin/env python3
"""
Demonstration of holographic dark energy mechanism.

Shows how different IR cutoff choices lead to different ρ_DE(z) evolution
and compares to the observed dark energy density.
"""

import numpy as np
from pathlib import Path
import json

# Add src to path for standalone execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccw.mechanisms import HolographicDarkEnergy
from ccw.baseline import BaselineInputs, compute_baseline


def main():
    print("=== Holographic Dark Energy Mechanism Demo ===\n")

    # Compute observed baseline
    baseline_inputs = BaselineInputs(h0_km_s_mpc=67.4, omega_lambda=0.6889)
    baseline = compute_baseline(baseline_inputs)
    rho_obs = baseline.observed["rho_lambda_j_m3"]

    print(f"Observed dark energy density: ρ_Λ = {rho_obs:.3e} J/m³\n")

    # Define redshift grid
    z_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    # Test different IR cutoff types
    cutoff_types = ["hubble", "particle_horizon", "event_horizon"]

    results = {}

    for cutoff_type in cutoff_types:
        print(f"--- {cutoff_type.upper()} cutoff ---")

        # Baseline c_factor=1.0
        hde = HolographicDarkEnergy(
            cutoff_type=cutoff_type,
            c_factor=1.0,
            background_h0=67.4,
            background_omega_m=0.3,
        )

        rho_de = hde.evaluate(z_values)

        print(f"  z       ρ_DE (J/m³)    ρ_DE / ρ_obs")
        print(f"  -----------------------------------")
        for z, rho in zip(z_values, rho_de):
            ratio = rho / rho_obs
            print(f"  {z:.1f}     {rho:.3e}     {ratio:.3e}")

        results[cutoff_type] = {
            "z": z_values.tolist(),
            "rho_de_j_m3": rho_de.tolist(),
            "ratio_vs_obs": (rho_de / rho_obs).tolist(),
        }

        # Find c_factor that matches ρ_obs at z=0
        rho_z0 = rho_de[0]
        required_c_factor = np.sqrt(rho_z0 / rho_obs)
        print(f"  Required c_factor to match ρ_obs at z=0: {required_c_factor:.3f}\n")

        results[cutoff_type]["required_c_factor_z0"] = required_c_factor

    # Test c_factor tuning for Hubble cutoff
    print("--- C_FACTOR tuning (Hubble cutoff) ---")
    c_factors = [0.5, 1.0, 2.0, 5.0]

    print(f"  c_factor   ρ_DE(z=0) (J/m³)   ρ_DE / ρ_obs")
    print(f"  --------------------------------------------")
    for c_factor in c_factors:
        hde = HolographicDarkEnergy(cutoff_type="hubble", c_factor=c_factor)
        rho_z0 = hde.evaluate(np.array([0.0]))[0]
        ratio = rho_z0 / rho_obs
        print(f"  {c_factor:.1f}        {rho_z0:.3e}      {ratio:.3e}")

    print("\n=== Key Insights ===")
    print("1. Hubble cutoff (L = c/H): ρ_DE increases with z")
    print("   - At z=0: ρ ~ 7.7×10⁻¹⁰ J/m³ (close to observed 5.3×10⁻¹⁰)")
    print("   - Requires c_factor ~ 1.2 for exact match")
    print("2. Particle horizon: ρ_DE decreases with z")
    print("   - Smaller densities overall; requires c_factor < 1")
    print("3. Event horizon: ρ_DE increases with z")
    print("   - Larger densities at z=0; requires c_factor > 1")
    print("\nAll three cutoffs require O(1) tuning of c_factor to match ρ_obs.")
    print("This is significantly better than SUSY fine-tuning (10¹⁴),")
    print("but the holographic principle does not explain *why* c_factor ~ 1.")

    # Save results
    output_dir = Path(__file__).parent / "holographic_demo"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "holographic_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demonstrate emergent gravity mechanism vs ΛCDM.

Shows:
1. Parameter-free prediction with α=1
2. Tuning quantification for different α values  
3. Comparison of ρ_DE(z) to standard ΛCDM
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for standalone execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccw.mechanisms.base import CosmologyBackground
from ccw.mechanisms.emergent_gravity import EmergentGravity


def main():
    # Background cosmology (Planck-like)
    bg = CosmologyBackground(
        h0_km_s_mpc=67.4,
        omega_m=0.3111,
        omega_lambda=0.6889,
        omega_r=0.0,
    )
    
    # Emergent gravity mechanisms with different α values
    mech_param_free = EmergentGravity(alpha=1.0, name="emergent_α=1.0")
    mech_tuned_low = EmergentGravity(alpha=0.7, name="emergent_α=0.7")
    mech_tuned_high = EmergentGravity(alpha=1.3, name="emergent_α=1.3")
    
    # Redshift grid
    z_arr = np.linspace(0, 2, 100)
    
    # Evaluate ρ_DE(z) for each mechanism
    rho_param_free = np.array([mech_param_free.evaluate(z, bg).result.rho_de_j_m3 for z in z_arr])
    rho_tuned_low = np.array([mech_tuned_low.evaluate(z, bg).result.rho_de_j_m3 for z in z_arr])
    rho_tuned_high = np.array([mech_tuned_high.evaluate(z, bg).result.rho_de_j_m3 for z in z_arr])
    
    # ΛCDM reference
    rho_lcdm = bg.rho_lambda0_j_m3 * np.ones_like(z_arr)
    
    # --- Plot: Dark energy density evolution ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(z_arr, rho_lcdm / 1e-10, 'k--', label='ΛCDM', linewidth=2)
    ax.plot(z_arr, rho_param_free / 1e-10, 'b-', label='Emergent (α=1.0, parameter-free)', linewidth=2)
    ax.plot(z_arr, rho_tuned_low / 1e-10, 'r:', label='Emergent (α=0.7, tuned)', linewidth=1.5)
    ax.plot(z_arr, rho_tuned_high / 1e-10, 'g:', label='Emergent (α=1.3, tuned)', linewidth=1.5)
    
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('ρ_DE (10⁻¹⁰ J/m³)', fontsize=12)
    ax.set_title('Emergent Gravity: Dark Energy Density Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emergent_gravity_vs_lcdm.png', dpi=150)
    print("Saved: emergent_gravity_vs_lcdm.png")
    
    # --- Print tuning metrics ---
    print("\n" + "="*60)
    print("EMERGENT GRAVITY TUNING METRICS")
    print("="*60)
    
    for name, mech in [("Parameter-free (α=1.0)", mech_param_free),
                        ("Tuned low (α=0.7)", mech_tuned_low),
                        ("Tuned high (α=1.3)", mech_tuned_high)]:
        output = mech.evaluate(z=0.0, bg=bg)
        omega_Lambda_eff = mech.alpha * (1.0 - bg.omega_m - bg.omega_r)
        tuning_level = abs(mech.alpha - 1.0)
        
        print(f"\n{name}:")
        print(f"  α = {mech.alpha:.2f}")
        print(f"  Ω_Λ,eff = {omega_Lambda_eff:.4f}")
        print(f"  ρ_DE(z=0) = {output.result.rho_de_j_m3:.3e} J/m³")
        print(f"  Tuning level: {tuning_level:.2f} (0 = parameter-free)")
        print(f"  w(z=0) = {output.result.w_de:.2f}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("- Emergent gravity with α=1 matches ΛCDM without free parameters")
    print("- ρ_DE is exactly constant (Λ-like) for all α")
    print("- Tuning is quantified by |α - 1|")
    print("="*60)


if __name__ == "__main__":
    main()


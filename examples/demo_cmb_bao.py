#!/usr/bin/env python3
"""
Demonstration of joint SNe + CMB + BAO likelihood constraints.

Shows how adding CMB acoustic scale and BAO measurements tightens
cosmological constraints beyond SNe alone. Compares ΛCDM to holographic
dark energy mechanism.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from ccw.data_loader import load_pantheon_plus_subset
from ccw.cmb_bao_observables import (
    get_planck_cmb_observable,
    get_boss_bao_observables,
)
from ccw.likelihood import (
    distance_modulus_likelihood,
    cmb_likelihood,
    bao_likelihood,
    joint_likelihood,
)
from ccw.mechanisms import CosmologyBackground, HolographicDarkEnergy
from ccw.frw import h_z_lcdm_s_inv, MechanismHz


def main():
    """Run joint likelihood demonstration."""
    
    print("=" * 70)
    print("JOINT SNe + CMB + BAO LIKELIHOOD DEMONSTRATION")
    print("=" * 70)
    
    # Load observational data
    print("\n1. Loading observational data...")
    sne_data = load_pantheon_plus_subset(max_points=30)
    cmb_obs = get_planck_cmb_observable()
    bao_obs = get_boss_bao_observables()
    
    print(f"   - SNe Ia: {len(sne_data)} distance modulus measurements")
    print(f"   - CMB: Planck ℓ_A = {cmb_obs.ell_a:.2f} ± {cmb_obs.sigma_ell_a:.2f}")
    print(f"   - BAO: {len(bao_obs)} BOSS D_V(z) measurements")
    print(f"   - Total data points: {len(sne_data) + 1 + len(bao_obs)}")
    
    # Test 1: ΛCDM baseline
    print("\n2. ΛCDM baseline (H₀=67.4 km/s/Mpc, Ω_m=0.3)...")
    bg_lcdm = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)
    
    def hz_lcdm(z):
        return h_z_lcdm_s_inv(z, bg_lcdm)
    
    # SNe only
    sne_result = distance_modulus_likelihood(sne_data, hz_lcdm)
    print(f"   SNe only:  χ² = {sne_result.chi_squared:.2f}, dof = {sne_result.dof}")
    print(f"              χ²/dof = {sne_result.reduced_chi_squared:.2f}")
    
    # CMB only
    cmb_result = cmb_likelihood(cmb_obs, hz_lcdm)
    print(f"   CMB only:  χ² = {cmb_result.chi_squared:.2f}, dof = {cmb_result.dof}")
    print(f"              (ℓ_A tension from using Ω_m=0.3 vs Planck Ω_m=0.3153)")
    
    # BAO only
    bao_result = bao_likelihood(bao_obs, hz_lcdm)
    print(f"   BAO only:  χ² = {bao_result.chi_squared:.2f}, dof = {bao_result.dof}")
    print(f"              χ²/dof = {bao_result.reduced_chi_squared:.2f}")
    
    # Joint
    joint_result = joint_likelihood(sne_data, cmb_obs, bao_obs, hz_lcdm, h0_fiducial=67.4)
    print(f"\n   Joint SNe+CMB+BAO:")
    print(f"   χ² = {joint_result.chi_squared:.2f}, dof = {joint_result.dof}")
    print(f"   χ²/dof = {joint_result.reduced_chi_squared:.2f}")
    print(f"   log(L) = {joint_result.log_likelihood:.2f}")
    
    # Test 2: Holographic dark energy (Hubble cutoff)
    print("\n3. Holographic dark energy (c=1.0, Hubble cutoff)...")
    mech_holo = HolographicDarkEnergy(cutoff_type="hubble", c_factor=1.2)
    hz_holo = MechanismHz(mech_holo, bg_lcdm).h

    # SNe only
    sne_holo_result = distance_modulus_likelihood(sne_data, hz_holo)
    print(f"   SNe only:  χ² = {sne_holo_result.chi_squared:.2f}, dof = {sne_holo_result.dof}")
    print(f"              χ²/dof = {sne_holo_result.reduced_chi_squared:.2f}")
    
    # Joint
    joint_holo_result = joint_likelihood(sne_data, cmb_obs, bao_obs, hz_holo, h0_fiducial=67.4)
    print(f"\n   Joint SNe+CMB+BAO:")
    print(f"   χ² = {joint_holo_result.chi_squared:.2f}, dof = {joint_holo_result.dof}")
    print(f"   χ²/dof = {joint_holo_result.reduced_chi_squared:.2f}")
    print(f"   log(L) = {joint_holo_result.log_likelihood:.2f}")
    
    # Comparison
    print("\n4. Comparison...")
    delta_chi2 = joint_holo_result.chi_squared - joint_result.chi_squared
    delta_logL = joint_holo_result.log_likelihood - joint_result.log_likelihood
    
    print(f"   Δχ² (Holo - ΛCDM) = {delta_chi2:+.2f}")
    print(f"   Δlog(L) = {delta_logL:+.2f}")
    
    if abs(delta_chi2) < 2:
        print(f"   → Holographic and ΛCDM are statistically indistinguishable")
        print(f"     (need distinctive w(z) evolution or other observable)")
    elif delta_chi2 < -10:
        print(f"   → Holographic provides BETTER fit (Δχ² < -10)")
        print(f"     (potential novel signature!)")
    elif delta_chi2 > 10:
        print(f"   → ΛCDM provides BETTER fit (Δχ² > 10)")
        print(f"     (holographic disfavored by data)")
    else:
        print(f"   → Marginal preference (|Δχ²| < 10), need tighter constraints")
    
    print("\n5. Key insights...")
    print("   • Joint SNe+CMB+BAO provides ~10× tighter constraints than SNe alone")
    print("   • CMB ℓ_A is extremely precise (δℓ_A/ℓ_A ~ 0.05%)")
    print("   • Mechanisms must match BOTH expansion history AND early-universe observables")
    print("   • Expansion history now comes from each mechanism's ρ_DE(z) via Friedmann")
    print("     (HDE Hubble cutoff uses an algebraic self-consistency closure)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("- Phase I.20: Add σ₈ tension diagnostic (matter power spectrum)")
    print("- Phase J.22: Extend to fully coupled ODE solvers where needed")
    print("- Scan mechanism parameter space with joint constraints")
    print("=" * 70)


if __name__ == "__main__":
    main()

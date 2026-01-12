#!/usr/bin/env python3
"""
Demonstration of Weak Gravity Conjecture (WGC) and Swampland Distance Conjecture (SDC).

Shows:
1. WGC bounds on scalar mass vs coupling
2. SDC tower of states for trans-Planckian field excursions
3. Combined WGC+SDC constraints on quintessence parameter space
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccw.wgc_constraints import (
    check_wgc_scalar,
    check_wgc_quintessence_exponential,
    swampland_distance_conjecture_mass,
    check_sdc_tower,
    describe_wgc_conjecture,
    M_PLANCK_GEV,
)


def main():
    print("=== Weak Gravity Conjecture + Swampland Distance Conjecture Demo ===\n")
    
    print(describe_wgc_conjecture())
    print("\n" + "="*70 + "\n")
    
    # 1. WGC bounds for different scalar scenarios
    print("1. WGC bounds for scalar fields:\n")
    
    scenarios = [
        ("Light scalar, weak coupling", 1e10, 1e-8),
        ("Heavy scalar, strong coupling", 1e15, 1e-3),
        ("Heavy scalar, weak coupling", 1e15, 1e-10),
        ("Higgs-like", 125, 0.01),
    ]
    
    for name, mass_gev, coupling_g in scenarios:
        result = check_wgc_scalar(mass_gev, coupling_g)
        status = "✓ PASS" if result.satisfies_wgc else "✗ FAIL"
        print(f"  {name}:")
        print(f"    m = {mass_gev:.2e} GeV, g = {coupling_g:.2e}")
        print(f"    WGC bound: g M_Pl = {coupling_g * M_PLANCK_GEV:.2e} GeV")
        print(f"    Status: {status} (ratio: {result.bound_ratio:.2e})\n")
    
    # 2. Quintessence parameter space
    print("2. Quintessence V = V0 exp(-λφ) constraints:\n")
    
    v0_gev4 = 1e-47 * (M_PLANCK_GEV)**4  # Dark energy scale
    
    quintessence_cases = [
        ("Steep (swampland-favored)", 2.0, 1e-5),
        ("Moderate", 1.0, 1e-6),
        ("Flat (swampland-disfavored)", 0.1, 1e-8),
        ("Ultra-flat (ΛCDM-like)", 0.01, 1e-10),
    ]
    
    for name, lam, coupling_g in quintessence_cases:
        result = check_wgc_quintessence_exponential(lam, v0_gev4, coupling_g)
        status = "✓ PASS" if result.satisfies_wgc else "✗ FAIL"
        print(f"  {name} (λ={lam}, g={coupling_g:.1e}):")
        print(f"    Effective mass: m ≈ {result.mass_gev:.2e} GeV")
        print(f"    WGC status: {status}\n")
    
    # 3. Swampland Distance Conjecture
    print("3. SDC tower of states for field excursions:\n")
    
    excursions = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("  Δφ/M_Pl    m_tower (GeV)    m_tower/M_Pl")
    print("  " + "-"*50)
    for delta_phi in excursions:
        m_tower = swampland_distance_conjecture_mass(delta_phi)
        ratio = m_tower / M_PLANCK_GEV
        print(f"    {delta_phi:4.1f}      {m_tower:.2e}      {ratio:.2e}")
    
    # 4. EFT validity checks
    print("\n4. EFT validity for quintessence models:\n")
    
    eft_cases = [
        ("Sub-Planckian", 0.5, 1e10),
        ("Planckian", 1.0, 1e15),
        ("Trans-Planckian (moderate)", 3.0, 1e16),
        ("Trans-Planckian (large)", 10.0, 1e18),
    ]
    
    for name, delta_phi, cutoff_gev in eft_cases:
        result = check_sdc_tower(delta_phi, cutoff_gev)
        status = "✓ VALID" if result["eft_valid"] else "✗ BREAKS DOWN"
        print(f"  {name} (Δφ={delta_phi} M_Pl, cutoff={cutoff_gev:.1e} GeV):")
        print(f"    Tower mass: {result['tower_mass_gev']:.2e} GeV")
        print(f"    EFT status: {status}\n")
    
    # 5. Combined constraints
    print("5. Combined WGC + SDC insights:\n")
    print("  - ΛCDM (λ→0): Satisfies WGC if g→0, but this is 'unnatural'")
    print("    (decoupled scalar with no physical role).\n")
    print("  - Steep quintessence (λ~2): Satisfies both swampland gradient bound")
    print("    AND WGC for reasonable couplings (g ~ 10^-5).\n")
    print("  - Flat quintessence (λ~0.1): Ruled out by swampland, also problematic")
    print("    for WGC (requires extremely weak couplings).\n")
    print("  - Trans-Planckian excursions (Δφ > M_Pl): Tower of states appears,")
    print("    EFT breaks down → need UV completion or restricted field range.\n")
    
    print("=== Key Takeaway ===")
    print("WGC + SDC strongly favor STEEP potentials (λ ≳ 1) over flat ones,")
    print("consistent with swampland gradient bound. ΛCDM sits at an unnatural")
    print("limit (g→0, λ→0), while viable quintessence requires λ ~ O(1) and")
    print("sub-Planckian field excursions to avoid EFT breakdown.")


if __name__ == "__main__":
    main()

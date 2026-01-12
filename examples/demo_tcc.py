#!/usr/bin/env python3
"""
Demonstration of Trans-Planckian Censorship Conjecture (TCC) constraints.

Shows:
1. ΛCDM satisfies TCC for late-time evolution
2. Standard inflation violates TCC (major tension)
3. How TCC constrains dark energy models
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccw.tcc_constraints import (
    check_tcc_lcdm,
    check_tcc_inflation,
    h0_km_s_mpc_to_gev,
    describe_tcc_conjecture,
    LAMBDA_TCC_GEV,
    M_PLANCK_GEV,
)


def main():
    print("=== Trans-Planckian Censorship Conjecture Demo ===\n")
    
    print(describe_tcc_conjecture())
    print("\n" + "="*60 + "\n")
    
    # 1. Check ΛCDM
    print("1. ΛCDM late-time evolution:")
    lcdm_result = check_tcc_lcdm(h0_km_s_mpc=67.4, omega_m=0.3, z_max=1100.0)
    print(f"   {lcdm_result.details}")
    print(f"   Status: {'✓ PASS' if lcdm_result.satisfies_tcc else '✗ FAIL'}\n")
    
    # 2. Check different inflation scales
    print("2. Inflationary epoch (different scales):")
    
    inflation_scales = [
        ("GUT-scale", 1e14),
        ("Intermediate", 1e10),
        ("Low-scale", 1e5),
        ("Ultra-low", 1e-13),
    ]
    
    for name, h_inf in inflation_scales:
        result = check_tcc_inflation(h_inf, n_efolds=60.0)
        status = "✓ PASS" if result.satisfies_tcc else "✗ FAIL"
        print(f"   {name} (H_inf = {h_inf:.1e} GeV):")
        print(f"     Margin: {result.margin:.2e}")
        print(f"     Status: {status}")
    
    print("\n3. TCC tension with cosmology:")
    print("   - Standard inflation requires H_inf ~ 10^13-10^14 GeV (from CMB)")
    print("   - TCC bound: H ≲ 10^-12 GeV")
    print("   - Conflict: ~25 orders of magnitude!")
    print("\n   Possible resolutions:")
    print("     a) TCC is incorrect (swampland conjectures are approximate)")
    print("     b) Inflation is modified (low-scale, bouncing, or emergent)")
    print("     c) Trans-Planckian physics is allowed (TCC censorship relaxed)")
    
    print("\n4. Dark energy constraints:")
    print("   - Current H_0 ~ 67 km/s/Mpc ~ 1.4×10^-42 GeV")
    print(f"   - TCC bound: H ≲ {LAMBDA_TCC_GEV:.0e} GeV")
    print("   - Headroom: ~30 orders of magnitude")
    print("   - Conclusion: Late-time dark energy easily satisfies TCC")
    
    print("\n=== Key Takeaway ===")
    print("TCC is satisfied by late-time cosmology (ΛCDM) but appears")
    print("incompatible with standard high-scale inflation. This is a")
    print("major open question in quantum gravity and early-universe physics.")


if __name__ == "__main__":
    main()

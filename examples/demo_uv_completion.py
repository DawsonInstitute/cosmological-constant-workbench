#!/usr/bin/env python3
"""Demonstration: UV completion checks for quintessence.

This demo illustrates how field excursions, operator dimensions, and strong
coupling constraints restrict viable quintessence parameter space.

Run:
  PYTHONPATH=src python examples/demo_uv_completion.py
"""

from ccw.uv_completion import quintessence_uv_check


def main() -> None:
    print("=" * 78)
    print("UV Completion Checks for Exponential Quintessence")
    print("=" * 78)
    print()
    print("Potential: V ~ M_Pl^4 exp(-λφ/M_Pl)")
    print("Field excursion: Δφ ~ λ M_Pl ln(1+z_max)")
    print()

    # Scan over λ and z_max
    lambda_values = [0.1, 0.5, 1.0, 2.0]
    z_max_values = [1.0, 5.0, 10.0]

    for z_max in z_max_values:
        print("-" * 78)
        print(f"z_max = {z_max:.1f}")
        print()
        for lam in lambda_values:
            result = quintessence_uv_check(lambda_coupling=lam, z_max=z_max, cutoff_gev=1e16)
            status = "✓ PASS" if result.uv_complete else "✗ FAIL"
            print(f"λ={lam:>4.1f}  Δφ/M_Pl={result.delta_phi_over_m_pl:>6.2f}  g_eff={result.g_eff:>5.2e}  {status}")
        print()

    print("-" * 78)
    print("Example: detailed check for λ=1.0, z_max=5.0")
    print("-" * 78)
    result = quintessence_uv_check(lambda_coupling=1.0, z_max=5.0)
    print(result.details)
    print()
    print("Interpretation:")
    print("- λ ~ O(1): trans-Planckian excursion (requires string theory UV completion)")
    print("- λ ~ 0.1: sub-Planckian (EFT valid, but see swampland constraints)")
    print("- Large z_max increases field excursion (pushes toward UV physics)")


if __name__ == "__main__":
    main()

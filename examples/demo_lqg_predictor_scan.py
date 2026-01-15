#!/usr/bin/env python3
"""Demonstration: bounded scan of LQG predictor knobs vs observed ρ_Λ.

Run:
  PYTHONPATH=src python examples/demo_lqg_predictor_scan.py

Goal:
  Quickly determine whether any bounded region of external predictor parameters
  yields ρ_vac close to observed ρ_Λ,0.

If not, the scan provides a no-go style empirical constraint:
  min |log10(ρ_pred/ρ_obs)| over the scanned region.
"""

import math

from ccw.integrations.lqg_predictor import lqg_predictor_available
from ccw.integrations.lqg_predictor_sweep import make_default_points, scan_points
from ccw.mechanisms import CosmologyBackground


def main() -> None:
    print("=" * 78)
    print("Hybrid Step: LQG Predictor Scan (bounded knobs)")
    print("=" * 78)
    print()

    if not lqg_predictor_available():
        print("⚠ LQG predictor not available.")
        print("This demo requires lqg-cosmological-constant-predictor to be importable.")
        return

    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_lambda=0.6889, omega_m=0.3111, omega_r=0.0)
    rho_obs = bg.rho_lambda0_j_m3

    points = make_default_points(target_scale_m=1e-15)
    best = scan_points(points, bg=bg, top_k=10)

    print(f"Observed ρ_Λ,0: {rho_obs:.3e} J/m³")
    print(f"Evaluations:    {len(points)}")
    print()

    print("Top candidates (closest |log10(ρ_pred/ρ_obs)|):")
    print("-" * 78)
    for i, ev in enumerate(best, start=1):
        p = ev.point.params_overrides
        print(
            f"{i:2d}. log10(ρ_pred/ρ_obs)={ev.log10_ratio:+.2f}  "
            f"ρ_pred={ev.rho_pred_j_m3:.3e} J/m³  "
            f"mu={p['mu_polymer']:.3g} gamma_coeff={p['gamma_coefficient']:.3g} "
            f"alpha={p['alpha_scaling']:.3g} jmax={p['volume_eigenvalue_cutoff']:.0f}"
        )

    min_abs = min(abs(ev.log10_ratio) for ev in best) if best else float("inf")
    print()
    print("No-go style bound over scanned region:")
    print("-" * 78)
    if math.isfinite(min_abs):
        print(f"min |log10(ρ_pred/ρ_obs)| ≈ {min_abs:.2f}")
        print("Interpretation: if this stays \u226b 1, the predictor (in this region)")
        print("cannot match observed ρ_Λ without leaving the bounded scan domain.")
    else:
        print("No finite evaluations produced.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demonstration: LQG-constrained mechanisms.

This demo shows how using LQG first-principles predictions as constraints
affects the free parameters in phenomenological mechanisms.

Key question: Does LQG prediction eliminate tuning in other mechanisms?

Run:
  PYTHONPATH=src python examples/demo_lqg_constrained.py
"""

from ccw.integrations.lqg_constrained import (
    holographic_constrained_by_lqg,
    sequestering_constrained_by_lqg,
)
from ccw.integrations.lqg_predictor import lqg_predictor_available
from ccw.mechanisms import CosmologyBackground


def main() -> None:
    print("=" * 78)
    print("LQG-Constrained Mechanisms: Does LQG Prediction Remove Tuning?")
    print("=" * 78)
    print()

    if not lqg_predictor_available():
        print("⚠ LQG predictor not available.")
        print("This demo requires lqg-cosmological-constant-predictor to be installed.")
        print("Skipping demonstration.")
        return

    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_lambda=0.6889, omega_m=0.3111, omega_r=0.0)

    print("Scenario 1: Holographic Dark Energy constrained by LQG")
    print("-" * 78)
    print()
    print("Question: If LQG predicts ρ_Λ from first principles, what c_factor does")
    print("holographic DE require to match it? If c ~ O(1), no tuning.")
    print()

    result_hde = holographic_constrained_by_lqg(bg)
    print(f"LQG target:       ρ_Λ = {result_hde.lqg_target_rho_j_m3:.3e} J/m³")
    print(f"Best c_factor:    {result_hde.best_fit_params['c_factor']:.2f}")
    print(f"Achieved:         ρ_DE = {result_hde.achieved_rho_j_m3:.3e} J/m³")
    print(f"Residual tuning:  log10(Δρ/ρ) = {result_hde.residual_tuning:.2f}")
    print(f"Success:          {result_hde.success}")
    print()
    print(result_hde.notes)
    print()
    print()

    print("Scenario 2: Sequestering constrained by LQG")
    print("-" * 78)
    print()
    print("Question: If LQG predicts ρ_Λ, how much cancellation f_cancel is required")
    print("to reduce Planck-scale vacuum ρ_vac ~ 10^113 J/m³ to ρ_Λ?")
    print()

    result_seq = sequestering_constrained_by_lqg(bg, rho_vac_j_m3=1e113)
    print(f"LQG target:       ρ_Λ = {result_seq.lqg_target_rho_j_m3:.3e} J/m³")
    print(f"Bare vacuum:      ρ_vac = 1.000e+113 J/m³")
    print(f"Required f_cancel: {result_seq.best_fit_params['f_cancel']:.15f}")
    print(f"Achieved:         ρ_DE = {result_seq.achieved_rho_j_m3:.3e} J/m³")
    print(f"Residual tuning:  ~10^{result_seq.residual_tuning:.0f}")
    print(f"Success:          {result_seq.success}")
    print()
    print(result_seq.notes)
    print()
    print()

    print("=" * 78)
    print("Conclusion:")
    print("=" * 78)
    print()
    print("- Holographic DE: LQG constraint determines c_factor. If c ~ O(1), the")
    print("  combination (LQG + holographic) is 'natural' modulo the LQG prediction itself.")
    print()
    print("- Sequestering: Even with LQG target, extreme fine-tuning remains (~120 orders")
    print("  of magnitude cancellation required). LQG prediction does NOT solve the")
    print("  fine-tuning problem for sequestering-style mechanisms.")
    print()
    print("Net assessment:")
    print("  LQG-as-constraint can make some mechanisms (holographic) 'natural', but")
    print("  it pushes the tuning question into the LQG prediction itself. Whether")
    print("  LQG's Λ is 'natural' depends on whether its inputs (Immirzi, polymer μ₀)")
    print("  are tuned or derived.")


if __name__ == "__main__":
    main()

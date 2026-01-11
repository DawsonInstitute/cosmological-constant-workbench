#!/usr/bin/env python3
"""Demonstration: swampland and holographic constraints in ccw-report.

This example shows how the new constraint checking identifies theoretical
violations (swampland) and phenomenological bounds (holographic) for scalar-field
quintessence mechanisms.

Key insight: The flat-potential limit (λ=0) used in our earlier ΛCDM-reproducing
tests is flagged as a swampland violation, highlighting the tension between
reproducing observed Λ and satisfying string-theory-inspired no-go conditions.
"""

from pathlib import Path

from ccw.report import ReportConfig, generate_report, write_report


def main() -> None:
    print("=== CCW Constraint Demonstration ===\n")

    # Example 1: λ=0 (ΛCDM-like) violates swampland
    print("Example 1: Flat potential (λ=0) — expected swampland violation")
    cfg1 = ReportConfig(
        mechanism="scalar_field_quintessence",
        grid=[
            {
                "potential": "exponential",
                "lam": 0.0,
                "x0": 0.0,
                "z_max": 2.0,
                "n_eval": 300,
            }
        ],
        z_values=(0.0, 1.0, 2.0),
        run_constraints=True,
    )

    report1 = generate_report(cfg1)
    out_dir = Path("examples/constraints_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_report(report1, out_dir / "report_lambda0.json", out_dir / "report_lambda0.md")

    swampland_check = next(
        (c for c in report1["constraints"][0]["checks"] if c["type"] == "swampland"),
        None,
    )
    if swampland_check:
        status = "✓ PASS" if swampland_check["ok"] else "✗ FAIL"
        print(f"  Swampland: {status} — {swampland_check['detail']}\n")

    # Example 2: λ=2.0 (steep potential) satisfies swampland
    print("Example 2: Steep potential (λ=2.0) — expected swampland pass")
    cfg2 = ReportConfig(
        mechanism="scalar_field_quintessence",
        grid=[
            {
                "potential": "exponential",
                "lam": 2.0,
                "x0": 0.0,
                "z_max": 2.0,
                "n_eval": 300,
            }
        ],
        z_values=(0.0, 1.0, 2.0),
        run_constraints=True,
    )

    report2 = generate_report(cfg2)
    write_report(report2, out_dir / "report_lambda2.json", out_dir / "report_lambda2.md")

    swampland_check2 = next(
        (c for c in report2["constraints"][0]["checks"] if c["type"] == "swampland"),
        None,
    )
    if swampland_check2:
        status = "✓ PASS" if swampland_check2["ok"] else "✗ FAIL"
        print(f"  Swampland: {status} — {swampland_check2['detail']}\n")

    # Example 3: Holographic bounds at multiple redshifts
    print("Example 3: Checking holographic bounds across z=[0, 1, 2]")
    holographic_checks = [c for c in report1["constraints"][0]["checks"] if c["type"] == "holographic"]
    for hc in holographic_checks:
        status = "✓ PASS" if hc["ok"] else "✗ FAIL"
        print(f"  z={hc['z']}: {status}")

    print(f"\nReports written to {out_dir}/")
    print("\nKey insight: The ΛCDM-reproducing flat potential (λ=0) is theoretically")
    print("disfavored by swampland conjectures, suggesting that any viable mechanism")
    print("must either violate these bounds (requiring UV completion beyond string theory)")
    print("or accept non-trivial dynamics inconsistent with a pure cosmological constant.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from ccw.report import ReportConfig, generate_report, write_report


def test_report_with_constraints_for_scalar_field(tmp_path: Path) -> None:
    """Test that ccw-report can include constraint checks for scalar-field mechanisms."""

    cfg = ReportConfig(
        mechanism="scalar_field_quintessence",
        grid=[{"potential": "exponential", "lam": 0.0, "x0": 0.0, "z_max": 2.0, "n_eval": 200}],
        z_values=(0.0, 1.0, 2.0),
        run_constraints=True,
    )

    report = generate_report(cfg)

    # Validate structure
    assert "constraints" in report
    assert len(report["constraints"]) == 1

    checks = report["constraints"][0]["checks"]
    assert any(c["type"] == "swampland" for c in checks)
    assert any(c["type"] == "holographic" for c in checks)

    # Write outputs
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    write_report(report, out_json, out_md)

    # Validate file creation
    assert out_json.exists()
    assert out_md.exists()

    # Check JSON contains constraints
    data = json.loads(out_json.read_text())
    assert "constraints" in data


def test_report_constraints_flags_swampland_violation(tmp_path: Path) -> None:
    """Verify that lam=0 is flagged as a swampland violation in the report."""

    cfg = ReportConfig(
        mechanism="scalar_field_quintessence",
        grid=[{"potential": "exponential", "lam": 0.0}],
        z_values=(0.0,),
        run_constraints=True,
        swampland_c_min=2.0 / (3.0**0.5),
    )

    report = generate_report(cfg)
    checks = report["constraints"][0]["checks"]

    swampland_check = next((c for c in checks if c["type"] == "swampland"), None)
    assert swampland_check is not None
    assert not swampland_check["ok"]


def test_report_without_constraints_omits_section(tmp_path: Path) -> None:
    """Verify that constraints are not run when run_constraints=False."""

    cfg = ReportConfig(
        mechanism="cpl_quintessence",
        grid=[{"w0": -1.0, "wa": 0.0}],
        z_values=(0.0,),
        run_constraints=False,
    )

    report = generate_report(cfg)
    assert "constraints" not in report

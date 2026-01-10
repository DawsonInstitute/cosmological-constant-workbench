from __future__ import annotations

from pathlib import Path

from ccw.report import ReportConfig, generate_report, write_report


def test_generate_report_has_expected_keys(tmp_path: Path) -> None:
    cfg = ReportConfig(mechanism="cpl", cutoffs=("electroweak", "planck"), z_values=(0.0, 1.0))
    report = generate_report(cfg)

    assert "observed" in report
    assert "naive_qft" in report
    assert isinstance(report["naive_qft"], list)

    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    write_report(report, out_json, out_md)

    assert out_json.exists()
    assert out_md.exists()
    assert out_md.read_text().startswith("# CCW report")

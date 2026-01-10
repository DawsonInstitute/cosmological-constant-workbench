from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import ReportConfig, generate_report, write_report


def _parse_grid_json(grid_json: str | None) -> list[dict] | None:
    if grid_json is None:
        return None
    obj = json.loads(grid_json)
    if not isinstance(obj, list) or any(not isinstance(x, dict) for x in obj):
        raise ValueError("--grid-json must be a JSON array of objects")
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ccw report: baseline + optional mechanism summary")
    p.add_argument("--h0", type=float, default=67.4)
    p.add_argument("--omega-lambda", type=float, default=0.6889)
    p.add_argument("--omega-m", type=float, default=0.3111)
    p.add_argument("--omega-r", type=float, default=0.0)
    p.add_argument("--omega-k", type=float, default=0.0)

    p.add_argument("--cutoffs", type=str, default="electroweak,1TeV,planck", help="Comma-separated cutoff names")

    p.add_argument("--mechanism", type=str, default=None, help="Optional: cpl | rvm | unimodular | sequestering")
    p.add_argument("--grid-json", type=str, default=None, help="Optional JSON array of mechanism param dicts")
    p.add_argument("--z", type=str, default="0,1,2", help="Comma-separated z values")

    p.add_argument("--out-json", type=str, default="results/report.json")
    p.add_argument("--out-md", type=str, default="results/report.md")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cutoffs = tuple(s.strip() for s in str(args.cutoffs).split(",") if s.strip())
    z_values = tuple(float(s.strip()) for s in str(args.z).split(",") if s.strip())

    cfg = ReportConfig(
        h0_km_s_mpc=args.h0,
        omega_lambda=args.omega_lambda,
        omega_m=args.omega_m,
        omega_r=args.omega_r,
        omega_k=args.omega_k,
        cutoffs=cutoffs,
        mechanism=args.mechanism,
        grid=_parse_grid_json(args.grid_json),
        z_values=z_values,
    )

    report = generate_report(cfg)
    write_report(report, Path(args.out_json), Path(args.out_md))
    print(f"Wrote {args.out_json} and {args.out_md}")

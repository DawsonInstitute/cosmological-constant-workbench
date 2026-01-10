from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .baseline import BaselineInputs, compute_baseline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cosmological Constant Workbench: baseline calculators")
    p.add_argument("--h0", type=float, default=67.4, help="H0 in km/s/Mpc")
    p.add_argument("--omega-lambda", type=float, default=0.6889, help="Ω_Λ")
    p.add_argument(
        "--cutoff",
        type=str,
        default="planck",
        help="Vacuum energy cutoff: planck | electroweak | 1TeV",
    )
    p.add_argument("--dof", type=float, default=1.0, help="Effective massless degrees of freedom")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = compute_baseline(
        BaselineInputs(
            h0_km_s_mpc=args.h0,
            omega_lambda=args.omega_lambda,
            cutoff_name=args.cutoff,
            qft_dof=args.dof,
        )
    )

    payload = asdict(out)
    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=True))

#!/usr/bin/env python3
"""Generate bounded predictor-scan data for the short paper.

Writes:
  papers/lqg_cc_constraints/data/scan_results.tsv

Usage:
  PYTHONPATH=src python papers/lqg_cc_constraints/generate_scan_data.py

Notes:
- Requires lqg-cosmological-constant-predictor to be importable.
- The scan domain is intentionally small/bounded (108 evals) for quick, reproducible evidence.
"""

from __future__ import annotations

import os

from ccw.integrations.lqg_predictor import lqg_predictor_available
from ccw.integrations.lqg_predictor_sweep import evaluate_all_points, make_default_points, write_tsv
from ccw.mechanisms import CosmologyBackground


def main() -> None:
    if not lqg_predictor_available():
        raise RuntimeError(
            "LQG predictor not available. Ensure sibling repo 'lqg-cosmological-constant-predictor' is present "
            "in the multi-root workspace or installed in the environment."
        )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "scan_results.tsv")

    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_lambda=0.6889, omega_m=0.3111, omega_r=0.0)

    points = make_default_points(target_scale_m=1e-15)
    evals = evaluate_all_points(points, bg=bg)
    write_tsv(evals, file_path=out_path)

    # Minimal console summary
    log_abs_min = min(abs(ev.log10_ratio) for ev in evals)
    print(f"Wrote: {out_path}")
    print(f"Evaluations: {len(evals)}")
    print(f"min |log10(ρ_pred/ρ_obs)| = {log_abs_min:.3f}")


if __name__ == "__main__":
    main()

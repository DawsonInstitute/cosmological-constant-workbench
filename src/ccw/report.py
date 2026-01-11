from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseline import BaselineInputs, BaselineOutputs, compute_baseline
from .constraints import check_scalar_field_swampland, holographic_bound_from_hz
from .frw import h_z_lcdm_s_inv, h_z_from_rho_de_s_inv
from .mechanisms import CosmologyBackground, ScalarFieldQuintessence
from .sweep import SweepRow, evaluate_mechanism, run_sweep


@dataclass(frozen=True)
class ReportConfig:
    h0_km_s_mpc: float = 67.4
    omega_lambda: float = 0.6889
    omega_m: float = 0.3111
    omega_r: float = 0.0
    omega_k: float = 0.0
    cutoffs: tuple[str, ...] = ("electroweak", "1TeV", "planck")

    mechanism: Optional[str] = None
    grid: Optional[List[Dict[str, Any]]] = None
    z_values: tuple[float, ...] = (0.0, 1.0, 2.0)

    # Constraint checks
    run_constraints: bool = False
    swampland_c_min: float = 2.0 / (3.0**0.5)
    holographic_c_factor: float = 1.0


def _markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _stringify_params(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        r2 = dict(r)
        if isinstance(r2.get("params"), dict):
            r2["params"] = json.dumps(r2["params"], sort_keys=True)
        out.append(r2)
    return out


def generate_report(cfg: ReportConfig) -> Dict[str, Any]:
    bg = CosmologyBackground(
        h0_km_s_mpc=cfg.h0_km_s_mpc,
        omega_lambda=cfg.omega_lambda,
        omega_m=cfg.omega_m,
        omega_r=cfg.omega_r,
        omega_k=cfg.omega_k,
    )

    baselines: List[Dict[str, Any]] = []
    for c in cfg.cutoffs:
        out: BaselineOutputs = compute_baseline(
            BaselineInputs(
                h0_km_s_mpc=cfg.h0_km_s_mpc,
                omega_lambda=cfg.omega_lambda,
                cutoff_name=c,
            )
        )
        baselines.append({"cutoff": c, **out.naive_qft})

    report: Dict[str, Any] = {
        "inputs": asdict(cfg),
        "observed": {
            "h0_km_s_mpc": cfg.h0_km_s_mpc,
            "omega_lambda": cfg.omega_lambda,
            "rho_lambda0_j_m3": bg.rho_lambda0_j_m3,
        },
        "naive_qft": baselines,
    }

    if cfg.mechanism:
        grid = cfg.grid if cfg.grid is not None else [{}]
        rows: List[SweepRow] = run_sweep(mechanism=cfg.mechanism, grid=grid, z_values=cfg.z_values, bg=bg)
        report["mechanism"] = {
            "name": cfg.mechanism,
            "z_values": list(cfg.z_values),
            "grid": grid,
            "rows": [asdict(r) for r in rows],
        }

        if cfg.run_constraints:
            constraints_results: List[Dict[str, Any]] = []
            for params in grid:
                constraint_entry: Dict[str, Any] = {"params": params, "checks": []}

                # Check swampland for scalar-field mechanisms
                if cfg.mechanism.lower() in {"scalar_field", "scalar_field_quintessence"}:
                    try:
                        # Reconstruct mechanism instance
                        mech = ScalarFieldQuintessence(
                            potential=params.get("potential", "exponential"),
                            lam=params.get("lam", 0.0),
                            alpha=params.get("alpha", 1.0),
                            phi0=params.get("phi0", 1.0),
                            x0=params.get("x0", 0.0),
                            z_max=max(cfg.z_values) if cfg.z_values else 5.0,
                            n_eval=400,
                        )
                        chk = check_scalar_field_swampland(mech, bg, z_values=cfg.z_values, c_min=cfg.swampland_c_min)
                        constraint_entry["checks"].append({"type": "swampland", "ok": chk.ok, "detail": chk.detail})
                    except Exception as e:
                        constraint_entry["checks"].append({"type": "swampland", "ok": False, "detail": f"Error: {e}"})

                # Check holographic bounds for all mechanisms
                for z in cfg.z_values:
                    try:
                        row_data = evaluate_mechanism(cfg.mechanism, params, z, bg)
                        rho_de = row_data.rho_de_j_m3

                        def hz(zz: float) -> float:
                            return h_z_from_rho_de_s_inv(zz, bg, rho_de)

                        chk_holo = holographic_bound_from_hz(
                            z, hz_s_inv=hz, rho_de_j_m3=rho_de, c_factor=cfg.holographic_c_factor
                        )
                        constraint_entry["checks"].append(
                            {"type": "holographic", "z": z, "ok": chk_holo.ok, "detail": chk_holo.detail}
                        )
                    except Exception as e:
                        constraint_entry["checks"].append({"type": "holographic", "z": z, "ok": False, "detail": f"Error: {e}"})

                constraints_results.append(constraint_entry)
            report["constraints"] = constraints_results

    return report


def write_report(report: Dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    md_lines: List[str] = []
    md_lines.append("# CCW report")
    md_lines.append("")
    md_lines.append("## Observed")
    md_lines.append(f"- H0: {report['observed']['h0_km_s_mpc']} km/s/Mpc")
    md_lines.append(f"- ΩΛ: {report['observed']['omega_lambda']}")
    md_lines.append(f"- ρΛ,0: {report['observed']['rho_lambda0_j_m3']:.3e} J/m³")
    md_lines.append("")

    md_lines.append("## Naive QFT cutoffs")
    table_rows = [
        {
            "cutoff": r.get("cutoff"),
            "E_cut (J)": f"{r.get('e_cutoff_joule', 0.0):.3e}",
            "rho_naive (J/m³)": f"{r.get('rho_naive_j_m3', 0.0):.3e}",
            "ratio": f"{r.get('rho_naive_over_rho_lambda', 0.0):.3e}",
        }
        for r in report.get("naive_qft", [])
    ]
    md_lines.append(_markdown_table(table_rows, ["cutoff", "E_cut (J)", "rho_naive (J/m³)", "ratio"]))
    md_lines.append("")

    if "mechanism" in report:
        md_lines.append("## Mechanism")
        md_lines.append(f"- name: {report['mechanism']['name']}")
        md_lines.append("")
        md_lines.append(
            _markdown_table(
                _stringify_params(report["mechanism"]["rows"]),
                ["mechanism", "z", "rho_de_j_m3", "w_de", "params"],
            )
        )

    if "constraints" in report:
        md_lines.append("")
        md_lines.append("## Constraints")
        for entry in report["constraints"]:
            md_lines.append(f"**Params**: {json.dumps(entry['params'], sort_keys=True)}")
            md_lines.append("")
            for chk in entry["checks"]:
                status = "✓ PASS" if chk.get("ok") else "✗ FAIL"
                z_info = f" (z={chk['z']})" if "z" in chk else ""
                md_lines.append(f"- {chk['type']}{z_info}: {status} — {chk['detail']}")
            md_lines.append("")

    out_md.write_text("\n".join(md_lines) + "\n")

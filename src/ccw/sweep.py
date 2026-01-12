from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .baseline import BaselineInputs, compute_baseline
from .mechanisms import (
    CPLQuintessence,
    CosmologyBackground,
    HolographicDarkEnergy,
    RunningVacuumRVM,
    ScalarFieldQuintessence,
    SequesteringToy,
    SUSYBreaking,
    UnimodularBookkeeping,
)


@dataclass(frozen=True)
class SweepRow:
    mechanism: str
    params: Dict[str, Any]
    z: float
    rho_de_j_m3: float
    w_de: Optional[float]


def evaluate_mechanism(name: str, params: Dict[str, Any], z: float, bg: CosmologyBackground) -> SweepRow:
    key = name.strip().lower()

    if key in {"cpl", "cpl_quintessence"}:
        mech = CPLQuintessence(w0=float(params.get("w0", -1.0)), wa=float(params.get("wa", 0.0)))
    elif key in {"rvm", "running_vacuum", "running_vacuum_rvm"}:
        mech = RunningVacuumRVM(nu=float(params.get("nu", 0.0)))
    elif key in {"scalar_field", "scalar_field_quintessence"}:
        mech = ScalarFieldQuintessence(
            potential=params.get("potential", "exponential"),
            lam=float(params.get("lam", 0.0)),
            alpha=float(params.get("alpha", 1.0)),
            phi0=float(params.get("phi0", 1.0)),
            x0=float(params.get("x0", 0.0)),
            z_max=float(params.get("z_max", 5.0)),
            n_eval=int(params.get("n_eval", 400)),
        )
    elif key in {"susy", "susy_breaking"}:
        mech = SUSYBreaking(
            m_susy_gev=float(params.get("m_susy_gev", 1e3)),
            loop_factor=float(params.get("loop_factor", 1.0 / (16.0 * 3.14159**2))),
            log_enhancement=bool(params.get("log_enhancement", True)),
        )
    elif key in {"holographic", "holographic_dark_energy", "hde"}:
        mech = HolographicDarkEnergy(
            cutoff_type=params.get("cutoff_type", "hubble"),
            c_factor=float(params.get("c_factor", 1.0)),
            background_h0=float(params.get("background_h0", 67.4)),
            background_omega_m=float(params.get("background_omega_m", 0.3)),
        )
    elif key in {"unimodular", "unimodular_bookkeeping"}:
        mech = UnimodularBookkeeping(
            lambda_bare_m_minus2=float(params.get("lambda_bare_m_minus2", 1e-52)),
            rho_vac_quantum_j_m3=float(params.get("rho_vac_quantum_j_m3", 1e113)),
            alpha_grav=float(params.get("alpha_grav", 0.0)),
        )
    elif key in {"sequestering", "sequestering_toy"}:
        mech = SequesteringToy(
            rho_vac_j_m3=float(params.get("rho_vac_j_m3", 1e113)),
            rho_pt_j_m3=float(params.get("rho_pt_j_m3", 1e80)),
            f_cancel=float(params.get("f_cancel", 1e-120)),
        )
    else:
        raise ValueError(f"Unknown mechanism: {name}")

    out = mech.evaluate(z=z, bg=bg).result
    return SweepRow(mechanism=mech.name, params=params, z=z, rho_de_j_m3=out.rho_de_j_m3, w_de=out.w_de)


def run_sweep(
    *,
    mechanism: str,
    grid: Iterable[Dict[str, Any]],
    z_values: Iterable[float],
    bg: CosmologyBackground,
) -> List[SweepRow]:
    rows: List[SweepRow] = []
    for params in grid:
        for z in z_values:
            rows.append(evaluate_mechanism(mechanism, params, float(z), bg))
    return rows


def write_json(path: Path, rows: List[SweepRow]) -> None:
    payload = [asdict(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, rows: List[SweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mechanism", "z", "rho_de_j_m3", "w_de", "params_json"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "mechanism": r.mechanism,
                    "z": r.z,
                    "rho_de_j_m3": r.rho_de_j_m3,
                    "w_de": r.w_de,
                    "params_json": json.dumps(r.params, sort_keys=True),
                }
            )

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .cosmology import observed_lambda_from_h0_omega
from .qft_vacuum import VacuumCutoff, cutoff_from_name, naive_vacuum_energy_density_massless_dof


@dataclass(frozen=True)
class BaselineInputs:
    h0_km_s_mpc: float = 67.4
    omega_lambda: float = 0.6889
    cutoff: Optional[VacuumCutoff] = None
    cutoff_name: str = "planck"
    qft_dof: float = 1.0


@dataclass(frozen=True)
class BaselineOutputs:
    observed: Dict[str, Any]
    naive_qft: Dict[str, Any]


def compute_baseline(inputs: BaselineInputs) -> BaselineOutputs:
    cutoff = inputs.cutoff or cutoff_from_name(inputs.cutoff_name)

    obs = observed_lambda_from_h0_omega(
        h0_km_s_mpc=inputs.h0_km_s_mpc,
        omega_lambda=inputs.omega_lambda,
    )

    rho_naive = naive_vacuum_energy_density_massless_dof(
        e_cutoff_joule=cutoff.e_cutoff_joule,
        dof=inputs.qft_dof,
    )

    ratio = rho_naive / obs.rho_lambda_j_m3 if obs.rho_lambda_j_m3 != 0 else float("inf")

    return BaselineOutputs(
        observed=asdict(obs),
        naive_qft={
            "cutoff_name": cutoff.name,
            "e_cutoff_joule": cutoff.e_cutoff_joule,
            "dof": inputs.qft_dof,
            "rho_naive_j_m3": rho_naive,
            "rho_naive_over_rho_lambda": ratio,
        },
    )

from __future__ import annotations

from dataclasses import dataclass

from .constants import C_M_S, GEV_J, HBAR_J_S, PI, TEV_J


@dataclass(frozen=True)
class VacuumCutoff:
    name: str
    e_cutoff_joule: float


def cutoff_from_name(name: str) -> VacuumCutoff:
    key = name.strip().lower()
    if key in {"planck", "mplanck", "eplanck"}:
        # Planck energy ~ 1.22e19 GeV
        return VacuumCutoff("planck", 1.2209e19 * GEV_J)
    if key in {"electroweak", "ew"}:
        # O(100 GeV)
        return VacuumCutoff("electroweak", 100.0 * GEV_J)
    if key in {"1tev", "tev"}:
        return VacuumCutoff("1TeV", 1.0 * TEV_J)

    raise ValueError(f"Unknown cutoff name: {name!r}")


def naive_vacuum_energy_density_massless_dof(e_cutoff_joule: float, dof: float = 1.0) -> float:
    """Naive zero-point energy density with a sharp energy cutoff.

    For a massless field with a sharp momentum cutoff, one convenient form is

      ρ ≈ dof * E_cut^4 / (16 π^2 ħ^3 c^3)

    This is *not* a physical prediction of QFT+GR; it is the scaling estimate
    used to demonstrate the magnitude of the cosmological constant problem.
    """

    if e_cutoff_joule <= 0:
        raise ValueError("Cutoff energy must be positive")
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive")

    return dof * (e_cutoff_joule**4) / (16.0 * (PI**2) * (HBAR_J_S**3) * (C_M_S**3))

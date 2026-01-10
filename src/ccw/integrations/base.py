from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class OptionalIntegrationResult:
    available: bool
    detail: str
    payload: Optional[Any] = None


def try_import_lqg_cosmological_constant_predictor() -> OptionalIntegrationResult:
    """Guarded import of the sibling repo `lqg-cosmological-constant-predictor`.

    This workbench treats it as an optional external model source. We do not
    vendor or depend on it at install time.
    """

    try:
        import cosmological_constant_predictor as lqg_ccp  # type: ignore

        return OptionalIntegrationResult(
            available=True,
            detail="Imported cosmological_constant_predictor from lqg-cosmological-constant-predictor",
            payload=lqg_ccp,
        )
    except Exception as e:  # noqa: BLE001
        return OptionalIntegrationResult(available=False, detail=f"Not available: {e}")


def try_import_coherence_gravity_coupling() -> OptionalIntegrationResult:
    """Guarded import of the sibling repo `coherence-gravity-coupling`.

    Placeholder hook for later, once that repo exposes a stable API that maps to
    an effective w(z) or stress-energy.
    """

    try:
        import coherence_gravity_coupling  # type: ignore

        return OptionalIntegrationResult(
            available=True,
            detail="Imported coherence_gravity_coupling",
            payload=coherence_gravity_coupling,
        )
    except Exception as e:  # noqa: BLE001
        return OptionalIntegrationResult(available=False, detail=f"Not available: {e}")


def lqg_predictor_available() -> bool:
    """Check if lqg-cosmological-constant-predictor is available for comparison adapter."""
    return try_import_lqg_cosmological_constant_predictor().available

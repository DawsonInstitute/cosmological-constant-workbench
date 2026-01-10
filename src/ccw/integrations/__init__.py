"""Integrations package for optional external adapters."""

from __future__ import annotations

from .base import (
    OptionalIntegrationResult,
    lqg_predictor_available,
    try_import_coherence_gravity_coupling,
    try_import_lqg_cosmological_constant_predictor,
)

__all__ = [
    "OptionalIntegrationResult",
    "lqg_predictor_available",
    "try_import_lqg_cosmological_constant_predictor",
    "try_import_coherence_gravity_coupling",
]

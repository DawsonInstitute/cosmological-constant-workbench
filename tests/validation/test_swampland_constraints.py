from __future__ import annotations

from ccw.constraints import (
    check_scalar_field_swampland,
    swampland_ds_gradient_bound_exponential,
    swampland_ds_gradient_bound_inverse_power,
)
from ccw.mechanisms import CosmologyBackground, ScalarFieldQuintessence


def test_swampland_exponential_flags_small_lambda() -> None:
    # With c_min ~ 1.1547, lambda=0 should fail.
    chk = swampland_ds_gradient_bound_exponential(0.0)
    assert not chk.ok


def test_swampland_exponential_passes_large_lambda() -> None:
    chk = swampland_ds_gradient_bound_exponential(2.0)
    assert chk.ok


def test_swampland_inverse_power_checks_phi_range() -> None:
    # alpha/phi must exceed c_min over the range; large phi should fail.
    chk = swampland_ds_gradient_bound_inverse_power(alpha=1.0, phi_values=[1.0, 10.0])
    assert not chk.ok


def test_check_scalar_field_swampland_exponential_uses_mech_lambda() -> None:
    bg = CosmologyBackground()
    mech = ScalarFieldQuintessence(potential="exponential", lam=0.0, x0=0.0, z_max=2.0, n_eval=200)
    chk = check_scalar_field_swampland(mech, bg, z_values=[0.0, 1.0, 2.0])
    assert not chk.ok

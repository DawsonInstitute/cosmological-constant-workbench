from __future__ import annotations

import math

import pytest

from ccw.mechanisms import (
    CPLQuintessence,
    CosmologyBackground,
    RunningVacuumRVM,
    ScalarFieldQuintessence,
    SequesteringToy,
    UnimodularBookkeeping,
)


@pytest.mark.parametrize(
    "mech",
    [
        CPLQuintessence(w0=-1.0, wa=0.0),
        RunningVacuumRVM(nu=0.0),
        ScalarFieldQuintessence(potential="exponential", lam=0.0, x0=0.0, z_max=5.0, n_eval=400),
        UnimodularBookkeeping(lambda_bare_m_minus2=1e-52, rho_vac_quantum_j_m3=1e113, alpha_grav=0.0),
        SequesteringToy(rho_vac_j_m3=1e113, rho_pt_j_m3=1e80, f_cancel=5e-124),
    ],
)
def test_default_mechanisms_are_finite_and_nonnegative_over_grid(mech) -> None:
    bg = CosmologyBackground()
    for z in [0.0, 0.5, 1.0, 2.0, 5.0]:
        out = mech.evaluate(z, bg).result
        assert math.isfinite(out.rho_de_j_m3)
        assert out.rho_de_j_m3 >= 0.0
        if out.w_de is not None:
            assert math.isfinite(out.w_de)
            # Loose physical-ish bounds for default stubs
            assert -5.0 < out.w_de < 2.0

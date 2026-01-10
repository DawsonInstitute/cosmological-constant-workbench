from __future__ import annotations

import math

from ccw.mechanisms import CosmologyBackground, ScalarFieldQuintessence


def test_scalar_field_lambda_zero_behaves_like_constant_lambda() -> None:
    bg = CosmologyBackground()
    mech = ScalarFieldQuintessence(potential="exponential", lam=0.0, x0=0.0, z_max=5.0, n_eval=600)

    rho0 = bg.rho_lambda0_j_m3
    for z in [0.0, 0.5, 1.0, 2.0, 5.0]:
        out = mech.evaluate(z, bg).result
        assert math.isfinite(out.rho_de_j_m3)
        # For λ=0 and x0=0, the field sits on a perfectly flat potential: u_DE stays constant.
        assert abs(out.rho_de_j_m3 - rho0) / rho0 < 5e-3
        assert out.w_de is not None
        assert abs(out.w_de + 1.0) < 5e-3

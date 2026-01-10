from __future__ import annotations

import math

from ccw.mechanisms import CPLQuintessence, CosmologyBackground, RunningVacuumRVM, SequesteringToy, UnimodularBookkeeping


def test_cpl_reduces_to_lambda_when_w0_minus1_wa0() -> None:
    bg = CosmologyBackground()
    m = CPLQuintessence(w0=-1.0, wa=0.0)

    r0 = m.evaluate(0.0, bg).result.rho_de_j_m3
    r2 = m.evaluate(2.0, bg).result.rho_de_j_m3

    assert math.isfinite(r0)
    assert math.isfinite(r2)
    assert abs(r0 - bg.rho_lambda0_j_m3) / bg.rho_lambda0_j_m3 < 1e-12
    assert abs(r2 - bg.rho_lambda0_j_m3) / bg.rho_lambda0_j_m3 < 1e-12


def test_unimodular_explicit_bookkeeping() -> None:
    bg = CosmologyBackground()
    # Default: lambda_bare ~ 1e-52, alpha_grav=0 => rho_eff = lambda_bare c^2/(8piG)
    m = UnimodularBookkeeping()
    rho = m.evaluate(0.0, bg).result.rho_de_j_m3
    assert math.isfinite(rho)
    assert rho > 0
    # Same at all z (constant w=-1)
    assert abs(m.evaluate(5.0, bg).result.rho_de_j_m3 - rho) / rho < 1e-12


def test_sequestering_explicit_cancellation() -> None:
    bg = CosmologyBackground()
    # Default: rho_vac=1e113, rho_PT=1e80, f_cancel=1e-120 => residual ~ (1e113+1e80)*1e-120
    m = SequesteringToy()
    out = m.evaluate(1.0, bg).result
    expected = (m.rho_vac_j_m3 + m.rho_pt_j_m3) * m.f_cancel
    assert math.isfinite(out.rho_de_j_m3)
    assert abs(out.rho_de_j_m3 - expected) < 1e-18


def test_running_vacuum_nu0_is_reasonable_at_z0() -> None:
    bg = CosmologyBackground()
    m = RunningVacuumRVM(nu=0.0)
    out0 = m.evaluate(0.0, bg).result

    assert math.isfinite(out0.rho_de_j_m3)
    assert out0.rho_de_j_m3 > 0

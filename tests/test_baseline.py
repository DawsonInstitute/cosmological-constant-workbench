from __future__ import annotations

from ccw.baseline import BaselineInputs, compute_baseline


def test_observed_lambda_reasonable_magnitude() -> None:
    out = compute_baseline(BaselineInputs(h0_km_s_mpc=67.4, omega_lambda=0.6889, cutoff_name="electroweak"))

    rho_lambda = out.observed["rho_lambda_j_m3"]
    lam = out.observed["lambda_m_inv2"]

    # Order-of-magnitude checks (avoid false precision)
    assert 1e-10 < rho_lambda < 1e-8
    assert 1e-53 < lam < 1e-51


def test_naive_qft_cutoff_scaling_is_huge() -> None:
    out = compute_baseline(BaselineInputs(cutoff_name="planck"))
    ratio = out.naive_qft["rho_naive_over_rho_lambda"]

    # The textbook discrepancy is ~1e120–1e123 depending on details.
    assert ratio > 1e110

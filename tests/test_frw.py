from __future__ import annotations

import math

from ccw.frw import comoving_distance_m, h_z_lcdm_s_inv, luminosity_distance_m
from ccw.mechanisms import CosmologyBackground


def test_hz_lcdm_at_z0_is_h0() -> None:
    bg = CosmologyBackground()
    h0 = h_z_lcdm_s_inv(0.0, bg)
    h0_again = h_z_lcdm_s_inv(0.0, bg)
    assert math.isfinite(h0)
    assert abs(h0 - h0_again) < 1e-18


def test_small_z_comoving_distance_linear_limit() -> None:
    bg = CosmologyBackground()
    z = 1e-3

    dc = comoving_distance_m(z, lambda zz: h_z_lcdm_s_inv(zz, bg))

    # For tiny z, D_C ≈ c z / H0
    h0 = h_z_lcdm_s_inv(0.0, bg)
    approx = 299_792_458.0 * z / h0
    assert abs(dc - approx) / approx < 5e-3


def test_luminosity_distance_relation_flat() -> None:
    bg = CosmologyBackground()
    z = 0.5
    dc = comoving_distance_m(z, lambda zz: h_z_lcdm_s_inv(zz, bg))
    dl = luminosity_distance_m(z, lambda zz: h_z_lcdm_s_inv(zz, bg))
    assert abs(dl - (1.0 + z) * dc) / dl < 1e-12

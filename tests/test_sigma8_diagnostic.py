import numpy as np

from ccw.frw import h_z_lcdm_s_inv
from ccw.mechanisms import CosmologyBackground
from ccw.sigma8_diagnostic import (
    growth_factor_D,
    growth_rate_f,
    sigma8_z,
    f_sigma8_z,
    fsigma8_chi_squared,
    get_boss_dr12_fsigma8_observables,
)


def test_growth_factor_normalized_to_one_today():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    d0 = growth_factor_D(0.0, hz_s_inv=hz, omega_m0=0.3)
    assert np.isfinite(d0)
    assert abs(d0 - 1.0) < 1e-3


def test_growth_factor_decreases_with_redshift():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    d_z0 = growth_factor_D(0.0, hz_s_inv=hz, omega_m0=0.3)
    d_z1 = growth_factor_D(1.0, hz_s_inv=hz, omega_m0=0.3)

    assert d_z1 < d_z0
    assert d_z1 > 0


def test_growth_rate_reasonable_range():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    f0 = growth_rate_f(0.0, hz_s_inv=hz, omega_m0=0.3)
    assert 0.2 < f0 < 1.5


def test_sigma8_and_fsigma8_finite_and_plausible():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    s8_0 = 0.811

    s8_z05 = sigma8_z(0.5, sigma8_0=s8_0, hz_s_inv=hz, omega_m0=0.3)
    fs8_z05 = f_sigma8_z(0.5, sigma8_0=s8_0, hz_s_inv=hz, omega_m0=0.3)

    assert np.isfinite(s8_z05)
    assert np.isfinite(fs8_z05)

    assert 0 < s8_z05 < s8_0
    assert 0.0 < fs8_z05 < 1.5


def test_fsigma8_chi_squared_nonnegative():
    bg = CosmologyBackground(h0_km_s_mpc=67.4, omega_m=0.3)

    def hz(z: float) -> float:
        return h_z_lcdm_s_inv(z, bg)

    obs = get_boss_dr12_fsigma8_observables()
    chi2 = fsigma8_chi_squared(obs, sigma8_0=0.811, hz_s_inv=hz, omega_m0=0.3)

    assert np.isfinite(chi2)
    assert chi2 >= 0

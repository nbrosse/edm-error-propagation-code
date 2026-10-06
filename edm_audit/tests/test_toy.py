"""Checks of the one-dimensional audit listed under "Software checks" in appendix J."""

import decimal
import math

import numpy as np
import pytest

from edm_audit.common import edm_coeffs, edm_sigmas
from edm_audit.run_oracle_refinement import nested_oracle_rows, refinement_orders
from edm_audit.run_toy_controls import (
    CONFIG as CONTROLS, euler_states, kappa, oracle_derivative_bound, relative_mesh, starting_noise_rows,
    synchronous_rows,
)
from edm_audit.toy import (
    B_D1, Mixture1D, accumulate, audit_grid, decade_sums, defect_shape, exact_oracle_rows,
    fixed_noise_rows, high_noise_threshold, local_defect_bound, midpoint_levels, misspecified,
    mlp_denoiser, perturbed_oracle, recursion_decomposition, spatial_envelope, step_metrics, train_mlp,
)


MIX = Mixture1D.standardized([0.15, 0.35, 0.35, 0.15], [-3.0, -1.0, 1.0, 3.0], 0.04, sigma_data=0.5)
WRONG = misspecified(MIX, [0.2, 0.3, 0.3, 0.2], 1.02, 0.7)
SPATIAL = {"radius_sd": 12.0, "points": 4097, "check_radius_sd": 16.0, "check_points": 8193}


def one_step(t, s, denoiser, sigma=0.3, sigma_next=0.2, spatial=SPATIAL):
    return step_metrics(t, s, denoiser, MIX, sigma, sigma_next, 0.5,
                        MIX.quantiles(midpoint_levels(len(t)), sigma_next), spatial)[0]


# --- the law ---------------------------------------------------------------------------------

def test_grids_are_nested():
    np.testing.assert_array_equal(edm_sigmas(17), edm_sigmas(34)[::2])
    np.testing.assert_allclose(edm_sigmas(17)[[0, -1]], [80.0, 0.002], rtol=1e-12)


def test_mixture_is_standardized():
    assert abs(MIX.mean()) < 1e-14 and abs(MIX.variance() - 0.25) < 1e-14


@pytest.mark.parametrize("sigma", [0.002, 0.05, 1.0, 80.0])
def test_quantiles_invert_the_cdf(sigma):
    u = midpoint_levels(1000)
    np.testing.assert_allclose(MIX.cdf(MIX.quantiles(u, sigma), sigma), u, atol=1e-12)


@pytest.mark.parametrize("sigma", [0.01, 0.3, 5.0])
def test_posterior_mean_is_tweedie(sigma):
    y, h = np.linspace(-1.0, 1.0, 41), 1e-5 * np.hypot(MIX.width, sigma)
    D, dD = MIX.posterior_mean(y, sigma)
    score = (np.log(MIX.pdf(y + h, sigma)) - np.log(MIX.pdf(y - h, sigma))) / (2 * h)
    np.testing.assert_allclose(D, y + sigma ** 2 * score, rtol=1e-6, atol=1e-8)
    fd = (MIX.posterior_mean(y + h, sigma)[0] - MIX.posterior_mean(y - h, sigma)[0]) / (2 * h)
    np.testing.assert_allclose(dD, fd, rtol=1e-5, atol=1e-6)


# --- one step --------------------------------------------------------------------------------

def test_secant_identity():
    """sigma E = <t - s, D(t) - D(s)> / e^2 - 1 and sigma Q = ||t - s - (D(t) - D(s))|| / e."""
    u = midpoint_levels(2048)
    t, s = MIX.quantiles(u, 0.3), WRONG.quantiles(u, 0.3)
    row = one_step(t, s, MIX.posterior_mean)
    dD = MIX.posterior_mean(t, 0.3)[0] - MIX.posterior_mean(s, 0.3)[0]
    e2 = np.mean((t - s) ** 2)
    np.testing.assert_allclose(0.3 * row["E_OT"], np.mean((t - s) * dD) / e2 - 1.0, rtol=1e-10)
    np.testing.assert_allclose(0.3 * row["Q_OT"], np.sqrt(np.mean((t - s - dD) ** 2) / e2), rtol=1e-10)


def test_law_directional_field_ordering():
    """log max(1, gamma) <= directional term <= field term, step by step and accumulated."""
    rows = audit_grid({"oracle": MIX.posterior_mean, "misspec": WRONG.posterior_mean}, MIX, edm_sigmas(17), 4096, 0.5, SPATIAL)
    for r in rows:
        assert math.log(max(1.0, r["gamma"])) <= r["directional_term"] + 1e-10
        assert r["directional_term"] <= r["field_term_pts"] + 1e-8
    for arm in ("oracle", "misspec"):
        acc = accumulate([r for r in rows if r["arm"] == arm], sigma_hi=1.0)
        assert acc["Lambda"] <= acc["Lambda_dir"] + 1e-10 <= acc["Lambda_field_pts"] + 2e-8


def test_transported_coupling_equals_gamma_for_an_increasing_predictor():
    u = midpoint_levels(2048)
    t, s = MIX.quantiles(u, 0.3), WRONG.quantiles(u, 0.3)
    for denoiser in (MIX.posterior_mean, WRONG.posterior_mean, perturbed_oracle(MIX, 0.1, 0.5)):
        row = one_step(t, s, denoiser, spatial=None)
        assert row["A_OT"] == pytest.approx(row["gamma"], rel=1e-12)


def test_a_non_monotone_predictor_gives_gamma_below_A_OT():
    t = MIX.quantiles(midpoint_levels(512), 0.3)
    row = one_step(t, t + 0.05, lambda y, sigma: (-(y ** 2), -2.0 * y), spatial=None)
    assert row["gamma"] < row["A_OT"] - 1e-6


def test_gamma_is_one_when_the_laws_coincide():
    t = MIX.quantiles(midpoint_levels(512), 0.1)
    row = one_step(t, t.copy(), MIX.posterior_mean, 0.1, 0.08, spatial=None)
    assert (row["e"], row["gamma"]) == (0.0, 1.0)


def test_spatial_interval_is_K_independent():
    t = MIX.quantiles(midpoint_levels(512), 0.3)
    a, b = one_step(t, t + 1e-3, MIX.posterior_mean), one_step(t, t + 2e-3, MIX.posterior_mean)
    for key in ("spatial_left", "spatial_right", "Los_pts", "Lv_pts"):
        assert a[key] == pytest.approx(b[key])
    assert math.isfinite(a["field_pts_rel_change"])


# --- the recursion ---------------------------------------------------------------------------

def test_one_step_recursion_and_its_unrolled_bound_hold():
    rows = audit_grid({"oracle": MIX.posterior_mean, "misspec": WRONG.posterior_mean}, MIX, edm_sigmas(17), 1024, 0.5)
    for arm in ("oracle", "misspec"):
        run = [r for r in rows if r["arm"] == arm]
        assert min(r["recursion_slack"] for r in run) >= -1e-12
        total = recursion_decomposition(run)
        assert total["recursion_bound"] >= total["final_error"] - 1e-12
        if arm == "oracle":
            assert total["propagated_learning"] == 0.0


def test_exact_start_leaves_only_discretization():
    total = recursion_decomposition(exact_oracle_rows(MIX, edm_sigmas(17), 2048))
    assert total["propagated_initialization"] == 0.0 and total["propagated_learning"] == 0.0
    assert 0.0 < total["propagated_discretization"] and total["final_error"] <= total["recursion_bound"] + 1e-12


def test_universal_defect_bound_dominates_the_measured_defect():
    assert B_D1 == 16.0
    for row in exact_oracle_rows(MIX, edm_sigmas(17), 2048):
        assert row["Delta_j"] >= row["discretization_term"]
        assert row["Delta_j"] == pytest.approx(local_defect_bound(row["sigma"], row["a"]))


def test_defect_shape_is_accurate_at_small_steps():
    decimal.getcontext().prec = 60
    for a in (1e-12, 1e-8, 1e-5, 1e-4, 1e-3, 0.3, 0.9):
        d = decimal.Decimal(a)
        exact = float(d + (1 - d) * (1 - d).ln())
        assert defect_shape(a) == pytest.approx(exact, rel=1e-12)


def test_perturbed_oracle_residual_is_bounded_by_eps():
    """The EDM-normalized residual |D_hat - D| / c_out is at most eps at every level."""
    for eps in (0.003, 0.01, 0.1):
        denoiser = perturbed_oracle(MIX, eps, 0.5)
        for sigma in (0.002, 0.05, 1.0, 80.0):
            y = MIX.quantiles(midpoint_levels(1024), sigma)
            residual = (denoiser(y, sigma)[0] - MIX.posterior_mean(y, sigma)[0]) / edm_coeffs(sigma, 0.5)[1]
            assert 0.0 < np.sqrt(np.mean(residual ** 2)) <= eps + 1e-15


def test_trained_denoiser_derivative_matches_finite_differences():
    net = train_mlp(MIX, 0, sigma_data=0.5, hidden=16, depth=2, steps=20, batch_size=128,
                    lr=1e-3, ema=0.9, grad_clip=1.0, P_mean=-1.2, P_std=1.2)
    denoise, y, h = mlp_denoiser(net), np.linspace(-1.0, 1.0, 11), 1e-6
    fd = (denoise(y + h, 0.3)[0] - denoise(y - h, 0.3)[0]) / (2 * h)
    np.testing.assert_allclose(denoise(y, 0.3)[1], fd, rtol=1e-6, atol=1e-8)


# --- accumulation ----------------------------------------------------------------------------

@pytest.mark.parametrize("b_hi", [0.5, 0.99])
def test_threshold_is_the_bottom_of_the_top_block(b_hi):
    small = {**SPATIAL, "points": 4097}
    levels = np.geomspace(80.0, 0.002, 60)
    sigma_hi = high_noise_threshold(MIX, MIX.posterior_mean, 0.002, 80.0, 0.5, b_hi, small, n=60)
    margin = lambda s: spatial_envelope(MIX, MIX.posterior_mean, s, 0.5, 12.0, 4097)["b_pts"]
    assert all(margin(s) >= b_hi for s in levels if s >= sigma_hi)
    below = [s for s in levels if s < sigma_hi]
    assert not below or margin(below[0]) < b_hi


def test_resummation_decades_and_fixed_noise_rows():
    rows = audit_grid({"oracle": MIX.posterior_mean}, MIX, edm_sigmas(68), 512, 0.5, SPATIAL)
    for threshold in (0.05, 1.0):
        acc = accumulate(rows, threshold)
        low = [r for r in rows if r["sigma_next"] < threshold]
        assert acc["n_low"] == len(low)
        assert acc["Lambda"] == pytest.approx(sum(math.log(max(1.0, r["gamma"])) for r in low), abs=1e-15)
        assert sum(decade_sums(rows, threshold).values()) == pytest.approx(acc["Lambda"])
    for entry in fixed_noise_rows(rows, [0.04, 0.15]):
        best = min(abs(math.log(r["sigma"] / entry["sigma_target"])) for r in rows)
        assert abs(math.log(entry["sigma_actual"] / entry["sigma_target"])) == pytest.approx(best)


# --- controls --------------------------------------------------------------------------------

def test_relative_mesh_and_the_excluded_clocks():
    sigmas = edm_sigmas(68, 80.0, 0.002, 7.0)
    eta = sigmas ** (1 / 7)
    assert relative_mesh(68, 80.0, 0.002, 7.0) == pytest.approx((eta[0] - eta[1]) / eta[-1])
    assert all(relative_mesh(K, 80.0, 0.002, 1.0) > 1.0 for K in (68, 136, 272, 544))
    assert relative_mesh(68, 80.0, 0.002, 7.0) < 0.06 and relative_mesh(544, 80.0, 0.002, 2.0) < 0.5


def test_kappa():
    assert kappa(1e-6, 1 / 7) == pytest.approx(1.0, abs=1e-5)
    assert kappa(0.2, 1 / 7) < kappa(0.02, 1 / 7) and math.isnan(kappa(1.5, 1 / 7))


def test_oracle_derivative_bound_holds():
    x = np.linspace(-8.0, 8.0, 2001)
    for sigma in (0.01, 0.1, 1.0, 4.0, 80.0):
        assert MIX.posterior_mean(x, sigma)[1].max() <= oracle_derivative_bound(MIX, sigma) + 1e-12


def test_starting_noise_grids_share_the_clock_step_and_are_supercritical():
    cfg = {**CONTROLS, "starting_noise": {**CONTROLS["starting_noise"], "reference_K": 68}}
    rows = starting_noise_rows(MIX, cfg, 256)
    assert len({round(r["h"], 15) for r in rows}) == 1
    for r in rows:
        eta = edm_sigmas(r["K"], r["sigma_max_actual"], 0.002, 7.0) ** (1 / 7)
        np.testing.assert_allclose(-np.diff(eta), r["h"], rtol=1e-10)
        assert 0.0 < r["sum_defect_high_damped"] < r["sum_defect_high_undamped"]
    reference = starting_noise_rows(MIX, CONTROLS, 256)[1]      # sigma_max = 80, K = 272
    assert reference["b_eff"] > 6 / 7 and reference["certificate_holds"]


def test_synchronous_calibration_keeps_ranks_and_levels():
    u = midpoint_levels(64)
    assert all(np.all(np.diff(state) > 0) for state in euler_states(MIX.posterior_mean, edm_sigmas(8), u))
    row = synchronous_rows(MIX, {"oracle": MIX.posterior_mean}, {**CONTROLS, "synchronous": {
        **CONTROLS["synchronous"], "K": [17]}}, 256)[0]
    assert row["n_levels"] == sum(1 for s in edm_sigmas(17)[:-1] if s <= 2.0)
    assert row["Lambda_window"] <= row["ot_linear"] + row["ot_quadratic"] + 1e-10


# --- oracle refinement -----------------------------------------------------------------------

def test_shared_quantiles_reproduce_independent_grids():
    grid = {"sigma_max": 80.0, "sigma_min": 0.002, "rho_edm": 7.0}
    summary, _ = nested_oracle_rows(MIX, [4, 8, 16], 128, grid, sigma_hi=1.8)
    for row in summary:
        direct = recursion_decomposition(exact_oracle_rows(MIX, edm_sigmas(row["K"], **grid), 128))
        for key, value in direct.items():
            assert row[key] == pytest.approx(value, rel=1e-11, abs=1e-13)


def test_orders_pair_the_same_quadrature():
    rows = [{"K": k, "M": m, "initialization": "exact", "final_error": e, "propagated_discretization": 2 * e,
             "sum_defect_measured": 1 / k, "Lambda": lam}
            for k, m, e, lam in [(8, 16, .2, .4), (4, 64, .4, .1), (4, 16, .4, .2), (8, 64, .1, .3)]]
    orders = {r["M"]: r for r in refinement_orders(rows)}
    assert orders[16]["order_error"] == pytest.approx(1.0)
    assert orders[64]["order_error"] == pytest.approx(2.0)

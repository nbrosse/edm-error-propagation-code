"""Checks of the CIFAR-side code listed under "Software checks" in appendix J, on small known maps."""

import math

import numpy as np
import pytest
import torch

from edm_audit.cifar import (
    bootstrap_indices, class_latents, energy_weighted_bootstrap, energy_weighted_summary, euler_trajectory,
    integrate, jvp, lanczos_top_eigenvalue, nested_counts, power_iteration, rayleigh_quotient, segment_levels,
    segment_step_metrics, sym_jvp, synchronous_linear_budget, synchronous_median_rate_budget,
    synchronous_step_metrics, trajectory_influence, vjp,
)
from edm_audit.common import edm_sigmas
from edm_audit.run_secant_recheck import simpson
from edm_audit.run_transport import class_average


GRID = {"sigma_max": 2.0, "sigma_min": 0.1, "rho_edm": 7.0}


class TinyNet:
    """A nonlinear denoiser on 1 x 2 x 2 images."""
    sigma_min, sigma_max = 0.002, 80.0
    img_channels, img_resolution, label_dim = 1, 2, 0

    def round_sigma(self, sigma):
        return sigma

    def __call__(self, x, sigma, labels=None):
        return torch.tanh(x / (1.0 + sigma))


class LinearNet(TinyNet):
    def __call__(self, x, sigma, labels=None):
        return 1.4 * x


def symmetric_with_spectrum(eigenvalues, seed):
    q, _ = torch.linalg.qr(torch.randn(len(eigenvalues), len(eigenvalues), dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(seed)))
    return (q * eigenvalues) @ q.T


def synchronous_pair(K=3, n=5, batch_size=2):
    net = TinyNet()
    latents, labels = class_latents(net, 0, n, 1234, "cpu")
    coarse = integrate(net, latents, labels, K, GRID, range(K + 1), batch_size)
    fine = integrate(net, latents, labels, 2 * K, GRID, range(0, 2 * K + 1, 2), batch_size)
    return coarse, fine


# --- sampler ---------------------------------------------------------------------------------

def test_schedules_nest_exactly_and_the_sampler_is_float32():
    latents = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    runs = [euler_trajectory(TinyNet(), latents, None, K, GRID, []) for K in (17, 34, 68)]
    assert runs[0].sigmas == runs[1].sigmas[::2] == runs[2].sigmas[::4]
    assert runs[0].x0.dtype == torch.float32


def test_latents_depend_on_the_class_but_not_on_the_batch_size():
    (a, _), (b, _) = synchronous_pair(batch_size=2), synchronous_pair(batch_size=5)
    assert all(torch.equal(a.states[i], b.states[i]) for i in a.states)
    other, _ = class_latents(TinyNet(), 1, 5, 1234, "cpu")
    assert not torch.equal(other, class_latents(TinyNet(), 0, 5, 1234, "cpu")[0])


def test_segment_levels_are_nearest_in_log_noise():
    targets = [0.004, 0.01, 0.1, 3.0, 8.0]
    for K in (68, 272):
        sigmas = edm_sigmas(K)
        for row in segment_levels(sigmas, targets):
            best = min(abs(math.log(s / row["target"])) for s in sigmas[:-1])
            assert abs(math.log(row["sigma_actual"] / row["target"])) == pytest.approx(best)


# --- synchronous metrics ---------------------------------------------------------------------

def test_synchronous_identity_and_moments():
    """A_sync^2 = 1 + 2 ell E_sync + ell^2 Q_sync^2, and the rows are means of the per-sample moments."""
    coarse, fine = synchronous_pair()
    rows, per_sample = synchronous_step_metrics(coarse, fine, 3)
    assert not rows[0]["ratio_defined"] and math.isnan(rows[0]["E_sync"])   # same latent: zeta_0 = 0
    for r, sample in zip(rows[1:], per_sample[1:]):
        assert r["A_sync"] ** 2 == pytest.approx(1 + 2 * r["ell_j"] * r["E_sync"] + (r["ell_j"] * r["Q_sync"]) ** 2, rel=1e-6)
        assert float(sample["cross"].mean()) == pytest.approx(r["mean_cross"])


def test_class_average_pools_moments_not_ratios():
    coarse, fine = synchronous_pair()
    rows = []
    for c, scale in ((0, 1.0), (1, 3.0)):
        row = dict(synchronous_step_metrics(coarse, fine, 3)[0][2], K=3, **{"class": c})
        row["mean_cross"] *= scale
        rows.append(row)
    avg = class_average(rows)
    delta2 = rows[0]["mean_delta2"]
    assert avg["sigma_E_sync"] == pytest.approx((rows[0]["mean_cross"] + rows[1]["mean_cross"]) / 2 / delta2 - 1.0)


# --- segment metrics -------------------------------------------------------------------------

def test_segment_chain_on_a_linear_map():
    """For D = 1.4 x every direction has the rate (1.4 - 1) / sigma, so all rates agree, and the
    integral of lambda_par reproduces the secant rate."""
    x = torch.randn(3, 1, 2, 2)
    result = segment_step_metrics(LinearNet(), x, x + torch.randn_like(x), 2.0, None, lanczos_iters=4, random_seed=5)
    expected = torch.full((3,), 0.2, dtype=torch.float64)
    for name in [f"lambda_{kind}_{n}" for kind in ("par", "rand", "max") for n in (3, 5, 9)] + ["lambda_par_integral"]:
        torch.testing.assert_close(result[name], expected, rtol=2e-6, atol=2e-7)


def test_nested_node_counts_and_simpson():
    assert nested_counts(9) == [3, 5, 9] and nested_counts(65)[-4:] == [9, 17, 33, 65]
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    assert float(simpson((3 * t ** 3 - t ** 2 + 2).unsqueeze(0))) == pytest.approx(3 / 4 - 1 / 3 + 2, abs=1e-12)


# --- Jacobian actions ------------------------------------------------------------------------

def test_sym_jvp_returns_the_jvp_and_the_vjp():
    g = torch.Generator().manual_seed(0)
    x, v, w = (torch.randn(*shape, generator=g, dtype=torch.float64) for shape in ((3, 5), (3, 5), (5, 5)))
    f = lambda z: torch.tanh(z @ w.T) + z.square()
    jv, jtv = sym_jvp(f, x, v)
    torch.testing.assert_close(jv, jvp(f, x, v))
    torch.testing.assert_close(jtv, vjp(f, x, v))


def test_power_iteration_converges_from_below_per_start():
    matrix = torch.tensor([[3.0, 1.0], [0.0, -1.0]])
    truth = float(torch.linalg.matrix_norm(matrix, ord=2))
    torch.manual_seed(0)
    out = power_iteration(lambda z: z @ matrix.T, torch.randn(4, 2), 32, 2, record=(8, 16))
    assert out[32].shape == (2, 4)
    assert bool((out[8] <= out[16] + 1e-6).all()) and bool((out[16] <= out[32] + 1e-6).all())
    assert bool((out[32] <= truth + 1e-6).all())
    torch.testing.assert_close(out[32].max(dim=0).values, torch.full((4,), truth), rtol=1e-4, atol=1e-4)


def test_lanczos_finds_the_top_eigenvalue_of_the_symmetric_part():
    d = 400
    eigenvalues = torch.linspace(-0.5, 1.0, d, dtype=torch.float64)
    eigenvalues[0], eigenvalues[-1] = -5.0, 1.6       # the most negative one dominates in modulus
    b = torch.randn(d, d, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    matrix = symmetric_with_spectrum(eigenvalues, 1) + 0.3 * (b - b.T)
    torch.manual_seed(3)
    out = lanczos_top_eigenvalue(lambda z: z @ matrix.T, torch.randn(4, d, dtype=torch.float64), 40)
    torch.testing.assert_close(out["lam"], torch.full((4,), 1.6, dtype=torch.float64), rtol=0, atol=1e-6)
    assert bool((out["residual"] < 1e-3).all())


def test_lanczos_handles_an_invariant_krylov_space():
    matrix = torch.diag(torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    torch.manual_seed(0)
    out = lanczos_top_eigenvalue(lambda z: z @ matrix.T, torch.randn(2, 5, dtype=torch.float64), 5)
    torch.testing.assert_close(out["lam"], torch.full((2,), 2.0, dtype=torch.float64))


def test_rayleigh_quotient_of_a_known_symmetric_matrix():
    matrix = symmetric_with_spectrum(torch.tensor([2.0, -1.0, 0.5, -3.0], dtype=torch.float64), 11)
    x, u = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    expected = ((u @ matrix.T) * u).sum(1) / u.square().sum(1)
    torch.testing.assert_close(rayleigh_quotient(lambda z: z @ matrix.T, x, u), expected, rtol=1e-10, atol=1e-12)


# --- aggregation -----------------------------------------------------------------------------

def test_energy_weights_zero_displacements_and_extreme_values():
    summary = energy_weighted_summary(torch.tensor([1.0, 3.0, 0.0]), torch.tensor([2.0, 4.0, 99.0]))
    assert summary["weighted_mean"] == pytest.approx(3.5) and summary["ess"] == pytest.approx(1.6)
    assert (summary["valid"], summary["n_zero"]) == (True, 1)
    extreme = energy_weighted_summary(torch.ones(3), torch.tensor([0.1, 0.1, 1e6]))
    assert extreme["weighted_mean"] == pytest.approx((0.2 + 1e6) / 3) and extreme["loo_max_abs_change"] > 1e5


def test_a_non_finite_value_invalidates_the_aggregate():
    delta2, values = torch.tensor([1.0, 2.0, 1.0]), torch.tensor([1.0, 2.0, float("nan")])
    summary = energy_weighted_summary(delta2, values, ids=["c0:l0", "c0:l1", "c3:l7"])
    assert not summary["valid"] and math.isnan(summary["weighted_mean"]) and summary["invalid_ids"] == "c3:l7"
    assert math.isnan(energy_weighted_bootstrap(delta2, values, bootstrap_indices([0, 0, 0], 50, 1))["boot_lo"])


def test_bootstrap_is_stratified_and_reproducible():
    classes = [0, 0, 0, 1, 1, 1]
    indices = bootstrap_indices(classes, 64, 3)
    labels = torch.as_tensor(classes)
    assert bool((labels[indices] == labels).all())
    torch.testing.assert_close(indices, bootstrap_indices(classes, 64, 3))


def influence_inputs():
    """[steps, classes, trajectories] moments; trajectory 2 of class 1 carries most of the energy."""
    delta2, cross = np.ones((3, 2, 4)), 1.5 * np.ones((3, 2, 4))
    delta2[:, 1, 2], cross[:, 1, 2] = 40.0, 90.0
    return delta2, cross, np.array([1.0, 0.5, 0.25]), np.array([0.5, 0.25, 0.1])


def test_removal_takes_the_same_latent_at_every_level_and_keeps_class_weights():
    delta2, cross, sigmas, ell = influence_inputs()
    uniform_delta2, uniform_cross = delta2.copy(), cross.copy()
    uniform_delta2[:, 1, 2], uniform_cross[:, 1, 2] = 1.0, 1.5
    dropped = synchronous_linear_budget(delta2, cross, sigmas, ell, drop=(1, 2))
    assert dropped == pytest.approx(synchronous_linear_budget(uniform_delta2[:, :, :3], uniform_cross[:, :, :3], sigmas, ell))
    report = trajectory_influence(delta2, cross, sigmas, ell, class_ids=[3, 7])
    assert (report["most_influential_class"], report["most_influential_latent"]) == (7, 2)
    assert report["budget_min"] <= report["budget"] <= report["budget_max"]
    # The median rate follows the typical pair, the energy-weighted rate the dominant one.
    assert synchronous_median_rate_budget(delta2, cross, sigmas, ell) == pytest.approx(dropped)
    assert synchronous_linear_budget(delta2, cross, sigmas, ell) > dropped

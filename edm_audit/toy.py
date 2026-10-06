"""One-dimensional benchmark: a Gaussian mixture, its denoisers, and the law-level quadrature.

A law on R is represented by its M midpoint quantiles. In one dimension sorting gives the
optimal coupling, so W2 between two laws is the RMS difference of their sorted quantiles, and
every law-level quantity of the manuscript (appendix J) is computed exactly up to this quadrature.

A denoiser is a function ``(y, sigma) -> (D, dD/dy)`` on float64 arrays.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri
from torch import nn

from edm_audit.common import edm_coeffs, nearest_in_log


Denoiser = Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]


# ----------------------------------------------------------------------------------------------
# The data law q and its noised laws mu_sigma = q * N(0, sigma^2)

@dataclass(frozen=True)
class Mixture1D:
    weights: np.ndarray
    means: np.ndarray
    width: float  # common standard deviation of the components

    @classmethod
    def standardized(cls, weights, centers, width: float, sigma_data: float = 0.5) -> "Mixture1D":
        """Mixture with mean exactly 0 and standard deviation exactly ``sigma_data``."""
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        c = np.asarray(centers, dtype=np.float64)
        c = c - w @ c
        spread = sigma_data ** 2 - width ** 2
        return cls(w, c * np.sqrt(spread / (w @ c ** 2)), float(width))

    def mean(self) -> float:
        return float(self.weights @ self.means)

    def variance(self) -> float:
        return float(self.weights @ (self.means ** 2 + self.width ** 2)) - self.mean() ** 2

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = rng.choice(len(self.weights), size=int(n), p=self.weights)
        return self.means[k] + self.width * rng.standard_normal(int(n))

    def pdf(self, x, sigma: float) -> np.ndarray:
        sd = np.hypot(self.width, sigma)
        z = (np.asarray(x, dtype=np.float64)[..., None] - self.means) / sd
        return np.exp(-0.5 * z ** 2) @ self.weights / (sd * np.sqrt(2.0 * np.pi))

    def cdf(self, x, sigma: float) -> np.ndarray:
        sd = np.hypot(self.width, sigma)
        return ndtr((np.asarray(x, dtype=np.float64)[..., None] - self.means) / sd) @ self.weights

    def quantiles(self, u, sigma: float, grid_size: int = 4096, iters: int = 45) -> np.ndarray:
        """Quantiles of mu_sigma at the levels ``u``: bisection on the CDF, bracketed on a grid."""
        u = np.asarray(u, dtype=np.float64)
        sd = np.hypot(self.width, sigma)
        grid = np.linspace(self.means.min() - 12.0 * sd, self.means.max() + 12.0 * sd, grid_size)
        i = np.clip(np.searchsorted(self.cdf(grid, sigma), u), 1, grid_size - 1)
        lo, hi = grid[i - 1], grid[i]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid, sigma) < u
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def posterior_mean(self, y, sigma) -> Tuple[np.ndarray, np.ndarray]:
        """Oracle denoiser: D = E[X0 | X0 + sigma Z = y] and dD/dy = Var(X0 | y) / sigma^2 (Tweedie)."""
        sig = np.asarray(sigma, dtype=np.float64)
        s = sig[..., None]
        y = np.asarray(y, dtype=np.float64)[..., None]
        s2 = self.width ** 2
        v = s2 + s ** 2
        logr = np.log(self.weights) - 0.5 * (y - self.means) ** 2 / v
        r = np.exp(logr - logr.max(axis=-1, keepdims=True))
        r /= r.sum(axis=-1, keepdims=True)                       # component responsibilities
        m = self.means + s2 / v * (y - self.means)               # E[X0 | y, component]
        c = s2 * s ** 2 / v                                      # Var[X0 | y, component]
        D = (r * m).sum(axis=-1)
        var = (r * (c + (m - D[..., None]) ** 2)).sum(axis=-1)
        return D, var / sig ** 2


def misspecified(mix: Mixture1D, weights, center_scale: float, width_scale: float) -> Mixture1D:
    """A deliberately wrong law, whose posterior mean serves as a denoiser with learning error."""
    w = np.asarray(weights, dtype=np.float64)
    return Mixture1D(w / w.sum(), mix.means * float(center_scale), mix.width * float(width_scale))


# ----------------------------------------------------------------------------------------------
# Denoisers other than posterior means

def perturbed_oracle(mix: Mixture1D, eps: float, sigma_data: float) -> Denoiser:
    """Oracle plus a residual of amplitude eps (eq:toy_perturbed_oracle).

    D_hat = D + eps c_out tanh(c_in (y - m)): the EDM-normalized residual is at most eps at every
    level, so this learning error does not vanish under refinement.
    """
    m = mix.mean()

    def denoise(y, sigma):
        D, dD = mix.posterior_mean(y, sigma)
        _, c_out, c_in = edm_coeffs(float(sigma), float(sigma_data))
        u = np.tanh(c_in * (y - m))
        return D + eps * c_out * u, dD + eps * c_out * c_in * (1.0 - u ** 2)

    return denoise


class EDMPrecondMLP(nn.Module):
    """D(y, sigma) = c_skip y + c_out F(c_in y, c_noise) with an MLP F (Karras et al., 2022)."""

    def __init__(self, sigma_data: float = 0.5, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.sigma_data = float(sigma_data)
        layers, width_in = [], 2
        for _ in range(int(depth)):
            layers += [nn.Linear(width_in, int(hidden)), nn.SiLU()]
            width_in = int(hidden)
        layers.append(nn.Linear(width_in, 1))
        self.F = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, sigma) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=y.dtype).expand_as(y)
        c_skip, c_out, c_in = edm_coeffs(sigma, self.sigma_data)
        c_noise = sigma.log() / 4.0
        return c_skip * y + c_out * self.F(torch.stack([c_in * y, c_noise], dim=-1)).squeeze(-1)


def edm_loss_weight(sigma, sigma_data: float):
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def train_mlp(mix: Mixture1D, seed: int, *, sigma_data: float, hidden: int, depth: int, steps: int,
              batch_size: int, lr: float, ema: float, grad_clip: float, P_mean: float, P_std: float
              ) -> EDMPrecondMLP:
    """EDM training on samples of ``mix``; returns the EMA network in float64."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    net = EDMPrecondMLP(sigma_data, hidden, depth)
    ema_net = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(steps))
    for step in range(int(steps)):
        x0 = torch.as_tensor(mix.sample(batch_size, rng), dtype=torch.float32)
        sigma = torch.exp(P_mean + P_std * torch.randn(batch_size))
        y = x0 + sigma * torch.randn(batch_size)
        loss = (edm_loss_weight(sigma, sigma_data) * (net(y, sigma) - x0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), float(grad_clip))
        opt.step()
        sched.step()
        with torch.no_grad():
            for p_ema, p in zip(ema_net.parameters(), net.parameters()):
                p_ema.lerp_(p, 1.0 - float(ema))
        if step % 2000 == 0 or step == int(steps) - 1:
            print(f"[train seed={seed}] step {step} loss {loss.item():.4f}")
    return ema_net.double().eval()


def validation_losses(net: EDMPrecondMLP, mix: Mixture1D, P_mean: float, P_std: float,
                      n: int = 65536) -> Dict[str, float]:
    """EDM loss of ``net`` on a fixed batch, and that of the exact posterior mean (the floor)."""
    rng = np.random.default_rng(987654321)
    x0 = mix.sample(n, rng)
    sigma = np.exp(P_mean + P_std * rng.standard_normal(n))
    y = x0 + sigma * rng.standard_normal(n)
    weight = edm_loss_weight(sigma, net.sigma_data)
    with torch.no_grad():
        d_net = net(torch.as_tensor(y), torch.as_tensor(sigma)).numpy()
    d_oracle, _ = mix.posterior_mean(y, sigma)
    return {"val_loss": float(np.mean(weight * (d_net - x0) ** 2)),
            "oracle_floor": float(np.mean(weight * (d_oracle - x0) ** 2))}


def mlp_denoiser(net: EDMPrecondMLP, chunk: int = 65536) -> Denoiser:
    """Wrap a float64 network as a denoiser; dD/dy by autograd (the map is elementwise)."""

    def denoise(y, sigma):
        D_out, dD_out = [], []
        for start in range(0, len(y), chunk):
            y_t = torch.as_tensor(y[start:start + chunk], dtype=torch.float64).requires_grad_(True)
            D = net(y_t, float(sigma))
            (dD,) = torch.autograd.grad(D.sum(), y_t)
            D_out.append(D.detach().numpy())
            dD_out.append(dD.numpy())
        return np.concatenate(D_out), np.concatenate(dD_out)

    return denoise


# ----------------------------------------------------------------------------------------------
# Law-level quantities of one Euler step Phi_j = (1 - a_j) I + a_j D, a_j = 1 - sigma_{j+1}/sigma_j

# B_{d,rho} = rho^{-2} sqrt(d) [|1 - rho| + 4(d + 3)] of the universal local-bias bound
# (thm:global_curvature_bound, lem:local_euler_defect), at d = 1 and rho = 1.
B_D1 = 16.0
# The constant of the earlier, tighter form rho^{-2}[|1 - rho| sqrt(d) + 4 sqrt(d(d+2)(d+4))], used by
# the frozen runs in audit_runs/results (Delta_j is linear in it; see figure_data.UNIVERSAL_RESCALE).
B_D1_FROZEN_RUNS = 4.0 * math.sqrt(15.0)


def midpoint_levels(M: int) -> np.ndarray:
    return (np.arange(int(M), dtype=np.float64) + 0.5) / int(M)


def defect_shape(a: float) -> float:
    """g(a) = a + (1 - a) log(1 - a); a series below 1e-4, where the two terms cancel to O(a^2)."""
    if not 0.0 <= a < 1.0:
        raise ValueError(f"the Euler weight must satisfy 0 <= a < 1, got {a!r}")
    if a < 1e-4:
        return a ** 2 / 2.0 + a ** 3 / 6.0 + a ** 4 / 12.0
    return a + (1.0 - a) * math.log1p(-a)


def local_defect_bound(sigma: float, a: float) -> float:
    """Delta_j = B_{1,1} sigma_j g(a_j), the universal bound on the local defect."""
    return B_D1 * float(sigma) * defect_shape(float(a))


def _rms(values) -> float:
    return math.sqrt(float(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _refined_max(fun, grid: np.ndarray, values: np.ndarray) -> Tuple[float, float, bool]:
    """Maximum of ``fun``: the four best grid points refined in their cells. Returns
    (value, argmax, whether the maximum sits on the boundary of the grid)."""
    order = np.argsort(values)[-min(4, len(values)):]
    best_i = int(order[-1])
    best_x, best_value = float(grid[best_i]), float(values[best_i])
    boundary = best_i in (0, len(grid) - 1)
    for i in map(int, order):
        if i in (0, len(grid) - 1):
            continue
        result = minimize_scalar(lambda x: -float(fun(np.asarray([x]))[0]),
                                 bounds=(float(grid[i - 1]), float(grid[i + 1])),
                                 method="bounded", options={"xatol": 1e-12})
        if -float(result.fun) > best_value:
            best_x, best_value, boundary = float(result.x), -float(result.fun), False
    return best_value, best_x, boundary


def spatial_envelope(mix: Mixture1D, denoiser: Denoiser, sigma: float, sigma_data: float,
                     radius_sd: float, points: int) -> Dict[str, float]:
    """Field maxima (eq:toy_field_proxy) over I_sigma = [m_min - R s, m_max + R s], s = sqrt(w^2 + sigma^2).

    The interval depends on the law and sigma, never on the sampler or K. These are truncated
    empirical maxima, not certificates on R.
    """
    scale = math.hypot(mix.width, sigma)
    left, right = float(np.min(mix.means) - radius_sd * scale), float(np.max(mix.means) + radius_sd * scale)
    grid = np.linspace(left, right, int(points))
    c_skip = edm_coeffs(float(sigma), float(sigma_data))[0]

    def expansion(x):                     # one-sided rate (D' - 1) / sigma
        return (denoiser(x, sigma)[1] - 1.0) / sigma

    def precond_deviation(x):             # |D' - c_skip|
        return np.abs(denoiser(x, sigma)[1] - c_skip)

    values = expansion(grid)
    los, arg_los, los_boundary = _refined_max(expansion, grid, values)
    lv, arg_lv, lv_boundary = _refined_max(lambda x: np.abs(expansion(x)), grid, np.abs(values))
    dev, arg_dev, dev_boundary = _refined_max(precond_deviation, grid, precond_deviation(grid))
    return {
        "spatial_left": left, "spatial_right": right,
        "spatial_radius_sd": float(radius_sd), "spatial_points": int(points),
        "Los_pts": los, "argmax_Los_pts": arg_los, "Los_pts_boundary": los_boundary,
        "Lv_pts": lv, "argmax_Lv_pts": arg_lv, "Lv_pts_boundary": lv_boundary,
        "b_pts": 1.0 - c_skip - dev, "argmax_precond_deviation": arg_dev,
        "precond_deviation_boundary": dev_boundary,
    }


def step_metrics(t: np.ndarray, s: np.ndarray, denoiser: Denoiser, mix: Mixture1D, sigma: float,
                 sigma_next: float, sigma_data: float, target_next: np.ndarray,
                 spatial: Optional[Dict] = None) -> Tuple[Dict, np.ndarray]:
    """One Euler step from the sorted sampler quantiles ``s``, compared with the target quantiles ``t``.

    ``target_next`` holds the target quantiles at ``sigma_next``, i.e. the exact flow image of
    ``t``, which gives the measured local defect and learning error of the one-step recursion
    e_{j+1} <= gamma_j e_j + delta_j + r_j. With ``spatial`` (the domain settings), the field
    proxy and its wider/finer check are added. Returns the row and the next sampler state.
    """
    ell = sigma - sigma_next
    a = ell / sigma
    Dt, _ = denoiser(t, sigma)
    Ds, _ = denoiser(s, sigma)
    phi_t, phi_s = (1.0 - a) * t + a * Dt, (1.0 - a) * s + a * Ds

    diff = t - s
    e = _rms(diff)
    # gamma re-optimizes the coupling of the pushforwards (by sorting); A_OT transports the input
    # coupling. They coincide when the predictor is increasing.
    num = _rms(np.sort(phi_t) - np.sort(phi_s))
    if e > 0:
        dv = ((t - Dt) - (s - Ds)) / sigma          # velocity difference
        gamma, A_OT = num / e, _rms(phi_t - phi_s) / e
        E_OT, Q_OT = -float(np.mean(diff * dv)) / e ** 2, _rms(dv) / e
    else:
        gamma, A_OT, E_OT, Q_OT = 1.0, 1.0, 0.0, 0.0

    row = {
        "sigma": sigma, "sigma_next": sigma_next, "ell": ell, "a": a, "e": e, "num": num,
        "gamma": gamma, "A_OT": A_OT, "E_OT": E_OT, "E_OT_plus": max(E_OT, 0.0), "Q_OT": Q_OT,
        "directional_term": ell * max(E_OT, 0.0) + 0.5 * (ell * Q_OT) ** 2,
        "b_law": (1.0 - gamma) / a,
    }
    if spatial is not None:
        main = spatial_envelope(mix, denoiser, sigma, sigma_data, spatial["radius_sd"], spatial["points"])
        check = spatial_envelope(mix, denoiser, sigma, sigma_data,
                                 spatial["check_radius_sd"], spatial["check_points"])
        field = ell * max(0.0, main["Los_pts"]) + 0.5 * (ell * main["Lv_pts"]) ** 2
        field_check = ell * max(0.0, check["Los_pts"]) + 0.5 * (ell * check["Lv_pts"]) ** 2
        row.update({
            "field_term_pts": field, "field_term_pts_check": field_check, **main,
            "Los_pts_check": check["Los_pts"], "Lv_pts_check": check["Lv_pts"],
            "b_pts_check": check["b_pts"],
            "spatial_check_boundary": bool(check["Los_pts_boundary"] or check["Lv_pts_boundary"]
                                           or check["precond_deviation_boundary"]),
            "field_pts_rel_change": abs(field_check - field) / max(abs(field_check), 1e-300),
        })
    row["Delta_j"] = local_defect_bound(sigma, a)

    D_exact = Dt if denoiser == mix.posterior_mean else mix.posterior_mean(t, sigma)[0]
    discretization = _rms(target_next - ((1.0 - a) * t + a * D_exact))
    learning = a * _rms(Dt - D_exact)
    e_next = _rms(target_next - np.sort(phi_s))
    row.update({
        "e_next": e_next, "stability_term": num, "discretization_term": discretization,
        "learning_term": learning, "recursion_rhs": num + discretization + learning,
        "recursion_slack": num + discretization + learning - e_next,
    })
    return row, np.sort(phi_s)


def audit_grid(denoisers: Dict, mix: Mixture1D, sigmas: np.ndarray, M: int, sigma_data: float,
               spatial: Optional[Dict] = None) -> List[Dict]:
    """Run every denoiser on G_K from the Gaussian start N(0, sigma_0^2), sharing the target quantiles."""
    u = midpoint_levels(M)
    states = {key: float(sigmas[0]) * ndtri(u) for key in denoisers}
    rows: List[Dict] = []
    for j in range(len(sigmas) - 1):
        sigma, sigma_next = float(sigmas[j]), float(sigmas[j + 1])
        target, target_next = mix.quantiles(u, sigma), mix.quantiles(u, sigma_next)
        for key, denoiser in denoisers.items():
            row, states[key] = step_metrics(target, np.sort(states[key]), denoiser, mix, sigma,
                                            sigma_next, sigma_data, target_next, spatial)
            rows.append({"arm": key, "K": len(sigmas) - 1, "step_index": j, **row})
    return rows


def exact_oracle_rows(mix: Mixture1D, sigmas: np.ndarray, M: int, sigma_data: float = 0.5) -> List[Dict]:
    """The oracle started from the exact top law: no initialization or learning error, only
    discretization."""
    u = midpoint_levels(M)
    state = mix.quantiles(u, float(sigmas[0]))
    rows = []
    for j in range(len(sigmas) - 1):
        sigma, sigma_next = float(sigmas[j]), float(sigmas[j + 1])
        row, state = step_metrics(mix.quantiles(u, sigma), state, mix.posterior_mean, mix, sigma,
                                  sigma_next, sigma_data, mix.quantiles(u, sigma_next))
        rows.append({"step_index": j, **row})
    return rows


def recursion_decomposition(rows: List[Dict]) -> Dict[str, float]:
    """Unroll the measured one-step recursion into its propagated contributions
    (eq:toy_propagated_decomposition).

    C_init + C_disc + C_learn is an a posteriori upper bound on the final error, not a
    decomposition of it. The same factors also propagate the universal bound Delta_j.
    """
    ordered = sorted(rows, key=lambda row: int(row["step_index"]))
    amplification, disc, learn, delta = 1.0, 0.0, 0.0, 0.0
    for row in reversed(ordered):
        disc += row["discretization_term"] * amplification
        learn += row["learning_term"] * amplification
        delta += row["Delta_j"] * amplification
        amplification *= row["gamma"]
    init = float(ordered[0]["e"]) * amplification
    final_error = float(ordered[-1]["e_next"])
    return {
        "final_error": final_error,
        "amplification_product": amplification,
        "propagated_initialization": init,
        "propagated_discretization": disc,
        "propagated_learning": learn,
        "recursion_bound": init + disc + learn,
        "recursion_slack": init + disc + learn - final_error,
        "min_one_step_slack": min(row["recursion_slack"] for row in ordered),
        "propagated_delta": delta,
        "recursion_bound_delta": init + delta + learn,
    }


# ----------------------------------------------------------------------------------------------
# Accumulation over the low-noise block

def high_noise_threshold(mix: Mixture1D, denoiser: Denoiser, sigma_min: float, sigma_max: float,
                         sigma_data: float, b_hi: float, spatial: Dict, n: int = 400) -> float:
    """Bottom of the top contiguous block of levels where the field margin b_pts >= b_hi.

    This is one denoiser's candidate; the experiment uses the largest candidate for every arm.
    """
    sigma_hi = None
    for sigma in np.geomspace(sigma_max, sigma_min, n):
        margin = spatial_envelope(mix, denoiser, float(sigma), sigma_data,
                                  spatial["radius_sd"], spatial["points"])["b_pts"]
        if margin < b_hi:
            if sigma_hi is None:
                raise ValueError(f"empty high-noise block: b_pts(sigma_max) = {margin:.4g} < b_hi = {b_hi:g}")
            break
        sigma_hi = float(sigma)
    return sigma_hi


def log_amplification(rows: List[Dict]) -> float:
    """Lambda = sum_j log max(1, gamma_j)."""
    return sum(math.log(max(1.0, row["gamma"])) for row in rows)


def accumulate(rows: List[Dict], sigma_hi: float) -> Dict[str, float]:
    """Lambda_K, its directional majorant and the field proxy over the steps ending below sigma_hi."""
    low = [row for row in rows if row["sigma_next"] < sigma_hi]
    return {
        "n_split": len(rows) - len(low),
        "n_low": len(low),
        "Lambda": log_amplification(low),
        "Lambda_dir": sum(row["directional_term"] for row in low),
        "Lambda_field_pts": sum(row["field_term_pts"] for row in low),
        "Lambda_field_pts_check": sum(row["field_term_pts_check"] for row in low),
        "max_field_pts_rel_change": max((row["field_pts_rel_change"] for row in low), default=float("nan")),
        "spatial_boundary_hits": sum(bool(row["Los_pts_boundary"] or row["Lv_pts_boundary"]
                                          or row["precond_deviation_boundary"]) for row in low),
        "spatial_check_boundary_hits": sum(bool(row["spatial_check_boundary"]) for row in low),
        "min_e": min((row["e"] for row in low), default=float("nan")),
    }


def decade_sums(rows: List[Dict], sigma_hi: float) -> Dict[str, float]:
    """Lambda split over the noise decades [10^n, 10^(n+1)), the same windows for every K."""
    out: Dict[str, float] = {}
    for row in rows:
        if row["sigma_next"] < sigma_hi:
            key = f"decade_{int(math.floor(math.log10(row['sigma'])))}"
            out[key] = out.get(key, 0.0) + math.log(max(1.0, row["gamma"]))
    return out


def fixed_noise_rows(rows: List[Dict], targets: Sequence[float]) -> List[Dict]:
    """For each target noise, the rates at the step of this grid nearest to it in log noise."""
    out = []
    for target in targets:
        row = rows[nearest_in_log([r["sigma"] for r in rows], target)]
        out.append({"sigma_target": float(target), "sigma_actual": float(row["sigma"]),
                    "step_index": int(row["step_index"]), "E_OT": float(row["E_OT"]),
                    "e": float(row["e"]), "Los_pts": float(row["Los_pts"]), "gamma": float(row["gamma"])})
    return out

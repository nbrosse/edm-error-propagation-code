"""CIFAR-10 side: the pretrained EDM network, a state-recording Euler sampler, exact Jacobian
actions (JVP/VJP, power iteration, Lanczos), and the synchronous and segment metrics.

Inference is float32, as deployed; reductions over trajectories are float64.

Synchronous coupling (appendix J, "Synchronous coupling"). The coarse state x = X_j^(K) and the
fine state y = X_2j^(2K) come from the same latent and sit at the same level sigma_j. With
zeta = y - x, Delta D = D(y) - D(x) and Delta v = (zeta - Delta D) / sigma_j, the Euler predictor
Phi_j = (1 - a_j) I + a_j D satisfies A_sync^2 = 1 + 2 ell_j E_sync + ell_j^2 Q_sync^2, with

    E_sync = -E<zeta, Delta v> / E|zeta|^2      (positive = expansive)
    Q_sync = ||Delta v|| / ||zeta||.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from edm_audit.common import CODE_DIR, edm_coeffs, nearest_in_log


Tensor = torch.Tensor
NETWORK_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
DATASET = CODE_DIR / "datasets" / "cifar10-32x32-test.zip"     # held-out probe images


# ----------------------------------------------------------------------------------------------
# Network and probes

class SyntheticEDMNet(nn.Module):
    """Offline stand-in for smoke runs: D(x, sigma) = c_skip x + c_out tanh(c_in x) W. Not a model."""

    def __init__(self, img_channels: int = 1, img_resolution: int = 4, label_dim: int = 2,
                 sigma_data: float = 0.5, seed: int = 0):
        super().__init__()
        self.img_channels, self.img_resolution, self.label_dim = img_channels, img_resolution, label_dim
        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, 0.0, float("inf")
        dim = img_channels * img_resolution ** 2
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("weight", torch.randn(dim, dim, generator=generator) / dim ** 0.5)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, class_labels=None):
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).reshape(-1, 1, 1, 1)
        c_skip, c_out, c_in = edm_coeffs(sigma, self.sigma_data)
        return c_skip * x + c_out * (torch.tanh(c_in * x).flatten(1) @ self.weight).view_as(x)


def load_network(url: Optional[str], device: torch.device) -> nn.Module:
    """The pretrained EDM network (EMA weights), or the synthetic stand-in if ``url`` is None."""
    if url is None:
        print("[cifar] synthetic offline network: the numbers are not estimates")
        return SyntheticEDMNet().eval().requires_grad_(False).to(device)
    import dnnlib  # from the NVlabs/edm code in code/

    with dnnlib.util.open_url(url, verbose=True) as f:
        net = pickle.load(f)["ema"]
    return net.eval().requires_grad_(False).to(device)


def call_F_branch(net, z: Tensor, sigma: float, labels: Optional[Tensor]) -> Tensor:
    """The raw network F(z, c_noise) inside D = c_skip x + c_out F(c_in x, c_noise)."""
    c_noise = torch.full([z.shape[0]], float(sigma), device=z.device).log() / 4.0
    return net.model(z.float(), c_noise, class_labels=labels).float()


@dataclass
class ProbeBank:
    """Fixed images x0 and one fixed noise draw z: the probe at level sigma is x0 + sigma z."""
    x0: Tensor
    z: Tensor
    labels: Optional[Tensor]
    batch_size: int

    def __len__(self) -> int:
        return int(self.x0.shape[0])

    def batches(self, sigma: float):
        """Yield (x0, x0 + sigma z, labels) batch by batch."""
        for start in range(0, len(self), self.batch_size):
            sl = slice(start, start + self.batch_size)
            yield self.x0[sl], self.x0[sl] + float(sigma) * self.z[sl], None if self.labels is None else self.labels[sl]


def load_probe_bank(net, num_probes: int, batch_size: int, device: torch.device, seed: int) -> ProbeBank:
    """The first ``num_probes`` images of a seeded shuffle of the test set, and a seeded noise draw."""
    from training.dataset import ImageFolderDataset  # from the NVlabs/edm code in code/

    dataset = ImageFolderDataset(path=str(DATASET), resolution=net.img_resolution, use_labels=net.label_dim > 0,
                                 random_seed=seed, cache=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    images, labels = [], []
    for image, label in loader:
        images.append(image.to(device).float() / 127.5 - 1.0)
        labels.append(label.to(device).float())
        if sum(len(b) for b in images) >= num_probes:
            break
    x0 = torch.cat(images)[:num_probes]
    labels = torch.cat(labels)[:num_probes] if net.label_dim > 0 else None
    z = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed)).to(device)
    return ProbeBank(x0=x0, z=z, labels=labels, batch_size=batch_size)


# ----------------------------------------------------------------------------------------------
# Sampler

@dataclass
class Trajectory:
    """States of one Euler run: ``states[i]`` at level ``sigmas[i]`` (recorded indices only),
    ``denoised[i]`` = D(states[i]), and the final image ``x0``."""
    sigmas: List[float]
    states: Dict[int, Tensor]
    x0: Tensor
    denoised: Dict[int, Tensor] = field(default_factory=dict)


def euler_trajectory(net, latents: Tensor, labels: Optional[Tensor], K: int, grid: Dict,
                     record: Sequence[int]) -> Trajectory:
    """Deterministic EDM sampler (Algorithm 2 of Karras et al., S_churn = 0) with Euler steps on G_K.

    The schedule is the upstream one (float64, round_sigma, then float32), so level j of G_K and
    level 2j of G_2K coincide exactly and two runs from the same latent form a synchronous pair.
    """
    sigma_min, sigma_max = max(grid["sigma_min"], net.sigma_min), min(grid["sigma_max"], net.sigma_max)
    i = torch.arange(K + 1, dtype=torch.float64, device=latents.device)
    rho = grid["rho_edm"]
    t = (sigma_max ** (1 / rho) + i / K * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])]).float()
    record = set(record)
    states, denoised = {}, {}
    x = latents.float() * t[0]
    for j in range(K + 1):
        D = net(x, t[j], labels).float()
        if j in record:
            states[j], denoised[j] = x.detach().clone(), D.detach()
        x = x + (t[j + 1] - t[j]) * ((x - D) / t[j])
    return Trajectory([float(s) for s in t[:-1]], states, x.detach().clone(), denoised)


def class_latents(net, class_id: int, n: int, seed: int, device) -> Tuple[Tensor, Optional[Tensor]]:
    """n latents drawn on the CPU from seed + class_id, and the one-hot labels of the class."""
    shape = [n, net.img_channels, net.img_resolution, net.img_resolution]
    latents = torch.randn(shape, generator=torch.Generator().manual_seed(seed + class_id)).to(device)
    if net.label_dim == 0:
        return latents, None
    labels = torch.zeros([n, net.label_dim], device=device)
    labels[:, class_id] = 1.0
    return latents, labels


def integrate(net, latents: Tensor, labels: Optional[Tensor], K: int, grid: Dict,
              record: Sequence[int], batch_size: int) -> Trajectory:
    """``euler_trajectory`` in batches of ``batch_size``."""
    parts = [euler_trajectory(net, latents[s:s + batch_size], None if labels is None else labels[s:s + batch_size],
                              K, grid, record)
             for s in range(0, latents.shape[0], batch_size)]
    cat = lambda name, i: torch.cat([getattr(p, name)[i] for p in parts])
    return Trajectory(parts[0].sigmas, {i: cat("states", i) for i in parts[0].states},
                      torch.cat([p.x0 for p in parts]), {i: cat("denoised", i) for i in parts[0].denoised})


def segment_levels(sigmas: Sequence[float], targets: Sequence[float]) -> List[Dict]:
    """The step of the grid nearest in log noise to each prescribed level (fixed in advance)."""
    rows = []
    for target in targets:
        j = nearest_in_log(list(sigmas)[:-1], target)
        rows.append({"target": float(target), "step_index": j, "sigma_actual": float(sigmas[j])})
    return rows


# ----------------------------------------------------------------------------------------------
# Jacobian actions (exact, by autograd), batched: one estimate per sample

def _norm(x: Tensor) -> Tensor:
    return x.flatten(1).norm(dim=1).clamp_min(1e-12)


def _normalize(x: Tensor) -> Tensor:
    return x / _norm(x).view(-1, *([1] * (x.ndim - 1)))


def jvp(f, x: Tensor, v: Tensor) -> Tensor:
    """J v."""
    return torch.autograd.functional.jvp(f, x.detach().requires_grad_(True), v.detach())[1]


def vjp(f, x: Tensor, u: Tensor) -> Tensor:
    """J^T u."""
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad((f(x) * u).sum(), x)[0]


def sym_jvp(f, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """(J v, J^T v) in one double backward, for a map with f(x).shape == x.shape."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        u = v.detach().clone().requires_grad_(True)
        (jtv,) = torch.autograd.grad(f(x), x, grad_outputs=u, create_graph=True)
        (jv,) = torch.autograd.grad(jtv, u, grad_outputs=v.detach())
    return jv.detach(), jtv.detach()


def power_iteration(f, x: Tensor, iters: int, starts: int, record: Sequence[int] = ()) -> Dict[int, Tensor]:
    """||J||_op by power iteration on J^T J, from ``starts`` random starts.

    Returns ``{n: [starts, batch]}`` for each n in ``record`` and n = ``iters``. The estimate
    converges from below, so a truncated run makes any margin derived from it optimistic;
    keeping the starts separate shows whether it has converged.
    """
    keep = set(record) | {iters}
    out = {n: [] for n in sorted(keep)}
    for _ in range(starts):
        v = _normalize(torch.randn_like(x))
        for n in range(1, iters + 1):
            u = _normalize(jvp(f, x, v).detach())
            v = _normalize(vjp(f, x, u).detach())
            if n in keep:
                out[n].append(_norm(jvp(f, x, v).detach()).cpu())
    return {n: torch.stack(values) for n, values in out.items()}


def rayleigh_quotient(f, x: Tensor, u: Tensor) -> Tensor:
    """u^T J u / |u|^2 = u^T Sym(J) u / |u|^2, one JVP."""
    uu = u.flatten(1).double()
    return ((uu * jvp(f, x, u).detach().flatten(1).double()).sum(1) / uu.square().sum(1)).cpu()


def _bdot(a: Tensor, b: Tensor) -> Tensor:
    return (a.flatten(1) * b.flatten(1)).sum(dim=1)


def _bscale(c: Tensor, v: Tensor) -> Tensor:
    return c.view(-1, *([1] * (v.ndim - 1))) * v


def _top_ritz(alphas: List[Tensor], betas: List[Tensor], k: int) -> Tuple[Tensor, Tensor]:
    """Largest eigenvalue of the leading k x k tridiagonal block and the last entry of its eigenvector."""
    T = torch.diag_embed(torch.stack(alphas[:k], dim=1))
    if k > 1:
        off = torch.stack(betas[:k - 1], dim=1)
        T = T + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    evals, evecs = torch.linalg.eigh(T)
    return evals[:, -1], evecs[:, -1, -1]


def lanczos_top_eigenvalue(f, x: Tensor, iters: int) -> Dict[str, Tensor]:
    """lambda_max(Sym J) by Lanczos with full reorthogonalization.

    Returns ``lam`` (largest Ritz value, a lower bound), ``lam_half`` (the same after iters // 2
    steps) and ``residual`` = ||Sym J y - lam y|| for the Ritz vector y. A sample whose Krylov
    space becomes invariant continues on a fresh orthogonal direction with zero coupling.
    """
    basis = [_normalize(torch.randn_like(x))]
    alphas, betas = [], []
    for k in range(iters):
        jv, jtv = sym_jvp(f, x, basis[k])
        w = 0.5 * (jv + jtv)
        alpha, scale = _bdot(basis[k], w), _norm(w)
        for _ in range(2):
            for p in basis:
                w = w - _bscale(_bdot(p, w), p)
        beta = _norm(w)
        broken = beta <= 1e-6 * scale
        alphas.append(alpha.double().cpu())
        betas.append(torch.where(broken, torch.zeros_like(beta), beta).double().cpu())
        if k == iters - 1:
            break
        if bool(broken.any()):
            fresh = torch.randn_like(x)
            for _ in range(2):
                for p in basis:
                    fresh = fresh - _bscale(_bdot(p, fresh), p)
            w = torch.where(broken.view(-1, *([1] * (x.ndim - 1))), fresh, w)
        basis.append(_normalize(w))
    lam, last = _top_ritz(alphas, betas, iters)
    lam_half, _ = _top_ritz(alphas, betas, max(1, iters // 2))
    return {"lam": lam, "lam_half": lam_half, "residual": betas[-1] * last.abs()}


# ----------------------------------------------------------------------------------------------
# Synchronous metrics

def secant_from_moments(delta2: float, cross: float, res2: float, sigma: float, suffix: str) -> Dict[str, float]:
    """E and Q from E|zeta|^2, E<zeta, Delta D> and E|zeta - Delta D|^2.

    sigma E = E<zeta, Delta D> / E|zeta|^2 - 1 and sigma Q = sqrt(E|zeta - Delta D|^2 / E|zeta|^2).
    """
    names = [f"r_D_{suffix}", f"sigma_E_{suffix}", f"sigma_Q_{suffix}", f"E_{suffix}", f"E_{suffix}_plus", f"Q_{suffix}"]
    if not delta2 > 0.0:        # the shared latent gives zeta = 0 at the first level
        return dict.fromkeys(names, float("nan"))
    r_D = cross / delta2
    sigma_Q = math.sqrt(max(res2, 0.0) / delta2)
    return {f"r_D_{suffix}": r_D, f"sigma_E_{suffix}": r_D - 1.0, f"sigma_Q_{suffix}": sigma_Q,
            f"E_{suffix}": (r_D - 1.0) / sigma, f"E_{suffix}_plus": max((r_D - 1.0) / sigma, 0.0),
            f"Q_{suffix}": sigma_Q / sigma}


def pair_moments(zeta: Tensor, delta_D: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Per sample, in float64: |zeta|^2, <zeta, Delta D> and |zeta - Delta D|^2."""
    d, o = zeta.flatten(1).double(), delta_D.flatten(1).double()
    return d.square().sum(1), (d * o).sum(1), (d - o).square().sum(1)


def synchronous_step_metrics(coarse: Trajectory, fine: Trajectory, K: int) -> Tuple[List[Dict], List[Dict]]:
    """Per-step synchronous metrics of a G_K / G_2K pair.

    Returns the rows (ensemble moments and rates) and, per step, the per-trajectory moments
    (float64, CPU), which the segment aggregation and the influence analysis reuse.
    """
    rows, per_sample = [], []
    for j in range(K):
        sigma, sigma_next = coarse.sigmas[j], coarse.sigmas[j + 1]
        a, ell = 1.0 - sigma_next / sigma, sigma - sigma_next
        zeta = fine.states[2 * j] - coarse.states[j]
        delta_D = fine.denoised[2 * j] - coarse.denoised[j]
        delta2, cross, res2 = pair_moments(zeta, delta_D)
        moments = {"delta2": delta2, "deltaD2": delta_D.flatten(1).double().square().sum(1),
                   "phi2": ((1.0 - a) * zeta + a * delta_D).flatten(1).double().square().sum(1),
                   "cross": cross, "res2": res2}
        per_sample.append({name: value.cpu() for name, value in moments.items()})
        mean = {name: float(value.mean()) for name, value in moments.items()}
        defined = mean["delta2"] > 0.0
        rows.append({
            "step_index": j, "sigma_j": sigma, "sigma_next": sigma_next, "ell_j": ell, "a_j": a,
            "n_samples": int(zeta.shape[0]), "ratio_defined": defined,
            **{f"mean_{name}": value for name, value in mean.items()},
            "A_sync": (mean["phi2"] / mean["delta2"]) ** 0.5 if defined else float("nan"),
            **secant_from_moments(mean["delta2"], mean["cross"], mean["res2"], sigma, "sync"),
        })
    return rows, per_sample


# ----------------------------------------------------------------------------------------------
# Segment metrics: expansion along the segment [x, y] between the coarse and the fine state

def directional_profile(net, x: Tensor, y: Tensor, sigma: float, labels, t_nodes: int) -> Tensor:
    """lambda_par(t) = -zeta^T Sym(grad v) zeta / |zeta|^2 at x + t zeta, on t_nodes equispaced
    nodes of [0, 1]; shape [batch, t_nodes], NaN where zeta = 0."""
    zeta = (y - x).detach()
    delta2 = zeta.flatten(1).double().square().sum(1).cpu()
    sigma_t = torch.tensor(float(sigma), dtype=x.dtype, device=x.device)
    rates = []
    for t in torch.linspace(0.0, 1.0, t_nodes, device=x.device, dtype=x.dtype):
        jv, jtv = sym_jvp(lambda inp: net(inp, sigma_t, labels), x + t * zeta, zeta)
        quad = (zeta.flatten(1).double() * (0.5 * (jv + jtv)).flatten(1).double()).sum(1).cpu()
        rate = (quad / delta2 - 1.0) / float(sigma)
        rates.append(torch.where(delta2 > 0, rate, torch.full_like(rate, float("nan"))))
    return torch.stack(rates, dim=1)


def nested_counts(t_nodes: int) -> List[int]:
    """Node counts 3, 5, 9, ..., t_nodes of the nested equispaced subsets."""
    counts = [t_nodes]
    while counts[-1] > 3:
        if (counts[-1] - 1) % 2:
            raise ValueError("t_nodes must be 2^k + 1")
        counts.append((counts[-1] - 1) // 2 + 1)
    return counts[::-1]


def random_unit_directions(shape, seed: int, dtype, device) -> Tensor:
    """One unit direction per segment, drawn on the CPU from a fixed seed."""
    raw = torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return (raw / raw.flatten(1).norm(dim=1).clamp_min(1e-30).view(-1, *([1] * (raw.ndim - 1)))).to(device=device, dtype=dtype)


def segment_step_metrics(net, x: Tensor, y: Tensor, sigma: float, labels, t_nodes: int = 9,
                         lanczos_iters: int = 32, random_seed: int = 0) -> Dict[str, Tensor]:
    """Three expansion rates at the same nodes of each segment (eq:segment_aggregate_chain):

    * lambda_par: along the displacement zeta (exact, one symmetric JVP per node);
    * lambda_rand: along a fixed random direction, the generic baseline (one JVP per node);
    * lambda_max: the largest local expansion, by Lanczos.

    Each is maximized over the nested node subsets (3, 5, ..., t_nodes nodes). The trapezoidal
    integral of lambda_par must reproduce the secant rate E_sync (a check of the node resolution).
    Returns CPU float64 tensors, one value per segment.
    """
    directional = directional_profile(net, x, y, sigma, labels, t_nodes)
    zeta = (y - x).detach()
    u_rand = random_unit_directions(tuple(x.shape), random_seed, x.dtype, x.device)
    sigma_t = torch.tensor(float(sigma), dtype=x.dtype, device=x.device)
    randoms, top, drift, residual = [], [], [], []
    for t in torch.linspace(0.0, 1.0, t_nodes, device=x.device, dtype=x.dtype):
        point = (x + t * zeta).detach()
        denoiser = lambda inp: net(inp, sigma_t, labels)
        randoms.append((rayleigh_quotient(denoiser, point, u_rand) - 1.0) / float(sigma))
        lanczos = lanczos_top_eigenvalue(denoiser, point, lanczos_iters)
        top.append((lanczos["lam"] - 1.0) / float(sigma))
        drift.append((top[-1] - (lanczos["lam_half"] - 1.0) / float(sigma)).abs())
        residual.append(lanczos["residual"] / float(sigma))
    rand, top = torch.stack(randoms, dim=1), torch.stack(top, dim=1)
    delta2 = zeta.flatten(1).double().square().sum(1).cpu()
    out = {"delta2": delta2, "defined": delta2 > 0}
    for n in nested_counts(t_nodes):
        nodes = list(range(0, t_nodes, (t_nodes - 1) // (n - 1)))
        out[f"lambda_par_{n}"] = directional[:, nodes].nan_to_num(nan=-math.inf).max(dim=1).values
        out[f"lambda_rand_{n}"] = rand[:, nodes].max(dim=1).values
        out[f"lambda_max_{n}"] = top[:, nodes].max(dim=1).values
    out["lambda_par_integral"] = torch.trapz(directional, dx=1.0 / (t_nodes - 1), dim=1)
    out["lanczos_half_drift"] = torch.stack(drift, dim=1).max(dim=1).values
    out["lanczos_residual"] = torch.stack(residual, dim=1).max(dim=1).values
    return out


# ----------------------------------------------------------------------------------------------
# Aggregation over trajectories

def energy_weighted_summary(delta2: Tensor, values: Tensor, ids: Optional[Sequence[str]] = None) -> Dict:
    """Mean of ``values`` with weights |zeta|^2 (eq:segment_energy_weights), untrimmed.

    A zero displacement has zero weight and is counted in n_zero. A non-finite value at positive
    weight invalidates the mean (NaN) and is reported by id rather than dropped.
    """
    delta2, values = delta2.double().cpu(), values.double().cpu()
    positive = torch.isfinite(delta2) & (delta2 > 0)
    invalid = torch.nonzero(positive & ~torch.isfinite(values)).flatten().tolist()
    base = {"n_total": int(delta2.numel()), "n_zero": int((torch.isfinite(delta2) & (delta2 <= 0)).sum()),
            "n_invalid": len(invalid), "invalid_ids": ";".join(str(ids[i] if ids is not None else i) for i in invalid)}
    if invalid or not bool(positive.any()):
        return {**base, "valid": False, "weighted_mean": float("nan"), "ess": float("nan"),
                "max_weight": float("nan"), "loo_max_abs_change": float("nan")}
    w = delta2[positive] / delta2[positive].sum()
    v = values[positive]
    mean = (w * v).sum()
    loo = float(((mean - w * v) / (1.0 - w) - mean).abs().max()) if len(w) > 1 else float("nan")
    return {**base, "valid": True, "weighted_mean": float(mean), "ess": float(1.0 / w.square().sum()),
            "max_weight": float(w.max()), "loo_max_abs_change": loo}


def bootstrap_indices(classes: Sequence[int], replicates: int, seed: int) -> Tensor:
    """Resample within each class, with replacement; the same indices serve every compared term."""
    labels = torch.as_tensor(list(classes))
    generator = torch.Generator().manual_seed(seed)
    out = torch.empty((replicates, labels.numel()), dtype=torch.long)
    for value in torch.unique(labels):
        positions = torch.nonzero(labels == value).flatten()
        out[:, positions] = positions[torch.randint(positions.numel(), (replicates, positions.numel()),
                                                    generator=generator)]
    return out


def energy_weighted_bootstrap(delta2: Tensor, values: Tensor, indices: Tensor) -> Dict[str, float]:
    """2.5% and 97.5% percentiles of the energy-weighted mean over the given resamples."""
    delta2, values = delta2.double().cpu(), values.double().cpu()
    positive = torch.isfinite(delta2) & (delta2 > 0)
    if not bool(positive.any()) or bool((positive & ~torch.isfinite(values)).any()):
        return {"boot_lo": float("nan"), "boot_hi": float("nan")}
    safe = torch.where(positive, values, torch.zeros_like(values))
    weights = delta2[indices].clamp_min(0.0)
    totals = weights.sum(dim=1)
    estimates = ((weights * safe[indices]).sum(dim=1) / totals.clamp_min(1e-300))[totals > 0]
    return {"boot_lo": float(torch.quantile(estimates, 0.025)), "boot_hi": float(torch.quantile(estimates, 0.975))}


def synchronous_linear_budget(delta2: np.ndarray, cross: np.ndarray, sigmas: np.ndarray, ell: np.ndarray,
                              drop: Optional[Tuple[int, int]] = None) -> float:
    """sum_j ell_j (E_sync_j)_+ from the per-trajectory moments, arrays [steps, classes, trajectories].

    Classes have equal weight (mean of class means). ``drop = (c, i)`` removes trajectory i of
    class c at every level, and that class is renormalized.
    """
    keep = np.ones(delta2.shape[1:], dtype=bool)
    if drop is not None:
        keep[drop] = False
    counts = keep.sum(axis=1)
    if np.any(counts == 0):
        return float("nan")
    pooled_delta2 = ((delta2 * keep).sum(axis=2) / counts).mean(axis=1)
    pooled_cross = ((cross * keep).sum(axis=2) / counts).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(pooled_delta2 > 0, (pooled_cross / pooled_delta2 - 1.0) / sigmas, 0.0)
    return float(np.sum(ell * np.maximum(rate, 0.0)))


def synchronous_median_rate_budget(delta2: np.ndarray, cross: np.ndarray, sigmas: np.ndarray, ell: np.ndarray) -> float:
    """sum_j ell_j (median_i E_sync_{j,i})_+: what the typical pair does. Descriptive only; it is not
    the energy-weighted rate that the theory bounds."""
    d2, cr = delta2.reshape(len(delta2), -1), cross.reshape(len(cross), -1)
    total = 0.0
    for j in range(len(d2)):
        positive = d2[j] > 0
        if positive.any():
            rates = (cr[j][positive] / d2[j][positive] - 1.0) / sigmas[j]
            total += float(ell[j]) * max(float(np.median(rates)), 0.0)
    return total


def trajectory_influence(delta2: np.ndarray, cross: np.ndarray, sigmas: np.ndarray, ell: np.ndarray,
                         class_ids: Sequence[int]) -> Dict:
    """The budget with each trajectory removed in turn: an influence range, not a confidence interval."""
    budget = synchronous_linear_budget(delta2, cross, sigmas, ell)
    rows = [{"class": int(class_ids[c]), "latent_index": i,
             "budget_without": synchronous_linear_budget(delta2, cross, sigmas, ell, drop=(c, i))}
            for c in range(delta2.shape[1]) for i in range(delta2.shape[2])]
    changes = np.asarray([row["budget_without"] - budget for row in rows])
    worst = rows[int(np.argmax(np.abs(changes)))]
    return {"budget": budget, "budget_min": min(r["budget_without"] for r in rows),
            "budget_max": max(r["budget_without"] for r in rows), "max_abs_change": float(np.abs(changes).max()),
            "most_influential_class": worst["class"], "most_influential_latent": worst["latent_index"],
            "rows": rows}

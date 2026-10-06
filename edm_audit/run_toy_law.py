"""One-dimensional law-level audit (appendix J, "One-dimensional mixture").

Nine denoisers (oracle, misspecified posterior mean, four perturbed oracles, three trained MLPs)
are run from the Gaussian start on the nested grids G_17 ... G_1088. For each grid and denoiser:
the per-step law quantities, the accumulated Lambda_K with its directional and field majorants,
the unrolled error recursion, the rates at fixed noise levels, and the re-summation at every
candidate threshold. The finest grid is repeated with 4M quantiles for a few arms.

    python -m edm_audit.run_toy_law [--smoke] [--out DIR]

Outputs: steps.csv, summary.csv, end_to_end.csv, refinement.csv, fixed_noise.csv,
threshold_sensitivity.csv, geometry.csv, summary.json, checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

from edm_audit.common import RUNS_DIR, edm_sigmas, write_csv, write_summary
from edm_audit.toy import (
    EDMPrecondMLP, Mixture1D, accumulate, audit_grid, decade_sums, fixed_noise_rows,
    high_noise_threshold, midpoint_levels, misspecified, mlp_denoiser, perturbed_oracle,
    recursion_decomposition, train_mlp, validation_losses,
)


CONFIG = {
    "sigma_data": 0.5,
    # The law: weights, centers rescaled to mean 0 and std sigma_data, common component width.
    "law": {"weights": [0.15, 0.35, 0.35, 0.15], "centers": [-3.0, -1.0, 1.0, 3.0], "width": 0.04},
    "misspec": {"weights": [0.2, 0.3, 0.3, 0.2], "center_scale": 1.02, "width_scale": 0.7},
    "perturbed_eps": [0.003, 0.01, 0.03, 0.1],
    "grid": {"sigma_max": 80.0, "sigma_min": 0.002, "rho_edm": 7.0, "K": [17, 34, 68, 136, 272, 544, 1088]},
    "M": 65536,                    # midpoint quantiles
    "check_factor": 4,             # the finest grid is repeated with check_factor * M quantiles ...
    "check_arms": ["oracle", "misspec", "perturbed0.01", "trained0"],   # ... for these arms
    "resolution_tolerance": 0.1,   # bound/error ratio reported only if |e(M) - e(4M)| < 0.1 e
    "fixed_sigmas": [0.04, 0.065, 0.09, 0.15],
    "b_hi": 0.5,                   # high-noise block: field margin b_pts >= b_hi
    "spatial": {"radius_sd": 12.0, "points": 4097, "check_radius_sd": 16.0, "check_points": 8193},
    "geometry": {"K": 272, "sigma": 0.2, "points": 1025},
    "trained": {"seeds": [0, 1, 2], "hidden": 128, "depth": 3, "steps": 20000, "batch_size": 4096,
                "lr": 1e-3, "ema": 0.999, "grad_clip": 1.0, "P_mean": -1.2, "P_std": 1.2},
}

SMOKE = {
    **CONFIG,
    "perturbed_eps": [0.01, 0.1],
    "grid": {**CONFIG["grid"], "K": [17, 34]},
    "M": 4096,
    "check_arms": None,            # every arm
    "fixed_sigmas": [0.04, 0.15],
    "spatial": {"radius_sd": 12.0, "points": 257, "check_radius_sd": 16.0, "check_points": 513},
    "geometry": {"K": 34, "sigma": 0.2, "points": 129},
    "trained": {**CONFIG["trained"], "seeds": [0], "steps": 200, "batch_size": 512},
}


def arm_name(arm: str, variant) -> str:
    return arm if variant is None else f"{arm}{variant:g}"


def load_or_train(mix: Mixture1D, seed: int, cfg: dict, sigma_data: float, directory: Path) -> EDMPrecondMLP:
    """Reuse a checkpoint only if it was trained on this law with these settings."""
    path, meta_path = directory / f"trained_seed{seed}.pt", directory / f"trained_seed{seed}.json"
    training = {k: cfg[k] for k in ("hidden", "depth", "steps", "batch_size", "lr", "ema", "grad_clip", "P_mean", "P_std")}
    meta = {"seed": seed, "sigma_data": sigma_data, "training": training,
            "law": {"weights": mix.weights.tolist(), "means": mix.means.tolist(), "width": mix.width}}
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        net = EDMPrecondMLP(sigma_data, cfg["hidden"], cfg["depth"]).double()
        net.load_state_dict(torch.load(path))
    else:
        net = train_mlp(mix, seed, sigma_data=sigma_data, **training)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return net.eval().requires_grad_(False)


def geometry_rows(mix: Mixture1D, cfg: dict, M: int) -> list:
    """Density, local expansion rate and directional error weight at one level (Figure hierarchy a).

    With lambda(x) = (D'(x) - 1) / sigma and w(x) = E[|z| 1{x in [V, U]}] / e^2 over the optimal
    coupling of the sampler law and mu_sigma, the directional rate is E_OT = int w lambda dx.
    """
    sigmas = edm_sigmas(cfg["geometry"]["K"], **{k: cfg["grid"][k] for k in ("sigma_max", "sigma_min", "rho_edm")})
    index = int(np.argmin(np.abs(sigmas[:-1] - cfg["geometry"]["sigma"])))
    sigma = float(sigmas[index])
    u = midpoint_levels(M)
    sampler = float(sigmas[0]) * ndtri(u)
    for j in range(index):                            # oracle Euler steps down to sigma
        a = (sigmas[j] - sigmas[j + 1]) / sigmas[j]
        sampler = np.sort((1.0 - a) * sampler + a * mix.posterior_mean(sampler, float(sigmas[j]))[0])
    target = mix.quantiles(u, sigma)
    radius, scale = cfg["spatial"]["radius_sd"], np.hypot(mix.width, sigma)
    x = np.linspace(np.min(mix.means) - radius * scale, np.max(mix.means) + radius * scale, cfg["geometry"]["points"])
    rate = (mix.posterior_mean(x, sigma)[1] - 1.0) / sigma
    # Each segment [sampler, target] of the 1D optimal coupling adds |z| / (M e^2) along its length.
    z = target - sampler
    lo = np.searchsorted(x, np.minimum(target, sampler), side="left").clip(0, len(x) - 1)
    hi = np.searchsorted(x, np.maximum(target, sampler), side="right").clip(0, len(x))
    counts = np.zeros(len(x) + 1)
    np.add.at(counts, lo, np.abs(z))
    np.add.at(counts, hi, -np.abs(z))
    weight = np.cumsum(counts[:-1]) / np.sum(z ** 2)
    return [{"sigma": sigma, "x": float(xi), "density": float(di), "rate": float(ri), "weight": float(wi)}
            for xi, di, ri, wi in zip(x, mix.pdf(x, sigma), rate, weight)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny settings that only exercise the code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else CONFIG
    out = args.out or RUNS_DIR / ("toy_law_smoke" if args.smoke else "toy_law")
    started = time.time()

    sd, M = cfg["sigma_data"], cfg["M"]
    grid = {k: cfg["grid"][k] for k in ("sigma_max", "sigma_min", "rho_edm")}
    mix = Mixture1D.standardized(cfg["law"]["weights"], cfg["law"]["centers"], cfg["law"]["width"], sd)
    wrong = misspecified(mix, **cfg["misspec"])

    # Every arm as ((arm, variant) -> denoiser).
    denoisers = {("oracle", None): mix.posterior_mean, ("misspec", None): wrong.posterior_mean}
    for eps in cfg["perturbed_eps"]:
        denoisers[("perturbed", float(eps))] = perturbed_oracle(mix, float(eps), sd)
    training = {}
    for seed in cfg["trained"]["seeds"]:
        net = load_or_train(mix, seed, cfg["trained"], sd, out / "checkpoints")
        training[f"seed{seed}"] = validation_losses(net, mix, cfg["trained"]["P_mean"], cfg["trained"]["P_std"])
        denoisers[("trained", seed)] = mlp_denoiser(net)

    # One threshold candidate per arm; the largest is used for all, so every curve covers the
    # same noise interval. The others serve for the threshold sensitivity.
    candidates = {key: high_noise_threshold(mix, den, grid["sigma_min"], grid["sigma_max"], sd,
                                            cfg["b_hi"], cfg["spatial"])
                  for key, den in denoisers.items()}
    sigma_hi = max(candidates.values())
    print(f"[toy_law] sigma_hi = {sigma_hi:.4f}")

    steps, summary, end_to_end, refinement, fixed_noise, sensitivity = [], [], [], [], [], []
    for K in cfg["grid"]["K"]:
        rows = audit_grid(denoisers, mix, edm_sigmas(K, **grid), M, sd, cfg["spatial"])
        for row in rows:
            row["arm"], row["variant"] = row["arm"]
        steps += rows
        for arm, variant in denoisers:
            run = [r for r in rows if (r["arm"], r["variant"]) == (arm, variant)]
            tag = {"arm": arm, "variant": variant, "K": K}
            acc = accumulate(run, sigma_hi)
            summary.append({**tag, "sigma_hi": sigma_hi, **acc})
            end_to_end.append({**tag, "initialization": "gaussian", **recursion_decomposition(run)})
            refinement.append({**tag, "Lambda": acc["Lambda"], "Lambda_dir": acc["Lambda_dir"],
                               "Lambda_field_pts": acc["Lambda_field_pts"],
                               "Lambda_ratio": acc["Lambda"] / acc["Lambda_field_pts"] if acc["Lambda_field_pts"] > 0 else float("nan"),
                               **decade_sums(run, sigma_hi)})
            fixed_noise += [{**tag, **entry} for entry in fixed_noise_rows(run, cfg["fixed_sigmas"])]
            for candidate in sorted(set(candidates.values())):
                resummed = accumulate(run, candidate)
                sensitivity.append({**tag, "threshold": candidate, "is_common": candidate == sigma_hi,
                                    **{k: resummed[k] for k in ("n_low", "Lambda", "Lambda_dir", "Lambda_field_pts")}})
        print(f"[toy_law] K={K} done")

    # Quadrature check: the finest grid again with check_factor * M quantiles.
    K = max(cfg["grid"]["K"])
    M_check = M * cfg["check_factor"]
    names = {arm_name(*key): key for key in denoisers}
    checked = {key: denoisers[key] for name, key in names.items()
               if cfg["check_arms"] is None or name in cfg["check_arms"]}
    fine_rows = audit_grid(checked, mix, edm_sigmas(K, **grid), M_check, sd, cfg["spatial"])
    quadrature_check = {}
    for key in checked:
        coarse = [r for r in steps if r["K"] == K and (r["arm"], r["variant"]) == key]
        fine = [r for r in fine_rows if r["arm"] == key]
        lam, lam_check = accumulate(coarse, sigma_hi)["Lambda"], accumulate(fine, sigma_hi)["Lambda"]
        err, err_check = (recursion_decomposition(rows)["final_error"] for rows in (coarse, fine))
        entry = {"K": K, "M": M, "M_check": M_check, "Lambda_M": lam, "Lambda_M_check": lam_check,
                 "rel_diff": abs(lam - lam_check) / max(abs(lam_check), 1e-300),
                 "final_error_M": err, "final_error_M_check": err_check,
                 "final_error_abs_diff": abs(err - err_check),
                 "final_error_rel_diff": abs(err - err_check) / max(err_check, 1e-300)}
        for c, f in zip(fixed_noise_rows(coarse, cfg["fixed_sigmas"]), fixed_noise_rows(fine, cfg["fixed_sigmas"])):
            entry[f"E_OT_rel_diff_{c['sigma_target']:g}"] = abs(c["E_OT"] - f["E_OT"]) / max(abs(f["E_OT"]), 1e-300)
        quadrature_check[arm_name(*key)] = entry

    # The bound/error ratio is reported only where the endpoint error is resolved by the quadrature.
    for row in end_to_end:
        check = quadrature_check.get(arm_name(row["arm"], row["variant"]))
        diff = check["final_error_abs_diff"] if check else float("nan")
        row["quadrature_abs_diff"] = diff
        row["resolved"] = bool(diff < cfg["resolution_tolerance"] * row["final_error"])
        row["bound_over_error"] = row["recursion_bound"] / row["final_error"] if row["resolved"] else float("nan")
    # Lambda_{2K} - Lambda_K, and the order of C_disc over each dyadic refinement.
    by_key = {(r["arm"], r["variant"], r["K"]): r for r in refinement}
    for row in refinement:
        finer = by_key.get((row["arm"], row["variant"], 2 * row["K"]))
        row["Lambda_increment"] = finer["Lambda"] - row["Lambda"] if finer else float("nan")
    by_key = {(r["arm"], r["variant"], r["K"]): r for r in end_to_end}
    for row in end_to_end:
        coarser = by_key.get((row["arm"], row["variant"], row["K"] // 2))
        row["order_propagated_discretization"] = (
            np.log2(coarser["propagated_discretization"] / row["propagated_discretization"])
            if coarser and row["K"] % 2 == 0 else float("nan"))

    for name, table in (("steps", steps), ("summary", summary), ("end_to_end", end_to_end),
                        ("refinement", refinement), ("fixed_noise", fixed_noise),
                        ("threshold_sensitivity", sensitivity), ("geometry", geometry_rows(mix, cfg, M))):
        write_csv(out / f"{name}.csv", table)
    write_summary(out, "toy_law", cfg, {
        "law": {"weights": mix.weights, "means": mix.means, "width": mix.width},
        "misspec_law": {"weights": wrong.weights, "means": wrong.means, "width": wrong.width},
        "training": training,
        "sigma_hi": sigma_hi,
        "sigma_hi_candidates": {arm_name(*key): value for key, value in candidates.items()},
        "quadrature_check": quadrature_check,
    }, started)


if __name__ == "__main__":
    main()

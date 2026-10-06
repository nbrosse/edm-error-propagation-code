"""Synchronous coupling and segment rates of the CIFAR-10 EDM network (appendix J, low noise).

For each pair G_K / G_2K and each class, the same 128 latents are integrated with Euler on both
grids. Two calculations:

* synchronous rates at every level, for every trajectory (cheap: they reuse the denoiser outputs
  of the sampler), accumulated over sigma <= 2 into the budget sum_j ell_j (E_sync_j)_+, with the
  influence of each trajectory measured by removing it;
* segment rates (lambda_par, lambda_rand, lambda_max; expensive) on K = 68, 272 only, at eight
  prescribed levels, for the first eight trajectories of each class, pooled with energy weights
  and a class-stratified bootstrap.

    python -m edm_audit.run_transport [--smoke] [--out DIR]

Outputs: synchronous_transport.csv, segment_max.csv, segment_levels.csv, budget.csv,
influence.csv, moments/class_{c}.npz (per-trajectory moments), segment_samples/class_{c}.npz.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from edm_audit.cifar import (
    NETWORK_URL, bootstrap_indices, class_latents, energy_weighted_bootstrap, energy_weighted_summary,
    integrate, load_network, secant_from_moments, segment_levels, segment_step_metrics,
    synchronous_step_metrics, trajectory_influence,
)
from edm_audit.common import RUNS_DIR, write_csv, write_summary


CONFIG = {
    "network": NETWORK_URL,
    "device": "cuda",
    "seed": 1234,
    "batch_size": 16,
    "grid": {"sigma_max": 80.0, "sigma_min": 0.002, "rho_edm": 7.0},
    "K": [17, 34, 68, 136, 272],      # each paired with 2K
    "num_trajectories": 128,          # per class, the same latents on every grid
    "classes": list(range(10)),
    "window": 2.0,                    # the synchronous budget sums the levels sigma <= window
    "segment": {
        "K": [68, 272],
        "levels": [0.004, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0],
        "num_trajectories_per_class": 8,
        "t_nodes": 9,                 # nested 3 / 5 / 9 nodes
        "lanczos_iters": 32,
        "bootstrap_replicates": 1000,
        "random_seed": 20260922,      # of the random-direction baseline
    },
}

SMOKE = {
    **CONFIG,
    "network": None,                  # synthetic offline stand-in
    "device": "cpu",
    "batch_size": 2,
    "K": [4, 8],
    "num_trajectories": 4,
    "classes": [0, 1],
    "segment": {**CONFIG["segment"], "K": [4, 8], "levels": [0.1, 3.0], "num_trajectories_per_class": 2,
                "lanczos_iters": 4, "bootstrap_replicates": 20},
}

SEGMENT_RATES = [f"lambda_{kind}_{n}" for kind in ("par", "rand", "max") for n in (3, 5, 9)]


def class_average(rows: list) -> dict:
    """Pool the classes of one step through their raw moments (never by averaging ratios)."""
    moments = {k: float(np.mean([r[k] for r in rows]))
               for k in ("mean_delta2", "mean_deltaD2", "mean_phi2", "mean_cross", "mean_res2")}
    defined = moments["mean_delta2"] > 0.0
    base = rows[0]
    return {"class": "all", **{k: base[k] for k in ("K", "step_index", "sigma_j", "sigma_next", "ell_j", "a_j")},
            "ratio_defined": defined, **moments,
            "A_sync": (moments["mean_phi2"] / moments["mean_delta2"]) ** 0.5 if defined else float("nan"),
            **secant_from_moments(moments["mean_delta2"], moments["mean_cross"], moments["mean_res2"],
                                  base["sigma_j"], "sync")}


def segment_row(class_id, K: int, sync_row: dict, moments: dict, segment: dict, ids: list,
                boot: torch.Tensor = None) -> dict:
    """Energy-weighted rates of one cell (class or pooled, K, level) and their diagnostics."""
    summaries = {name: energy_weighted_summary(moments["delta2"], segment[name], ids)
                 for name in SEGMENT_RATES + ["lambda_par_integral"]}
    par = summaries["lambda_par_9"]
    row = {
        "class": class_id, "K": K, **{k: sync_row[k] for k in ("step_index", "sigma_j", "sigma_next", "ell_j")},
        "n_segments": int(moments["delta2"].numel()),
        **secant_from_moments(float(moments["delta2"].mean()), float(moments["cross"].mean()),
                              float(moments["res2"].mean()), sync_row["sigma_j"], "sync_subset"),
        **{name: summaries[name]["weighted_mean"] for name in SEGMENT_RATES + ["lambda_par_integral"]},
        "ess": par["ess"], "max_weight": par["max_weight"],
        "lambda_par_9_loo_max_abs_change": par["loo_max_abs_change"],
        "lambda_max_9_loo_max_abs_change": summaries["lambda_max_9"]["loo_max_abs_change"],
        "n_zero": par["n_zero"], "n_invalid": par["n_invalid"], "invalid_ids": par["invalid_ids"],
        "valid": all(summaries[name]["valid"] for name in SEGMENT_RATES),
        "lanczos_half_drift": float(segment["lanczos_half_drift"].max()),
        "lanczos_residual": float(segment["lanczos_residual"].max()),
    }
    # The gaps of eq:segment_aggregate_chain; the integral of lambda_par must equal E_sync.
    row["gap_sync_to_directional"] = row["lambda_par_9"] - row["E_sync_subset"]
    row["gap_directional_to_segment"] = row["lambda_max_9"] - row["lambda_par_9"]
    row["gap_directional_to_random"] = row["lambda_par_9"] - row["lambda_rand_9"]
    row["secant_integral_gap"] = row["lambda_par_integral"] - row["E_sync_subset"]
    if boot is not None:
        for name in ("lambda_par_9", "lambda_rand_9", "lambda_max_9"):
            interval = energy_weighted_bootstrap(moments["delta2"], segment[name], boot)
            row[f"{name}_boot_lo"], row[f"{name}_boot_hi"] = interval["boot_lo"], interval["boot_hi"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="synthetic network on the CPU: only exercises the code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else CONFIG
    out = args.out or RUNS_DIR / ("transport_smoke" if args.smoke else "transport")
    started = time.time()

    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    net = load_network(cfg["network"], device)
    seg = cfg["segment"]
    take = seg["num_trajectories_per_class"]
    # Levels to record on each run: all of G_K for the coarse run, the even ones of G_2K.
    record = {}
    for K in cfg["K"]:
        record.setdefault(K, set()).update(range(K + 1))
        record.setdefault(2 * K, set()).update(range(0, 2 * K + 1, 2))

    sync_rows, segment_rows, level_rows = [], [], []
    by_step, pooled = {}, {}
    ensemble = {K: {"delta2": [], "cross": [], "res2": []} for K in cfg["K"]}
    window_levels = {}
    for class_id in cfg["classes"]:
        latents, labels = class_latents(net, class_id, cfg["num_trajectories"], cfg["seed"], device)
        runs, moments_file, samples_file = {}, {}, {}
        for K in cfg["K"]:
            t0 = time.time()
            for n in (K, 2 * K):
                if n not in runs:
                    runs[n] = integrate(net, latents, labels, n, cfg["grid"], sorted(record[n]), cfg["batch_size"])
            coarse, fine = runs[K], runs[2 * K]
            assert all(fine.sigmas[2 * j] == coarse.sigmas[j] for j in range(K + 1)), "G_K is not nested in G_2K"
            rows, per_sample = synchronous_step_metrics(coarse, fine, K)
            for row in rows:
                row.update({"class": class_id, "K": K})
                sync_rows.append(row)
                by_step.setdefault((K, row["step_index"]), []).append(row)

            # Per-trajectory moments over the budget window, kept for the influence analysis.
            steps = [j for j, row in enumerate(rows) if row["sigma_j"] <= cfg["window"]]
            for name in ("delta2", "deltaD2", "cross", "res2"):
                moments_file[f"K{K}_{name}"] = np.stack([per_sample[j][name].numpy() for j in steps])
            for name in ("delta2", "cross", "res2"):
                ensemble[K][name].append(moments_file[f"K{K}_{name}"])
            window_levels[K] = (np.array([rows[j]["sigma_j"] for j in steps]), np.array([rows[j]["ell_j"] for j in steps]))
            moments_file[f"K{K}_sigma_j"], moments_file[f"K{K}_ell_j"] = window_levels[K]
            moments_file["latent_index"] = np.arange(cfg["num_trajectories"])
            moments_file["seed"] = np.asarray([cfg["seed"] + class_id])

            if K in seg["K"]:
                chosen = segment_levels(coarse.sigmas, seg["levels"])
                if class_id == cfg["classes"][0]:
                    level_rows += [{"K": K, **row} for row in chosen]
                for j in sorted({row["step_index"] for row in chosen}):
                    segment = segment_step_metrics(
                        net, coarse.states[j][:take], fine.states[2 * j][:take], rows[j]["sigma_j"],
                        None if labels is None else labels[:take], seg["t_nodes"], seg["lanczos_iters"], seg["random_seed"])
                    moments = {name: per_sample[j][name][:take] for name in ("delta2", "cross", "res2")}
                    ids = [f"c{class_id}:l{i}" for i in range(take)]
                    segment_rows.append(segment_row(class_id, K, rows[j], moments, segment, ids))
                    cell = pooled.setdefault((K, j), {"row": rows[j], "chunks": [], "ids": [], "classes": []})
                    cell["chunks"].append({**segment, **moments})
                    cell["ids"] += ids
                    cell["classes"] += [class_id] * take
                    samples_file.update({f"K{K}_j{j}_{name}": value.numpy() for name, value in {**segment, **moments}.items()})
            # Free the coarse run once no later grid needs it.
            runs = {n: run for n, run in runs.items() if n == 2 * K}
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[transport] class={class_id} K={K}: {time.time() - t0:.1f} s", flush=True)

        for name, arrays in (("moments", moments_file), ("segment_samples", samples_file)):
            if arrays:
                (out / name).mkdir(parents=True, exist_ok=True)
                np.savez(out / name / f"class_{class_id}.npz", **arrays)
        write_csv(out / "synchronous_transport.csv", sync_rows)
        write_csv(out / "segment_max.csv", segment_rows)

    # Pooled over classes: the synchronous table, then the segment cells with the bootstrap.
    if len(cfg["classes"]) > 1:
        sync_rows += [class_average(rows) for _, rows in sorted(by_step.items())]
    for (K, j), cell in sorted(pooled.items()):
        combined = {name: torch.cat([chunk[name] for chunk in cell["chunks"]]) for name in cell["chunks"][0]}
        boot = bootstrap_indices(cell["classes"], seg["bootstrap_replicates"], cfg["seed"] + 100000 * K + j)
        segment_rows.append(segment_row("all", K, cell["row"], combined, combined, cell["ids"], boot))

    # The accumulated synchronous budget of the whole ensemble, and the influence of each trajectory.
    budget_rows, influence_rows = [], []
    for K in cfg["K"]:
        delta2, cross, res2 = (np.stack(ensemble[K][name], axis=1) for name in ("delta2", "cross", "res2"))
        sigmas, ell = window_levels[K]
        report = trajectory_influence(delta2, cross, sigmas, ell, cfg["classes"])
        # The quadratic term 1/2 sum_j ell_j^2 Q_sync_j^2, with classes pooled through their moments.
        Q2 = res2.mean(axis=2).mean(axis=1) / np.maximum(delta2.mean(axis=2).mean(axis=1), 1e-300) / sigmas ** 2
        budget_rows.append({"K": K, "n_steps": len(sigmas), "n_trajectories": delta2.shape[1] * delta2.shape[2],
                            "budget_linear": report["budget"], "budget_quadratic": float(np.sum(0.5 * ell ** 2 * Q2)),
                            "budget_min": report["budget_min"],
                            "budget_max": report["budget_max"], "max_abs_change": report["max_abs_change"],
                            "most_influential_class": report["most_influential_class"],
                            "most_influential_latent": report["most_influential_latent"]})
        influence_rows += [{"K": K, **row, "change": row["budget_without"] - report["budget"]} for row in report["rows"]]

    for name, table in (("synchronous_transport", sync_rows), ("segment_max", segment_rows),
                        ("segment_levels", level_rows), ("budget", budget_rows), ("influence", influence_rows)):
        write_csv(out / f"{name}.csv", table)
    write_summary(out, "transport", cfg, {"device": str(device), "budgets": budget_rows}, started)


if __name__ == "__main__":
    main()

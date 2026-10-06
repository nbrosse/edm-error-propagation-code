"""Node resolution of the K = 272 segments of ``run_transport`` (appendix J, reliability table).

The trapezoidal integral of lambda_par over a segment must equal its secant rate
(<zeta, D(y) - D(x)> / |zeta|^2 - 1) / sigma. On K = 272 the 9-node integral misses it at four
levels. This rebuilds the same segments (same seed, latents and first sampler batch) and
evaluates lambda_par on 65 nodes, nested over the 9 reference nodes.

    python -m edm_audit.run_secant_recheck [--smoke] [--out DIR]

Outputs: secant_segments.csv (per segment), secant_cells.csv (energy-weighted, per class and
pooled), secant_profiles.npz (lambda_par on every node), summary.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from edm_audit.cifar import (
    NETWORK_URL, class_latents, directional_profile, energy_weighted_summary, integrate, load_network,
    nested_counts, pair_moments, secant_from_moments, segment_levels,
)
from edm_audit.common import RUNS_DIR, write_csv, write_summary


CONFIG = {
    "network": NETWORK_URL,
    "device": "cuda",
    "seed": 1234,                   # seed, num_trajectories and batch_size must match run_transport:
    "num_trajectories": 128,        # they fix the latents and the batch the states were integrated in
    "batch_size": 16,
    "grid": {"sigma_max": 80.0, "sigma_min": 0.002, "rho_edm": 7.0},
    "classes": list(range(10)),
    "K": 272,
    "levels": [0.004, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0],
    "num_trajectories_per_class": 8,
    "t_nodes": 65,
}

SMOKE = {**CONFIG, "network": None, "device": "cpu", "num_trajectories": 4, "batch_size": 2,
         "classes": [0, 1], "K": 8, "levels": [0.1, 3.0], "num_trajectories_per_class": 2, "t_nodes": 17}


def simpson(values: torch.Tensor) -> torch.Tensor:
    """Composite Simpson rule on [0, 1] over an odd number of equispaced nodes (last axis)."""
    n = values.shape[1]
    weights = torch.ones(n, dtype=values.dtype)
    weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
    return (values * weights).sum(dim=1) / (3.0 * (n - 1))


def cell_row(class_id, K, j, sigma, delta2, cross, res2, per, ids, counts) -> dict:
    row = {"class": class_id, "K": K, "step_index": j, "sigma_j": sigma, "n_segments": int(delta2.numel()),
           **secant_from_moments(float(delta2.mean()), float(cross.mean()), float(res2.mean()), sigma, "sync_subset")}
    exact = energy_weighted_summary(delta2, per["exact"], ids)
    row.update(exact_weighted=exact["weighted_mean"], ess=exact["ess"], max_weight=exact["max_weight"])
    for n in counts:
        for name in ("trap", "simpson", "max"):
            row[f"{name}_{n}"] = energy_weighted_summary(delta2, per[f"{name}_{n}"], ids)["weighted_mean"]
        row[f"trap_gap_{n}"] = row[f"trap_{n}"] - row["E_sync_subset"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="synthetic network on the CPU: only exercises the code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else CONFIG
    out = args.out or RUNS_DIR / ("secant_recheck_smoke" if args.smoke else "secant_recheck")
    started = time.time()

    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    net = load_network(cfg["network"], device)
    K, t_nodes, take, batch = cfg["K"], cfg["t_nodes"], cfg["num_trajectories_per_class"], cfg["batch_size"]
    counts = [n for n in nested_counts(t_nodes) if n >= 9]
    segment_rows, cell_rows, profiles, pooled = [], [], {}, {}
    for class_id in cfg["classes"]:
        latents, labels = class_latents(net, class_id, cfg["num_trajectories"], cfg["seed"], device)
        latents, labels = latents[:batch], None if labels is None else labels[:batch]   # the first batch only
        coarse = integrate(net, latents, labels, K, cfg["grid"], range(K + 1), batch)
        steps = sorted({row["step_index"] for row in segment_levels(coarse.sigmas, cfg["levels"])})
        fine = integrate(net, latents, labels, 2 * K, cfg["grid"], [2 * j for j in steps], batch)
        for j in steps:
            sigma = coarse.sigmas[j]
            x, y = coarse.states[j][:take], fine.states[2 * j][:take]
            delta2, cross, res2 = (m.cpu() for m in pair_moments(y - x, fine.denoised[2 * j][:take] - coarse.denoised[j][:take]))
            profile = directional_profile(net, x, y, sigma, None if labels is None else labels[:take], t_nodes)
            profiles[f"c{class_id}_j{j}"] = profile.numpy()
            per = {"exact": (cross / delta2 - 1.0) / sigma}      # the secant rate, by the FTC
            for n in counts:
                sub = profile[:, ::(t_nodes - 1) // (n - 1)]
                per[f"trap_{n}"] = torch.trapz(sub, dx=1.0 / (n - 1), dim=1)
                per[f"simpson_{n}"] = simpson(sub)
                per[f"max_{n}"] = sub.nan_to_num(nan=-np.inf).max(dim=1).values
            for i in range(take):
                segment_rows.append({"class": class_id, "latent": i, "K": K, "step_index": j, "sigma_j": sigma,
                                     "delta2": float(delta2[i]), "weight": float(delta2[i] / delta2.sum()),
                                     **{name: float(values[i]) for name, values in per.items()}})
            ids = [f"c{class_id}:l{i}" for i in range(take)]
            cell_rows.append(cell_row(class_id, K, j, sigma, delta2, cross, res2, per, ids, counts))
            store = pooled.setdefault(j, {"sigma": sigma, "ids": [], "values": []})
            store["ids"] += ids
            store["values"].append({"delta2": delta2, "cross": cross, "res2": res2, **per})
        print(f"[secant recheck] class={class_id} done", flush=True)

    for j, store in sorted(pooled.items()):
        merged = {name: torch.cat([v[name] for v in store["values"]]) for name in store["values"][0]}
        cell_rows.append(cell_row("all", K, j, store["sigma"], merged["delta2"], merged["cross"], merged["res2"],
                                  merged, store["ids"], counts))
    write_csv(out / "secant_segments.csv", segment_rows)
    write_csv(out / "secant_cells.csv", cell_rows)
    np.savez(out / "secant_profiles.npz", **profiles)
    write_summary(out, "secant_recheck", cfg, {"nested_counts": counts}, started)


if __name__ == "__main__":
    main()

"""High-noise diagnostics of the pretrained CIFAR-10 EDM network (appendix J, "CIFAR-10, high noise").

All parts use one fixed bank of held-out probes x0 + sigma z:

* margin: the local stretch S_F = ||grad F||_op of the raw branch at c_in x (power iteration,
  two random starts) and the probe margin b_probe = 1 - c_skip - c_out c_in S_F, on the levels
  of G_136 with sigma >= 0.5;
* residual: r_reg, the RMS of the normalized EDM regression residual, on every level of G_272;
* convergence: the stretch at 8, 16 and 32 power iterations, per random start, at the levels of
  G_136 that bracket the two crossings of the q90 margin (b = 0 and b = 6/7) plus eight more.

    python -m edm_audit.run_stability [--smoke] [--parts margin residual convergence] [--out DIR]

Outputs: large_noise_margin.csv, regression_residual.csv, margin_convergence.csv,
margin_convergence_probes.csv, summary.json.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from edm_audit.cifar import NETWORK_URL, call_F_branch, load_network, load_probe_bank, power_iteration
from edm_audit.common import RUNS_DIR, edm_coeffs, edm_sigmas, nearest_in_log, stats, write_csv, write_summary


CONFIG = {
    "device": "cuda",
    "seed": 1234,                  # with batch_size, fixes the probe bank
    "batch_size": 8,
    "num_probes": 128,
    "grid": {"sigma_max": 80.0, "sigma_min": 0.002, "rho_edm": 7.0},
    "margin": {"K": 136, "sigma_min": 0.5, "power_iters": 8, "random_starts": 2},
    "residual": {"K": 272},
    "convergence": {
        "K": 136,
        "iters": [8, 16, 32],
        "random_starts": 2,
        "levels": [9.376974006210345, 8.877462928241497, 8.400935309099815,      # around b_q90 = 0
                   42.62276147499661, 40.78557379650796, 39.01668479354711,      # around b_q90 = 6/7
                   60.05296637046716, 29.72207616356115, 20.324220585062683, 12.25238611881397,
                   7.946503042423345, 5.976588892004314, 3.9293449058789607],
    },
}

SMOKE = {
    **CONFIG,
    "device": "cpu",
    "batch_size": 1,
    "num_probes": 2,
    "margin": {"K": 1, "sigma_min": 0.5, "power_iters": 1, "random_starts": 1},
    "residual": {"K": 1},
    "convergence": {"K": 1, "iters": [1, 2], "random_starts": 2, "levels": [80.0]},
}

QUANTILES = ["mean", "q50", "q75", "q90", "q95", "q99", "max"]


def margin_rows(net, probes, sigmas, cfg, sigma_data) -> list:
    rows = []
    for sigma in sigmas:
        started = time.time()
        c_skip, c_out, c_in = edm_coeffs(sigma, sigma_data)
        stretch = []
        for _, x, labels in probes.batches(sigma):
            f = lambda z: call_F_branch(net, z, sigma, labels)
            estimates = power_iteration(f, c_in * x, cfg["power_iters"], cfg["random_starts"])
            stretch += estimates[cfg["power_iters"]].max(dim=0).values.tolist()
        row = {"sigma": sigma, "seconds": time.time() - started, **stats(stretch, "SF")}
        for q in QUANTILES:   # the margin induced by each quantile of the stretch
            row[f"b_probe_{q}"] = 1.0 - c_skip - c_out * c_in * row[f"SF_{q}"]
            row[f"SF_over_threshold_{q}"] = row[f"SF_{q}"] / (sigma / sigma_data)
        rows.append(row)
        print(f"[margin] sigma={sigma:.6g} b_probe_q90={row['b_probe_q90']:.4g}", flush=True)
    return rows


def residual_rows(net, probes, sigmas, sigma_data) -> list:
    dim = net.img_channels * net.img_resolution ** 2
    rows = []
    for sigma in sigmas:
        started = time.time()
        c_skip, c_out, c_in = edm_coeffs(sigma, sigma_data)
        sq = []
        for x0, x, labels in probes.batches(sigma):
            target = (x0 - c_skip * x) / c_out          # the regression target of F
            with torch.no_grad():
                sq += (call_F_branch(net, c_in * x, sigma, labels) - target).flatten(1).square().sum(1).tolist()
        sq = np.asarray(sq, dtype=np.float32)
        rows.append({"sigma": sigma, "sigma_data": sigma_data, "n_samples": sq.size, "dim": dim,
                     "R_reg_l2": math.sqrt(float(sq.mean())), "r_reg": math.sqrt(float(sq.mean()) / dim),
                     "normalized_mse_per_dim": float(sq.mean()) / dim, "normalized_mse_sum": float(sq.mean()),
                     "seconds": time.time() - started,
                     **stats(np.sqrt(sq), "R_sample_l2"), **stats(np.sqrt(sq / dim), "r_sample")})
        print(f"[residual] sigma={sigma:.6g} r_reg={rows[-1]['r_reg']:.4g}", flush=True)
    return rows


def convergence_rows(net, probes, sigmas, cfg, sigma_data):
    """Margin statistics per iteration count (max over starts), and every probe and start."""
    iters, starts = sorted(cfg["iters"]), cfg["random_starts"]
    summary, per_probe = [], []
    for sigma in sigmas:
        started = time.time()
        c_skip, c_out, c_in = edm_coeffs(sigma, sigma_data)
        best = {n: [] for n in iters}
        index = 0
        for _, x, labels in probes.batches(sigma):
            f = lambda z: call_F_branch(net, z, sigma, labels)
            estimates = power_iteration(f, c_in * x, iters[-1], starts, record=iters)
            for n in iters:
                best[n] += estimates[n].max(dim=0).values.tolist()
            for i in range(x.shape[0]):
                for s in range(starts):
                    row = {"sigma": sigma, "probe_index": index + i, "start": s}
                    for n in iters:
                        row[f"SF_iter{n}"] = float(estimates[n][s, i])
                        row[f"b_probe_iter{n}"] = 1.0 - c_skip - c_out * c_in * row[f"SF_iter{n}"]
                    per_probe.append(row)
            index += x.shape[0]
        for n in iters:
            row = {"sigma": sigma, "power_iters": n, "random_starts": starts,
                   "seconds_level": time.time() - started, **stats(best[n], "SF")}
            for q in QUANTILES:
                row[f"b_probe_{q}"] = 1.0 - c_skip - c_out * c_in * row[f"SF_{q}"]
            summary.append(row)
        print(f"[convergence] sigma={sigma:.6g} b_probe_q90 "
              + " ".join(f"it{r['power_iters']}={r['b_probe_q90']:+.5f}" for r in summary[-len(iters):]), flush=True)
    return summary, per_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="CPU, two probes: only exercises the code")
    parser.add_argument("--parts", nargs="+", default=["margin", "residual", "convergence"],
                        choices=["margin", "residual", "convergence"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else CONFIG
    out = args.out or RUNS_DIR / ("stability_smoke" if args.smoke else "stability")
    started = time.time()

    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["seed"])
    net = load_network(NETWORK_URL, device)
    sigma_data = float(net.sigma_data)
    probes = load_probe_bank(net, cfg["num_probes"], cfg["batch_size"], device, cfg["seed"])
    levels = lambda K: [float(s) for s in edm_sigmas(K, **cfg["grid"])]

    if "margin" in args.parts:
        sigmas = [s for s in levels(cfg["margin"]["K"]) if s >= cfg["margin"]["sigma_min"]]
        write_csv(out / "large_noise_margin.csv", margin_rows(net, probes, sigmas, cfg["margin"], sigma_data))
    if "residual" in args.parts:
        write_csv(out / "regression_residual.csv", residual_rows(net, probes, levels(cfg["residual"]["K"]), sigma_data))
    if "convergence" in args.parts:
        grid = levels(cfg["convergence"]["K"])
        sigmas = sorted({grid[nearest_in_log(grid, s)] for s in cfg["convergence"]["levels"]}, reverse=True)
        summary, per_probe = convergence_rows(net, probes, sigmas, cfg["convergence"], sigma_data)
        write_csv(out / "margin_convergence.csv", summary)
        write_csv(out / "margin_convergence_probes.csv", per_probe)
    write_summary(out, "stability", cfg, {"parts": args.parts, "sigma_data": sigma_data,
                                          "num_probes": len(probes)}, started)


if __name__ == "__main__":
    main()

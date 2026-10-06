"""Oracle from the exact start on nested grids: error, amplification and their refinement orders.

Reads the law, schedule and common threshold from a finished ``run_toy_law`` campaign. The
nested grids share the target quantiles but each has its own Euler state. Orders compare
adjacent grids at the same quadrature M (main figure, panels a-c; paired-order table).

    python -m edm_audit.run_oracle_refinement --M 65536 --out DIR
    python -m edm_audit.run_oracle_refinement --M 262144 --K 272 544 1088 --out DIR

Outputs: summary.csv, steps.csv, orders.csv, metadata.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy

from edm_audit.common import CODE_DIR, RESULTS_DIR, edm_sigmas, write_csv
from edm_audit.toy import (
    Mixture1D, log_amplification, midpoint_levels, recursion_decomposition, step_metrics,
)


def nested_oracle_rows(mix: Mixture1D, sizes: Sequence[int], M: int, grid: Dict, sigma_hi: float,
                       sigma_data: float = 0.5):
    """Exact-start oracle on every grid of ``sizes`` (divisors of the finest), in one sweep."""
    sizes = sorted(set(int(k) for k in sizes))
    finest = sizes[-1]
    if any(finest % k for k in sizes):
        raise ValueError("step counts must divide the finest grid")
    sigmas = edm_sigmas(finest, grid["sigma_max"], grid["sigma_min"], grid["rho_edm"])
    u = midpoint_levels(M)
    target = mix.quantiles(u, float(sigmas[0]))
    states = {k: target.copy() for k in sizes}
    previous = {k: target for k in sizes}
    rows = {k: [] for k in sizes}
    for i in range(1, finest + 1):
        target = mix.quantiles(u, float(sigmas[i]))
        for k in sizes:
            stride = finest // k
            if i % stride == 0:
                row, states[k] = step_metrics(previous[k], states[k], mix.posterior_mean, mix,
                                              float(sigmas[i - stride]), float(sigmas[i]), sigma_data, target)
                rows[k].append({"K": k, "M": M, "initialization": "exact", "step_index": i // stride - 1, **row})
                previous[k] = target
        if i % 64 == 0 or i == finest:
            print(f"[oracle refinement] M={M}, level {i}/{finest}", flush=True)
    summary = [{
        "K": k, "M": M, "initialization": "exact", "sigma_hi": sigma_hi,
        "Lambda": log_amplification([r for r in rows[k] if r["sigma_next"] < sigma_hi]),
        "sum_defect_measured": sum(r["discretization_term"] for r in rows[k]),
        "sum_defect_universal": sum(r["Delta_j"] for r in rows[k]),
        **recursion_decomposition(rows[k]),
    } for k in sizes]
    return summary, [row for k in sizes for row in rows[k]]


def refinement_orders(summary: List[Dict]) -> List[Dict]:
    """Local orders log2(C_{K/2} / C_K) over each pair (K/2, K) computed with the same M."""
    by_M: Dict[int, Dict[int, Dict]] = {}
    for row in summary:
        by_M.setdefault(int(row["M"]), {})[int(row["K"])] = row
    out = []
    for M, group in sorted(by_M.items()):
        for k, row in sorted(group.items()):
            prev = group.get(k // 2) if k % 2 == 0 else None
            if prev is None:
                continue
            out.append({
                "K": k, "K_coarse": k // 2, "M": M, "initialization": "exact",
                "order_error": math.log2(prev["final_error"] / row["final_error"]),
                "order_measured_bound": math.log2(prev["propagated_discretization"] / row["propagated_discretization"]),
                "order_local_defects": math.log2(prev["sum_defect_measured"] / row["sum_defect_measured"]),
                # The order that K^-1 exp(Lambda_K) would have with the measured budget.
                "order_amplification_scale": 1 - (row["Lambda"] - prev["Lambda"]) / math.log(2),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=RESULTS_DIR / "toy_law_2026-09-22")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, nargs="+", help="grids (default: those of the source campaign)")
    args = parser.parse_args()

    source = args.source_dir / "summary.json"
    campaign = json.loads(source.read_text())
    cfg, sigma_hi = campaign["config"], campaign["results"]["sigma_hi"]
    law = cfg["law"]
    mix = Mixture1D.standardized(law["weights"], law["centers"], law["width"], cfg["sigma_data"])
    sizes = args.K or cfg["grid"]["K"]
    summary, steps = nested_oracle_rows(mix, sizes, args.M, cfg["grid"], sigma_hi, cfg["sigma_data"])
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "steps.csv", steps)
    write_csv(args.out / "orders.csv", refinement_orders(summary))

    package = Path(__file__).resolve().parent
    inputs = [source, Path(__file__), package / "toy.py", package / "common.py"]
    metadata = {
        "experiment": "oracle_refinement", "initialization": "exact", "M": args.M, "K": sizes,
        "sigma_hi": sigma_hi, "grid": cfg["grid"], "law": law, "sigma_data": cfg["sigma_data"],
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "inputs_sha256": {str(p.resolve().relative_to(CODE_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in inputs},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

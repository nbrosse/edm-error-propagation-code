"""Numbers the manuscript prose reads from the audit runs.

Writes to generated/data/: audit_macros.tex (margin crossings, residual range, and the
one-dimensional numbers quoted in section 9), oracle_refinement_checks.tex (the paired-order table),
toy_refinement_table.tex (error, defects, amplification and orders under refinement),
toy_hierarchy_table.tex (Lambda_K and its majorants by denoiser) and PROVENANCE.json (hashes of the
inputs). ``--check`` fails if any of these files is stale.

    python -m edm_audit.figure_data [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from edm_audit.common import CODE_DIR, RESULTS_DIR, read_csv as rows
from edm_audit.figures import OMEGA_FIT_MIN_K, SEGMENT_GRID, TOY_DIR, TRANSPORT_DIR, top_block
from edm_audit.figures import ORACLE_REFINEMENT_DIR as ORACLE_DIR
from edm_audit.figures import synchronous_budgets_from_moments
from edm_audit.toy import B_D1, B_D1_FROZEN_RUNS

# The frozen oracle runs stored Delta_j with B_D1_FROZEN_RUNS; Delta_j is linear in B_{1,1}, so their
# universal quantities are rescaled to the current constant. Set to 1 after rerunning with B_D1.
UNIVERSAL_RESCALE = B_D1 / B_D1_FROZEN_RUNS

# Denoisers of the hierarchy table, in display order: (arm, variant, label).
HIERARCHY_ARMS = (
    ("oracle", "", "oracle"),
    ("trained", "0", "trained, seed 0"),
    ("trained", "1", "trained, seed 1"),
    ("trained", "2", "trained, seed 2"),
    ("misspec", "", "misspecified mean"),
)
DASH = r"\multicolumn{1}{c}{--}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def threshold(data, column: str, level: float) -> float:
    return top_block([float(row["sigma"]) for row in data], [float(row[column]) for row in data], level)


def oracle_rows(results: Path):
    """Exact-start oracle refinement: summary rows sorted by K, and pair orders keyed by the finer K."""
    summary = sorted(rows(results / ORACLE_DIR / "main/summary.csv"), key=lambda row: int(row["K"]))
    orders = {int(row["K"]): row for row in rows(results / ORACLE_DIR / "main/orders.csv")}
    if any(row["initialization"] != "exact" for row in summary):
        raise ValueError("the refinement table requires exact initialization")
    return summary, orders


def toy_macros(results: Path) -> dict:
    """The one-dimensional numbers the prose of section 9 quotes."""
    summary, orders = oracle_rows(results)
    K = [int(row["K"]) for row in summary]
    fine = [row for row in summary if int(row["K"]) >= OMEGA_FIT_MIN_K]
    # Same fit as the slope drawn in the toy figure.
    omega, _ = np.polyfit(np.log([int(row["K"]) for row in fine]), [float(row["Lambda"]) for row in fine], 1)
    error_orders = [float(row["order_error"]) for row in orders.values()]
    excess = [float(row["recursion_bound"]) / float(row["final_error"]) - 1.0 for row in summary]
    last = summary[-1]
    toy = {(row["arm"], row["variant"], int(row["K"])): row
           for row in rows(results / TOY_DIR / "summary.csv")}

    def ratio(arm: str, variant: str, k: int, key: str) -> float:
        row = toy[(arm, variant, k)]
        return float(row[key]) / float(row["Lambda"])

    dir_excess = max(ratio(arm, variant, K[-1], "Lambda_dir") - 1.0
                     for arm, variant, _ in HIERARCHY_ARMS)
    # Gap between the observed and amplification-based orders on the two finest pairs.
    finest = [orders[k] for k in sorted(orders)[-2:]]
    order_gap = max(abs(float(row["order_error"]) - float(row["order_amplification_scale"])) for row in finest)
    gaussian = {int(row["K"]): row for row in rows(results / TOY_DIR / "end_to_end.csv")
                if row["arm"] == "oracle" and row["initialization"] == "gaussian"}
    return {
        "resKmin": f"{K[0]}",
        "resKmax": f"{K[-1]}",
        "resErrorCoarse": sci(float(summary[0]['final_error'])),
        "resErrorFine": sci(float(last['final_error'])),
        "resErrorOrderMin": f"{min(error_orders):.2f}",
        "resErrorOrderMax": f"{max(error_orders):.2f}",
        "resLambdaFine": f"{float(last['Lambda']):.2f}",
        "resOmega": f"{omega:.2f}",
        "resOmegaFitK": f"{OMEGA_FIT_MIN_K}",
        "resScaledDefectsFine": f"{K[-1] * float(last['sum_defect_measured']):.2f}",
        "resBoundExcessMin": f"{100 * min(excess):.0f}",
        "resBoundExcessMax": f"{100 * max(excess):.0f}",
        "resUniversalRatio": f"{universal_bound(last) / float(last['final_error']):.0f}",
        "resDirExcessMax": f"{100 * dir_excess:.1f}",
        "resFieldRatioCoarse": ratio_cell(ratio('oracle', '', K[0], 'Lambda_field_pts')),
        "resFieldRatioFine": ratio_cell(ratio('oracle', '', K[-1], 'Lambda_field_pts')),
        "resOrderGapFine": f"{order_gap:.3f}",
        "resGaussInitMax": tex_sci(max(float(row["propagated_initialization"]) for row in gaussian.values())),
        "resGaussDiscFine": f"{float(gaussian[K[-1]]['propagated_discretization']):.4f}",
        "resMisspecLambdaFine": f"{float(toy[('misspec', '', K[-1])]['Lambda']):.2f}",
    }


def transport_macros(results: Path) -> dict:
    """The CIFAR-10 low-noise numbers the prose of section 9 quotes."""
    directory = results / TRANSPORT_DIR
    segment = sorted((row for row in rows(directory / "segment_max.csv")
                      if row["class"] == "all" and int(row["K"]) == SEGMENT_GRID),
                     key=lambda row: float(row["sigma_j"]))
    low = segment[0]
    worst_over_par = lambda row: float(row["lambda_max_9"]) / float(row["lambda_par_9"])
    # Levels near sigma = 0.3 and 1, where the displacement is closest to the worst direction.
    middle = [worst_over_par(row) for row in segment if 0.2 <= float(row["sigma_j"]) <= 1.5]
    budget = sorted(rows(directory / "budget.csv"), key=lambda row: int(row["K"]))
    ks = [int(row["K"]) for row in budget]
    linear = [float(row["budget_linear"]) for row in budget]
    median = synchronous_budgets_from_moments(directory, ks, linear)
    fine = [float(row["change"]) for row in rows(directory / "influence.csv") if int(row["K"]) == ks[-1]]
    top_two = -sum(sorted(fine)[:2]) / linear[-1]
    return {
        "auditSegK": f"{SEGMENT_GRID}",
        "auditSegSigmaLow": f"{float(low['sigma_j']):.3f}",
        "auditSegRandLow": f"{float(low['lambda_rand_9']):.0f}",
        "auditSegParLow": f"{float(low['lambda_par_9']):.2f}",
        "auditSegWorstOverParLow": f"{worst_over_par(low):.0f}",
        "auditSegWorstOverParMidMin": f"{min(middle):.0f}",
        "auditSegWorstOverParMidMax": f"{max(middle):.0f}",
        "auditSyncTrajectories": f"{int(budget[-1]['n_trajectories'])}",
        "auditSyncKmin": f"{ks[0]}",
        "auditSyncKmax": f"{ks[-1]}",
        "auditSyncRateCoarse": f"{linear[0]:.2f}",
        "auditSyncRateFine": f"{linear[-1]:.2f}",
        "auditSyncTopTwoShare": f"{100 * top_two:.0f}",
        "auditSyncMedianFine": f"{median[-1]:.2f}",
    }


def sci(value: float) -> str:
    """Three significant digits in a table cell, e.g. 1.13e-1 as 0.113 and 9.88e-3 as 0.00988."""
    return f"{value:#.3g}" if value >= 1e-3 else f"{value:.2e}"


def tex_sci(value: float) -> str:
    """Two significant digits in scientific notation for prose, e.g. 6.19e-5 as 6.2\\times10^{-5}."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def ratio_cell(value: float) -> str:
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


def universal_bound(row: dict) -> float:
    """Recursion bound with the universal Delta_j, the propagated Delta_j term rescaled to B_D1."""
    return float(row["recursion_bound_delta"]) + (UNIVERSAL_RESCALE - 1.0) * float(row["propagated_delta"])


def render_toy_refinement(results: Path) -> str:
    """Exact-start oracle under refinement: error, local biases, amplification, local orders and
    c_K = K e_K exp(-Lambda_K)."""
    summary, orders = oracle_rows(results)
    lines = [
        "% GENERATED by edm_audit/figure_data.py -- do not edit.",
        r"\begin{tabular}{@{}r rr rr rrr rr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{final error} & \multicolumn{2}{c}{summed biases}"
        r" & \multicolumn{3}{c}{amplification} & \multicolumn{2}{c}{bound\,/\,error} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}",
        r"\(K\) & \(e_K\) & order & \(\sum_j\widetilde\delta_{j,K}\) & order & \(\Lambda_K\)"
        r" & ampl.\ order & \(c_K\) & measured & universal \\",
        r"\midrule",
    ]
    for row in summary:
        k = int(row["K"])
        order = orders.get(k)
        cell = (lambda key: f"{float(order[key]):.3f}") if order else (lambda key: DASH)
        error = float(row["final_error"])
        cells = [
            f"{k}", sci(error), cell("order_error"),
            sci(float(row["sum_defect_measured"])), cell("order_local_defects"),
            f"{float(row['Lambda']):.3f}", cell("order_amplification_scale"),
            f"{k * error * math.exp(-float(row['Lambda'])):.3f}",
            f"{float(row['recursion_bound']) / error:.2f}",
            f"{universal_bound(row) / error:.0f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_toy_hierarchy(results: Path) -> str:
    """Lambda_K and its two majorants, as ratios, on the coarsest and finest grids."""
    data = {(row["arm"], row["variant"], int(row["K"])): row
            for row in rows(results / TOY_DIR / "summary.csv")}
    ks = sorted({k for (_, _, k) in data})
    coarse, fine = ks[0], ks[-1]
    lines = [
        "% GENERATED by edm_audit/figure_data.py -- do not edit.",
        r"\begin{tabular}{@{}l rrr rrr@{}}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{\(K={coarse}\)}} & \multicolumn{{3}}{{c}}{{\(K={fine}\)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"denoiser & \(\Lambda_K\) & directional & field & \(\Lambda_K\) & directional & field \\",
        r"\midrule",
    ]
    for arm, variant, label in HIERARCHY_ARMS:
        cells = [label]
        for k in (coarse, fine):
            row = data[(arm, variant, k)]
            budget = float(row["Lambda"])
            cells += [f"{budget:.3f}", f"{float(row['Lambda_dir']) / budget:.3f}",
                      ratio_cell(float(row['Lambda_field_pts']) / budget)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render(results: Path):
    stability = results / "stability_full"
    margin_path = stability / "large_noise_margin.csv"
    residual_path = stability / "regression_residual.csv"
    margin, residual = rows(margin_path), rows(residual_path)
    residual_values = [float(row["r_reg"]) for row in residual]
    definitions = {
        "auditBcrossQninety": f"{threshold(margin, 'b_probe_q90', 0.0):.1f}",
        "auditBsupercritQninety": f"{threshold(margin, 'b_probe_q90', 6.0 / 7.0):.1f}",
        "auditResidualMin": f"{min(residual_values):.2f}",
        "auditResidualMax": f"{max(residual_values):.2f}",
    }
    definitions.update(toy_macros(results))
    definitions.update(transport_macros(results))
    lines =["% GENERATED by edm_audit/figure_data.py -- do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in definitions.items())
    provenance = {
        "oracle_refinement": {
            f"{ORACLE_DIR}/{run}/{name}": sha256(results / ORACLE_DIR / run / name)
            for run in ("main", "check")
            for name in ("summary.csv", "orders.csv", "steps.csv", "metadata.json")
        },
        "toy_law": {
            f"{TOY_DIR}/{name}": sha256(results / TOY_DIR / name)
            for name in ("summary.csv", "end_to_end.csv")
        },
        "transport": {
            f"{TRANSPORT_DIR}/{name}": sha256(results / TRANSPORT_DIR / name)
            for name in ("segment_max.csv", "budget.csv", "influence.csv")
        },
        "high_noise": {
            str(margin_path.relative_to(results)): sha256(margin_path),
            str(residual_path.relative_to(results)): sha256(residual_path),
        },
    }
    return "\n".join(lines), provenance


def render_oracle_orders(results: Path) -> str:
    """Paired quadrature sensitivity, with each order using a single M."""
    main = {int(row["K"]): row for row in rows(results / ORACLE_DIR / "main/orders.csv")}
    check_path = results / ORACLE_DIR / "check/orders.csv"
    lines = [
        "% GENERATED by edm_audit/figure_data.py -- do not edit.",
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"pair $(K/2,K)$ & quantiles & error order & ampl.-based order & bias-sum order \\",
        r"\midrule",
    ]
    for check in sorted(rows(check_path), key=lambda row: int(row["K"])):
        k = int(check["K"])
        for row in (main[k], check):
            if row["initialization"] != "exact":
                raise ValueError("paired oracle orders must use exact initialization")
            values = " & ".join(f"{float(row[key]):.4f}" for key in
                                ("order_error", "order_amplification_scale", "order_local_defects"))
            lines.append(f"$({k // 2},{k})$ & {int(row['M'])} & {values}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=CODE_DIR / "generated" / "data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    macros, provenance = render(args.results_dir)
    expected = {
        args.out_dir / "audit_macros.tex": macros + "\n",
        args.out_dir / "PROVENANCE.json": json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        args.out_dir / "oracle_refinement_checks.tex": render_oracle_orders(args.results_dir),
        args.out_dir / "toy_refinement_table.tex": render_toy_refinement(args.results_dir),
        args.out_dir / "toy_hierarchy_table.tex": render_toy_hierarchy(args.results_dir),
    }
    if args.check:
        stale = [str(path) for path, content in expected.items() if not path.exists() or path.read_text() != content]
        if stale:
            raise SystemExit("stale generated files: " + ", ".join(stale))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

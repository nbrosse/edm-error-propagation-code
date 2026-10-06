"""The four numerical figures of the paper, from the runs in audit_runs/results/.

    python -m edm_audit.figures [--png]

One colour means one thing in every figure:

    black       the realized quantity (observed error, measured Lambda_K, rate along the
                displacement, margin) and, in the denoiser comparison, the oracle
    grey        references and fits (K^{-1}, fitted slope, 6/7, first order, zero)
    blue        discretization (local defects and their propagation); the trained
                networks in the denoiser comparison
    green       predictions built from the measured Lambda_K
    vermillion  worst-case bounds (universal defect, field proxy, worst direction)
    orange      initialization
    purple      learning; the misspecified posterior mean in the denoiser comparison,
                whose error is dominated by learning

Line style separates a quantity from its majorants: solid for the realized value, dashed for the
directional bound, dotted for the field-level bound.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from edm_audit.cifar import synchronous_linear_budget, synchronous_median_rate_budget
from edm_audit.common import CODE_DIR, RESULTS_DIR, read_csv


BLACK = "#1a1a1a"
GREY = "#8c8c8c"
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
LIGHT_BLUE = "#56B4E9"
SHADE = "#ececec"

WIDTH = 7.0
ROW = 2.35     # height of a one-row figure
COMPACT_WIDTH = 6.75   # \textwidth of a two-column layout
COMPACT_ROW = 1.95
TEASER_WIDTH = 2.6   # 0.8\columnwidth
TEASER_HEIGHT = 1.75

TOY_DIR = "toy_law_2026-09-22"
ORACLE_REFINEMENT_DIR = "oracle_refinement_2026-09-24"
TRANSPORT_DIR = "transport_2026-09-22"
# The main segment panel shows the coarse grid only. On G_272 one segment carries
# nearly all the energy (ESS 1.2 of 80) and its 9-node maxima are node-limited at
# sigma in [0.03, 0.3] (secant_recheck_2026-09-23); those cells are discussed in
# the appendix, not plotted.
SEGMENT_GRID = 68
# Common noise axis of the CIFAR panels that are plotted against sigma.
SIGMA_RANGE = (1.5e-3, 1.2e2)

# Grids from which the growth exponent of the oracle Lambda_K is fitted: the coarse
# grids are preasymptotic (their increments per doubling are still rising).
OMEGA_FIT_MIN_K = 136


def _style(compact: bool = False) -> None:
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
        "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False,
        "legend.handlelength": 2.2, "figure.dpi": 160, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    if compact:
        # Four panels on one row of a two-column text width: one point smaller
        # throughout, so the panels keep the manuscript's proportions.
        mpl.rcParams.update({
            "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
            "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
            "legend.handlelength": 1.8, "lines.linewidth": 1.1, "lines.markersize": 3.0,
        })


def _save(fig: plt.Figure, path: Path, png: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if png:
        # Rasterize the PDF itself: matplotlib's own PNG route under usetex needs dvipng.
        subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", str(path),
                        str(path.with_suffix(""))], check=True)


def _by_arm(rows: Sequence[dict], families: Sequence[str]) -> Dict[str, List[dict]]:
    """Group rows by run name, keeping only the requested arm families."""
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        if row["arm"] not in families:
            continue
        variant = row.get("variant", "")
        grouped.setdefault(f"{row['arm']}{variant}", []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["K"]))
    return grouped


def _steps_axis(ax: plt.Axes, label: str = "steps $K$") -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlabel(label)


def _sigma_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*SIGMA_RANGE)
    ax.set_xlabel(r"noise level $\sigma$")


def _slope_triangle(ax: plt.Axes, x0: float, y0: float, slope: float, factor: float,
                    label: str, color: str, below: bool) -> None:
    """A log-log slope marker: hypotenuse of slope ``-slope`` from (x0, y0) over ``factor``."""
    x1, y1 = x0 * factor, y0 * factor ** (-slope)
    corner = (x0, y1) if below else (x1, y0)
    ax.add_patch(plt.Polygon([(x0, y0), (x1, y1), corner], closed=True, fill=False,
                             edgecolor=color, lw=0.8))
    tx = x0 / 1.12 if below else x1 * 1.12
    ax.text(tx, np.sqrt(y0 * y1), label, color=color, fontsize=7.5, va="center",
            ha="right" if below else "left")


def _interp_loglog(K: Sequence[int], values: Sequence[float], k: float) -> float:
    return float(np.exp(np.interp(np.log(k), np.log(K), np.log(values))))


def _toy_rows(results: Path):
    """Exact-initialization oracle refinement: summary rows and local orders, sorted by K."""
    refinement = results / ORACLE_REFINEMENT_DIR
    exact = sorted(read_csv(refinement / "main/summary.csv"), key=lambda row: int(row["K"]))
    if any(row["initialization"] != "exact" for row in exact):
        raise ValueError("the error/amplification comparison requires exact initialization")
    orders = sorted(read_csv(refinement / "main/orders.csv"), key=lambda row: int(row["K"]))
    return exact, orders


def _panel_toy_error(ax: plt.Axes, exact: Sequence[dict], title: str = "(a) Error and local biases",
                     legend_fontsize: Optional[float] = 6.8) -> None:
    """With exact initialization and the oracle, error and bound are pure discretization."""
    K = [int(row["K"]) for row in exact]
    error = [float(row["final_error"]) for row in exact]
    bound = [float(row["recursion_bound"]) for row in exact]
    defects = [float(row["sum_defect_measured"]) for row in exact]
    ax.plot(K, error, marker="o", color=BLACK, label=r"final error $e_{K}$")
    ax.plot(K, bound, marker="o", color=BLUE, label="propagated biases")
    ax.plot(K, defects, marker="o", mfc="white", ls="--", color=BLUE,
            label=r"summed biases $\sum_j\widehat\delta_{j}$")
    ax.set_yscale("log")
    _steps_axis(ax)
    ax.set_title(title)
    ax.set_ylim(1e-3, 0.7)
    _slope_triangle(ax, 128.0, 0.45, 1.0, 4.0, "1", GREY, False)
    _slope_triangle(ax, 128.0, 1.55 * _interp_loglog(K, bound, 128.0), 0.58, 4.0, "0.58", GREY,
                    False)
    ax.legend(loc="lower left", fontsize=legend_fontsize)


def _panel_toy_budget(ax: plt.Axes, exact: Sequence[dict],
                      title: str = r"(b) Cumulative log-amplification",
                      legend_fontsize: Optional[float] = 7, ylabel: Optional[str] = None,
                      legend_loc: str = "upper left", ymax: Optional[float] = None) -> None:
    """The oracle Lambda_K against log K, with the slope fitted on the fine grids."""
    K = [int(row["K"]) for row in exact]
    budget = [float(row["Lambda"]) for row in exact]
    fine = [i for i, k in enumerate(K) if k >= OMEGA_FIT_MIN_K]
    omega, intercept = np.polyfit(np.log([K[i] for i in fine]), [budget[i] for i in fine], 1)
    ax.plot(K, omega * np.log(K) + intercept, color=GREY, ls="--", lw=0.9,
            label=rf"fit $\omega\log K$, $\omega={omega:.2f}$")
    ax.plot(K, budget, marker="o", color=GREEN, label=r"measured $\Lambda_K$")
    _steps_axis(ax)
    ax.set_ylim(bottom=0.0, top=ymax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)


def _panel_toy_orders(ax: plt.Axes, orders: Sequence[dict],
                      title: str = "(c) Observed and amplification-based orders",
                      xlabel: str = "finer grid $K$ of the pair $(K/2,K)$",
                      predicted_label: str = r"amplification-based, $1-\Delta\Lambda_K/\log 2$",
                      legend_fontsize: Optional[float] = 7,
                      legend_anchor: Sequence[float] = (1.0, 0.42)) -> None:
    """Local orders against 1 - (Lambda_K - Lambda_{K/2})/log 2, the order that
    K^{-1} e^{Lambda_K} would have with the measured Lambda_K."""
    K_obs = [int(row["K"]) for row in orders]
    ax.axhline(1.0, color=GREY, ls=":", lw=0.9)
    for key, label, color, style, face in (
        ("order_local_defects", r"summed biases", BLUE, "--", "white"),
        ("order_amplification_scale", predicted_label, GREEN, "-", None),
        ("order_error", r"final error $e_K$", BLACK, "-", None),
    ):
        ax.plot(K_obs, [float(row[key]) for row in orders], marker="o", mfc=face,
                color=color, ls=style, label=label)
    _steps_axis(ax, xlabel)
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("local order $p_K$")
    ax.set_title(title)
    ax.legend(loc="center right", bbox_to_anchor=tuple(legend_anchor), fontsize=legend_fontsize)


def toy_figure(results: Path, output: Path, png: bool) -> None:
    """Finding 1: first-order defects, sublinear error, and the growth of Lambda_K."""
    _style()
    exact, orders = _toy_rows(results)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, ROW))
    _panel_toy_error(axes[0], exact)
    _panel_toy_budget(axes[1], exact)
    _panel_toy_orders(axes[2], orders)
    fig.tight_layout(w_pad=1.2)
    _save(fig, output, png)


HIERARCHY_FAMILIES = (("oracle", "oracle", BLACK), ("trained", "trained (3 seeds)", BLUE),
                      ("misspec", "misspecified", PURPLE))


# Direct labels of the denoisers in the compact layout: name, number of trailing grids
# the label spans, and whether it sits above (else below) the field-proxy curve.
DENOISER_LABELS = {"oracle": ("oracle", 2, True), "trained": ("trained", 3, False),
                   "misspec": ("misspecified", 4, True)}


def _hierarchy_summary(results: Path) -> Dict[str, List[dict]]:
    return _by_arm(read_csv(results / TOY_DIR / "summary.csv"), ["oracle", "misspec", "trained"])


def _stacked(summary: Dict[str, List[dict]], family: str, key: str):
    """Grid and values of ``key`` for every run of ``family`` (one row per seed)."""
    groups = [group for group in summary.values() if group[0]["arm"] == family]
    K = [int(row["K"]) for row in groups[0]]
    return K, np.asarray([[float(row[key]) for row in group] for group in groups])


def _panel_geometry(ax: plt.Axes, geometry: Sequence[dict]) -> None:
    """Geometry at one noise level: the local rate, the error weight w over the optimal
    coupling, and the two levels the bounds pay: sup of the rate (field) and its w-average
    (directional, E_OT). The error sits where the flow contracts."""
    x = np.asarray([float(row["x"]) for row in geometry])
    rate = np.asarray([float(row["rate"]) for row in geometry])
    weight = np.asarray([float(row["weight"]) for row in geometry])
    directional = float(np.sum(weight * rate) * (x[1] - x[0]))
    shade = ax.twinx()
    shade.fill_between(x, weight, color=SHADE, lw=0)
    shade.plot(x, weight, color=GREY, lw=0.6)
    shade.set_ylim(0.0, 2.2 * weight.max())
    shade.set_yticks([])
    shade.spines["right"].set_visible(False)
    ax.set_zorder(shade.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.axhline(0.0, color=GREY, lw=0.5)
    ax.plot(x, rate, color=VERMILLION, lw=1.0, label=r"local rate $\lambda_j(x)$")
    ax.axhline(rate.max(), color=VERMILLION, ls=":", lw=0.9, label="field: supremum")
    ax.axhline(directional, color=BLACK, ls="--", lw=0.9, label=r"directional: $w_j$-average")
    ax.plot([], [], color=GREY, lw=4, alpha=0.4, label=r"error weight $w_j$ (shaded)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.05 * rate[np.abs(x) <= 1.5].min(), 3.3 * rate.max())
    ax.set_xlabel("state $x$")
    ax.set_ylabel("expansion rate")
    ax.set_title(rf"(a) Geometry at $\sigma\approx{float(geometry[0]['sigma']):.1f}$")
    ax.legend(loc="upper left", fontsize=6.2)


def _panel_hierarchy_budget(ax: plt.Axes, summary: Dict[str, List[dict]]) -> None:
    """The measured Lambda_K for each denoiser; seeds as median and range."""
    for family, label, color in HIERARCHY_FAMILIES:
        K, values = _stacked(summary, family, "Lambda")
        ax.plot(K, np.median(values, axis=0), marker="o", color=color, label=label)
        if len(values) > 1:
            ax.fill_between(K, values.min(axis=0), values.max(axis=0), color=color,
                            alpha=0.15, lw=0)
    _steps_axis(ax)
    ax.set_ylim(bottom=0.0)
    ax.set_title(r"(b) Measured $\Lambda_K$")
    ax.legend(loc="upper left", fontsize=7)


def _panel_majorant_ratio(ax: plt.Axes, summary: Dict[str, List[dict]],
                          title: str = "(c) Cost of each majorant",
                          legend_fontsize: Optional[float] = 7,
                          denoiser_labels: bool = False) -> None:
    """Each majorant divided by Lambda_K: the directional bound stays within a few
    percent, the field proxy loses one to three orders of magnitude.

    ``denoiser_labels`` names each denoiser above the right end of its field-proxy
    curve, for layouts without a Lambda_K panel whose legend would carry the colours."""
    for family, _, color in HIERARCHY_FAMILIES:
        K, budget = _stacked(summary, family, "Lambda")
        for key, style in (("Lambda_dir", "--"), ("Lambda_field_pts", ":")):
            _, values = _stacked(summary, family, key)
            ratio = np.median(values / budget, axis=0)
            ax.plot(K, ratio, marker="o", ms=2.8, ls=style, color=color)
            if denoiser_labels and key == "Lambda_field_pts":
                # Clear the markers under the label: longer names span more grids.
                name, span, above = DENOISER_LABELS[family]
                y = 1.3 * ratio[-span:].max() if above else ratio[-span:].min() / 1.3
                ax.text(K[-1], y, name, color=color, ha="right",
                        va="bottom" if above else "top",
                        fontsize=mpl.rcParams["legend.fontsize"])
    ax.axhline(1.0, color=GREY, lw=0.8)
    ax.set_yscale("log")
    _steps_axis(ax)
    ax.set_ylim(0.7, 3e3)
    ax.set_ylabel(r"majorant $/\ \Lambda_K$")
    ax.set_title(title)
    ax.legend(handles=[Line2D([], [], color=GREY, ls=":", label="field proxy"),
                       Line2D([], [], color=GREY, ls="--", label="directional bound")],
              loc="upper right", fontsize=legend_fontsize)


def hierarchy_figure(results: Path, output: Path, png: bool) -> None:
    """Finding 2: where expansion sits, Lambda_K by denoiser, and the cost of each majorant."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, ROW))
    _panel_geometry(axes[0], read_csv(results / TOY_DIR / "geometry.csv"))
    summary = _hierarchy_summary(results)
    _panel_hierarchy_budget(axes[1], summary)
    _panel_majorant_ratio(axes[2], summary)
    fig.tight_layout(w_pad=1.2)
    _save(fig, output, png)


def toy_supplementary_figure(results: Path, output: Path, png: bool) -> None:
    """Appendix: the terms of the measured recursion from the canonical Gaussian start."""
    _style()
    directory = results / TOY_DIR
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.8, ROW), sharey=True)
    rows = read_csv(directory / "end_to_end.csv")
    terms = (
        ("final_error", r"final error $e_K$", BLACK, "-"),
        ("recursion_bound", "recursion bound", GREY, "-"),
        ("propagated_discretization", "discretization", BLUE, "--"),
        ("propagated_learning", "learning", PURPLE, "--"),
        ("propagated_initialization", "initialization", ORANGE, "--"),
    )
    for ax, arm, title in ((axes[0], "oracle", "(a) Oracle denoiser"),
                           (axes[1], "trained", "(b) Trained networks (3 seeds)")):
        selected = [row for row in rows if row["arm"] == arm and row["initialization"] == "gaussian"]
        ks = sorted({int(row["K"]) for row in selected})
        for key, label, color, style in terms:
            values = [np.asarray([float(row[key]) for row in selected if int(row["K"]) == k])
                      for k in ks]
            if arm == "oracle" and key == "propagated_learning":
                continue
            ax.plot(ks, [np.median(v) for v in values], marker="o", ms=2.8, color=color,
                    ls=style, label=label)
            if len(values[0]) > 1:
                ax.fill_between(ks, [v.min() for v in values], [v.max() for v in values],
                                color=color, alpha=0.12, lw=0)
        ax.set_yscale("log")
        _steps_axis(ax)
        ax.set_title(title)
    axes[0].legend(handles=[Line2D([], [], color=color, ls=style, marker="o", ms=2.8, label=label)
                            for _, label, color, style in terms],
                   loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=7)
    fig.tight_layout(w_pad=1.0)
    _save(fig, output, png)


def synchronous_budgets_from_moments(directory: Path, ks: Sequence[int],
                                     reference: Sequence[float]) -> List[float]:
    """Median-rate budgets rebuilt from the per-trajectory moments.

    The energy-weighted budget is rebuilt from the same moments and must match
    ``budget.csv``; otherwise the moments and the table disagree and nothing is
    drawn from them.
    """
    classes = sorted(directory.glob("moments/class_*.npz"),
                     key=lambda path: int(path.stem.split("_")[1]))
    moments = [np.load(path) for path in classes]
    medians = []
    for K, expected in zip(ks, reference):
        delta2 = np.stack([m[f"K{K}_delta2"] for m in moments], axis=1)
        cross = np.stack([m[f"K{K}_cross"] for m in moments], axis=1)
        sigmas, ell = moments[0][f"K{K}_sigma_j"], moments[0][f"K{K}_ell_j"]
        rebuilt = synchronous_linear_budget(delta2, cross, sigmas, ell)
        if abs(rebuilt - expected) > 1e-12 * max(1.0, abs(expected)):
            raise ValueError(f"moments do not reproduce budget.csv at K={K}: {rebuilt} vs {expected}")
        medians.append(synchronous_median_rate_budget(delta2, cross, sigmas, ell))
    return medians


def top_block(sigma: Sequence[float], values: Sequence[float], level: float) -> float:
    """Bottom of the top contiguous block of levels where ``values`` exceeds ``level``.

    The figure marks these scales and ``figure_data`` quotes them in the prose.
    """
    answer = None
    for s, v in sorted(zip(sigma, values), key=lambda item: -item[0]):
        if v <= level:
            break
        answer = s
    if answer is None:
        raise ValueError(f"no top contiguous block exceeds {level}")
    return answer


def _cifar_margin(results: Path):
    """Large-noise margin rows sorted by sigma, and the two marked scales: where the
    90th-percentile margin turns positive and where it exceeds 6/7."""
    margin = sorted(read_csv(results / "stability_full" / "large_noise_margin.csv"),
                    key=lambda row: float(row["sigma"]))
    sigma_m = [float(row["sigma"]) for row in margin]
    # The q90 curve is the margin induced by the 90th-percentile stretch, not the
    # 90th percentile of the margins.
    q90 = [float(row["b_probe_q90"]) for row in margin]
    return margin, top_block(sigma_m, q90, 0.0), top_block(sigma_m, q90, 6.0 / 7.0)


def _segment_rows(results: Path) -> List[dict]:
    return sorted((row for row in read_csv(results / TRANSPORT_DIR / "segment_max.csv")
                   if row["class"] == "all" and int(row["K"]) == SEGMENT_GRID),
                  key=lambda row: float(row["sigma_j"]))


def _shade_margin(ax: plt.Axes, crossing: float) -> None:
    """The noise range where the conservative local margin is positive."""
    ax.axvspan(crossing, SIGMA_RANGE[1], color=SHADE, lw=0, zorder=0)


def _panel_residual(ax: plt.Axes, residual: Sequence[dict]) -> None:
    ax.plot([float(row["sigma"]) for row in residual], [float(row["r_reg"]) for row in residual],
            color=BLACK)
    _sigma_axis(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"$r_{\mathrm{reg}}$ per coordinate")
    ax.set_title("(a) Normalized regression residual")


def _panel_margin(ax: plt.Axes, margin: Sequence[dict], crossing: float,
                  supercritical: float) -> None:
    sigma_m = [float(row["sigma"]) for row in margin]
    q90 = [float(row["b_probe_q90"]) for row in margin]
    ax.axhline(0.0, color=GREY, lw=0.8)
    ax.axhline(6.0 / 7.0, color=GREY, lw=0.8, ls="--")
    ax.text(SIGMA_RANGE[0] * 1.3, 6.0 / 7.0 + 0.12, r"$6/7$", color=GREY, fontsize=7.5)
    ax.plot(sigma_m, [float(row["b_probe_q50"]) for row in margin], color=GREY, lw=1.0,
            label="median stretch")
    ax.plot(sigma_m, q90, color=BLACK, label="90th-percentile stretch")
    for s, text in ((crossing, rf"$\sigma={crossing:.1f}$"),
                    (supercritical, rf"$\sigma={supercritical:.1f}$")):
        ax.axvline(s, color=GREY, lw=0.6, ls=":")
        ax.text(s / 1.15, -2.75, text, rotation=90, fontsize=7, color=BLACK, ha="right",
                va="bottom")
    _sigma_axis(ax)
    ax.set_ylim(-3.0, 1.3)
    ax.set_ylabel(r"empirical margin $b$")
    ax.set_title("(b) High-noise damping margin")
    ax.legend(loc="lower left", fontsize=7)


SEGMENT_RATES = {
    "lambda_max_9": (r"$\lambda^{\max}$, worst", VERMILLION, "-"),
    "lambda_par_9": (r"$\lambda^{\parallel}$, displacement", BLACK, "-"),
    "E_sync_subset": (r"$\mathcal{E}^{\mathrm{sync}}$, secant", BLACK, "--"),
    "lambda_rand_9": (r"$\lambda^{\mathrm{rand}}$, random", LIGHT_BLUE, "-"),
}


def _panel_segment_rates(ax: plt.Axes, segment: Sequence[dict],
                         keys: Sequence[str] = tuple(SEGMENT_RATES),
                         title: str = rf"(c) Rates on the same segments, $K={SEGMENT_GRID}$",
                         legend_fontsize: Optional[float] = 6.5,
                         legend_kw: Optional[dict] = None,
                         short_labels: bool = False) -> None:
    """Absolute rates on a log scale; open markers flag contraction (negative rate).

    ``short_labels`` keeps only the symbols in the legend (the caption names them)."""
    s = np.asarray([float(row["sigma_j"]) for row in segment])
    for key in keys:
        label, color, style = SEGMENT_RATES[key]
        if short_labels:
            label = label.split(",")[0]
        values = np.asarray([float(row[key]) for row in segment])
        ax.plot(s, np.abs(values), color=color, ls=style, label=label)
        positive = values > 0
        ax.plot(s[positive], values[positive], ls="none", marker="o", color=color)
        ax.plot(s[~positive], -values[~positive], ls="none", marker="o", mfc="white", color=color)
    _sigma_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 6e2)
    ax.set_ylabel(r"$|$expansion rate$|$")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", mfc="white", color=GREY))
    labels.append("negative" if short_labels else "open: contraction")
    options = dict(loc="upper right", fontsize=legend_fontsize, handlelength=1.8)
    options.update(legend_kw or {})
    ax.legend(handles, labels, **options)


def _panel_sync_budget(ax: plt.Axes, directory: Path) -> None:
    """The accumulated contribution comes from the full synchronous table
    over the whole window. No cumulative field curve is built from the
    eight sparse segment levels: summing only those steps would not be a
    budget. The band is a single-trajectory influence range; the median
    rate is a descriptive statistic of the typical pair, not a budget."""
    budget = sorted(read_csv(directory / "budget.csv"), key=lambda row: int(row["K"]))
    ks = [int(row["K"]) for row in budget]
    linear = [float(row["budget_linear"]) for row in budget]
    median = synchronous_budgets_from_moments(directory, ks, linear)
    ax.fill_between(ks, [float(row["budget_min"]) for row in budget],
                    [float(row["budget_max"]) for row in budget],
                    color=BLACK, alpha=0.15, lw=0, label="leave-one-out range")
    ax.plot(ks, linear, marker="o", color=BLACK, label="energy-weighted, all pairs")
    ax.plot(ks, median, marker="o", mfc="white", ls="--", color=GREY, label="median pair")
    _steps_axis(ax)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("accumulated rate")
    ax.set_title(r"(d) Accumulated synchronous rate, $\sigma\le2$")
    ax.legend(loc="upper left", fontsize=7)


def cifar_figure(results: Path, output: Path, png: bool) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5))
    stability = results / "stability_full"
    margin, crossing, supercritical = _cifar_margin(results)

    # The noise range where the conservative local margin is positive, shaded on
    # every panel plotted against sigma.
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        _shade_margin(ax, crossing)

    residual = sorted(read_csv(stability / "regression_residual.csv"), key=lambda row: float(row["sigma"]))
    _panel_residual(axes[0, 0], residual)
    _panel_margin(axes[0, 1], margin, crossing, supercritical)
    _panel_segment_rates(axes[1, 0], _segment_rows(results))
    _panel_sync_budget(axes[1, 1], results / TRANSPORT_DIR)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    _save(fig, output, png)


def summary_row_figure(results: Path, output: Path, png: bool) -> None:
    """The one-row summary: Lambda_K and its log fit, the observed orders, the cost
    of each majorant, and the CIFAR-10 segment rates."""
    _style(compact=True)
    exact, orders = _toy_rows(results)
    summary = _hierarchy_summary(results)
    _, crossing, _ = _cifar_margin(results)
    fig, axes = plt.subplots(1, 4, figsize=(COMPACT_WIDTH, COMPACT_ROW))
    _panel_toy_budget(axes[0], exact, title=r"(a) Log-amplification $\Lambda_K$",
                      legend_fontsize=None, ymax=2.6)
    _panel_toy_orders(axes[1], orders, title="(b) Observed and ampl.-based orders",
                      xlabel="finer grid $K$ of $(K/2,K)$",
                      predicted_label="amplification-based", legend_fontsize=None,
                      legend_anchor=(1.0, 0.5))
    _panel_majorant_ratio(axes[2], summary, title="(c) Cost of each majorant",
                          legend_fontsize=None, denoiser_labels=True)
    _shade_margin(axes[3], crossing)
    _panel_segment_rates(axes[3], _segment_rows(results),
                         keys=("lambda_max_9", "lambda_par_9", "lambda_rand_9"),
                         title=rf"(d) CIFAR--10 rates on $G_{{{SEGMENT_GRID}}}$",
                         legend_fontsize=None, short_labels=True,
                         legend_kw=dict(handlelength=1.4, bbox_to_anchor=(1.03, 1.0)))
    fig.tight_layout(w_pad=0.8)
    _save(fig, output, png)


def teaser_figure(results: Path, output: Path, png: bool) -> None:
    """The teaser: with the oracle and exact initialization, the summed
    local biases decay at first order while the final error and the propagated biases do not."""
    _style(compact=True)
    exact, _ = _toy_rows(results)
    K = [int(row["K"]) for row in exact]
    error = [float(row["final_error"]) for row in exact]
    bound = [float(row["recursion_bound"]) for row in exact]
    defects = [float(row["sum_defect_measured"]) for row in exact]
    fig, ax = plt.subplots(figsize=(TEASER_WIDTH, TEASER_HEIGHT))
    ax.plot(K, defects, marker="o", mfc="white", ls="--", color=BLUE,
            label="summed biases")
    ax.plot(K, bound, marker="o", color=BLUE, label="propagated biases")
    ax.plot(K, error, marker="o", color=BLACK, label=r"final error $e_K$")
    ax.set_yscale("log")
    _steps_axis(ax, "sampling steps $K$")
    ax.set_xticks(K, [str(k) for k in K])
    ax.minorticks_off()
    ax.set_ylim(1.5e-3, 0.4)
    _slope_triangle(ax, 272.0, 0.5 * _interp_loglog(K, defects, 272.0), 1.0, 3.0, "1", GREY,
                    True)
    _slope_triangle(ax, 136.0, 1.6 * _interp_loglog(K, bound, 136.0), 0.58, 3.0, "0.58", GREY,
                    False)
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, output, png)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=CODE_DIR / "generated" / "figures")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="write only the compact figures numerical_summary_row.pdf and "
                             "numerical_teaser.pdf")
    args = parser.parse_args(argv)
    if args.compact:
        summary_row_figure(args.results_dir, args.out_dir / "numerical_summary_row.pdf", args.png)
        teaser_figure(args.results_dir, args.out_dir / "numerical_teaser.pdf", args.png)
        return 0
    toy_figure(args.results_dir, args.out_dir / "numerical_toy.pdf", args.png)
    hierarchy_figure(args.results_dir, args.out_dir / "numerical_hierarchy.pdf", args.png)
    toy_supplementary_figure(args.results_dir, args.out_dir / "numerical_toy_supplementary.pdf",
                             args.png)
    cifar_figure(args.results_dir, args.out_dir / "numerical_cifar.pdf", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

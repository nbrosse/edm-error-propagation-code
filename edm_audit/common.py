"""Helpers shared by the toy and CIFAR experiments: EDM grid and coefficients, I/O."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = CODE_DIR / "audit_runs"
RESULTS_DIR = RUNS_DIR / "results"


def edm_sigmas(K: int, sigma_max: float = 80.0, sigma_min: float = 0.002, rho_edm: float = 7.0) -> np.ndarray:
    """The K + 1 decreasing levels of the EDM grid G_K (uniform in sigma^(1/rho_edm)).

    ``rho_edm`` = 7 is the schedule exponent of Karras et al. (2022); the manuscript's clock
    exponent is its reciprocal. Level j of G_K is level 2j of G_2K, bit for bit.
    """
    i = np.arange(int(K) + 1, dtype=np.float64)
    lo, hi = sigma_min ** (1.0 / rho_edm), sigma_max ** (1.0 / rho_edm)
    return (hi + i / int(K) * (lo - hi)) ** rho_edm


def edm_coeffs(sigma, sigma_data):
    """(c_skip, c_out, c_in) of the EDM preconditioning; works on floats, arrays and tensors.

    In the manuscript alpha = c_skip and beta = c_out.
    """
    norm2 = sigma ** 2 + sigma_data ** 2
    return sigma_data ** 2 / norm2, sigma * sigma_data / norm2 ** 0.5, 1.0 / norm2 ** 0.5


def nearest_in_log(levels: Sequence[float], target: float) -> int:
    """Index of the level closest to ``target`` in log noise."""
    return min(range(len(levels)), key=lambda i: abs(np.log(float(levels[i]) / float(target))))


def stats(values, prefix: str) -> Dict[str, float]:
    """Mean, spread and quantiles of a sample, as ``{prefix}_{stat}`` columns."""
    arr = np.asarray(values, dtype=np.float32).ravel()
    out = {f"{prefix}_n": int(arr.size), f"{prefix}_mean": float(arr.mean()),
           f"{prefix}_std": float(arr.std()), f"{prefix}_min": float(arr.min())}
    for q in (50, 75, 90, 95, 99):
        out[f"{prefix}_q{q}"] = float(np.quantile(arr, q / 100))
    out[f"{prefix}_max"] = float(arr.max())
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write dict rows; the header is the union of their keys, in order of appearance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj)}")


def write_summary(outdir: Path, experiment: str, config: Dict, results: Dict, started: float) -> None:
    """``summary.json``: the parameters of the run, its duration and its scalar results."""
    summary = {"experiment": experiment, "seconds": time.time() - started, "config": config, "results": results}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    print(f"Done: {outdir}")

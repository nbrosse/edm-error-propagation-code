"""Three analytic controls on the one-dimensional mixture (appendix J, table of controls).

* clock: the oracle from the exact start on EDM schedules rho_edm = 2, 3, 7 at equal K,
  with the relative mesh theta_K of each grid;
* starting noise: the clock step h is held fixed while the top of the schedule moves;
* synchronous calibration: the coarse/fine synchronous rates, the only ones observable on
  CIFAR, beside the optimal-coupling rates on the same levels.

    python -m edm_audit.run_toy_controls [--smoke] [--out DIR]

Outputs: clock.csv, starting_noise.csv, synchronous_calibration.csv, summary.json.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from edm_audit.common import RUNS_DIR, edm_sigmas, write_csv, write_summary
from edm_audit.toy import (
    Mixture1D, audit_grid, exact_oracle_rows, log_amplification, midpoint_levels, misspecified,
    recursion_decomposition,
)


CONFIG = {
    "sigma_data": 0.5,
    "law": {"weights": [0.15, 0.35, 0.35, 0.15], "centers": [-3.0, -1.0, 1.0, 3.0], "width": 0.04},
    "misspec": {"weights": [0.2, 0.3, 0.3, 0.2], "center_scale": 1.02, "width_scale": 0.7},
    "sigma_max": 80.0,
    "sigma_min": 0.002,
    "M": 4096,
    "check_factor": 2,             # the finest clock grids are repeated with 2M quantiles
    # rho_edm = 1 is excluded: theta_K > 1 for every K considered.
    "clock": {"rho_edm": [2.0, 3.0, 7.0], "K": [68, 136, 272, 544]},
    # The clock step of G_272 (rho_edm = 7) is kept; the top of the schedule moves.
    "starting_noise": {"rho_edm": 7.0, "reference_sigma_max": 80.0, "reference_K": 272,
                       "targets": [20.0, 80.0, 320.0], "sigma_hi": 4.0, "b_hi": 0.9},
    "synchronous": {"rho_edm": 7.0, "K": [68, 136, 272], "sigma_window": 2.0},
}

SMOKE = {
    **CONFIG,
    "M": 256,
    "clock": {"rho_edm": [2.0, 7.0], "K": [17, 34]},
    "starting_noise": {**CONFIG["starting_noise"], "reference_K": 34, "targets": [20.0, 80.0]},
    "synchronous": {**CONFIG["synchronous"], "K": [17, 34]},
}


def relative_mesh(K: int, sigma_max: float, sigma_min: float, rho_edm: float) -> float:
    """theta_K = h_K / eta_min in the clock eta = sigma^(1/rho_edm) (lem:schedule_family_mesh)."""
    eta_max, eta_min = sigma_max ** (1.0 / rho_edm), sigma_min ** (1.0 / rho_edm)
    return (eta_max - eta_min) / (K * eta_min)


def kappa(theta: float, rho: float) -> float:
    """Finite-mesh damping factor (eq:def_kappa_mesh_edm_predictor); tends to 1 as theta -> 0."""
    if not 0.0 < theta < 1.0:
        return float("nan")
    return (1.0 - (1.0 - theta) ** (1.0 / rho)) / (math.log(1.0 / (1.0 - theta)) / rho)


def oracle_derivative_bound(mix: Mixture1D, sigma: float) -> float:
    """Global bound on D'_sigma for a mixture: within-component variance plus a quarter of the
    squared span of the means."""
    w2 = mix.width ** 2
    span = float(np.max(mix.means) - np.min(mix.means))
    return w2 / (w2 + sigma ** 2) + sigma ** 2 * span ** 2 / (4.0 * (w2 + sigma ** 2) ** 2)


def clock_rows(mix: Mixture1D, cfg: dict, M: int) -> list:
    rows = []
    for rho_edm in cfg["clock"]["rho_edm"]:
        for K in cfg["clock"]["K"]:
            steps = exact_oracle_rows(mix, edm_sigmas(K, cfg["sigma_max"], cfg["sigma_min"], rho_edm), M)
            total = recursion_decomposition(steps)
            theta = relative_mesh(K, cfg["sigma_max"], cfg["sigma_min"], rho_edm)
            rows.append({
                "rho_edm": rho_edm, "K": K, "M": M, "theta_K": theta,
                "outside_mesh": bool(theta >= 0.5),   # the two-regime estimates do not apply
                "kappa": kappa(theta, 1.0 / rho_edm),
                "final_error": total["final_error"],
                "sum_defect_measured": sum(r["discretization_term"] for r in steps),
                "sum_defect_universal": sum(r["Delta_j"] for r in steps),
                "propagated_discretization": total["propagated_discretization"],
                "propagated_delta": total["propagated_delta"],
                "amplification_product": total["amplification_product"],
            })
    return rows


def starting_noise_rows(mix: Mixture1D, cfg: dict, M: int) -> list:
    c = cfg["starting_noise"]
    rho_edm, sigma_min = c["rho_edm"], cfg["sigma_min"]
    rho = 1.0 / rho_edm
    eta_min = sigma_min ** rho
    h = (c["reference_sigma_max"] ** rho - eta_min) / c["reference_K"]   # the fixed clock step
    rows = []
    for target in c["targets"]:
        K = int(round((target ** rho - eta_min) / h))
        sigma_max = (eta_min + K * h) ** rho_edm
        steps = exact_oracle_rows(mix, edm_sigmas(K, sigma_max, sigma_min, rho_edm), M)
        high = [r for r in steps if r["sigma_next"] >= c["sigma_hi"]]
        # Each high-noise defect carried to the bottom of the block through (1 - b_hi a_k).
        damped, weight = 0.0, 1.0
        for r in reversed(high):
            damped += r["Delta_j"] * weight
            weight *= max(0.0, 1.0 - c["b_hi"] * r["a"])
        theta_hi = h / c["sigma_hi"] ** rho
        certificate = max((oracle_derivative_bound(mix, r["sigma"]) for r in high), default=float("nan"))
        rows.append({
            "target_sigma_max": target, "sigma_max_actual": sigma_max, "K": K, "h": h, "M": M,
            "n_high": len(high),
            "final_error": recursion_decomposition(steps)["final_error"],
            "sum_defect_high_undamped": sum(r["Delta_j"] for r in high),
            "sum_defect_high_damped": damped,
            "theta_hi": theta_hi, "kappa_hi": kappa(theta_hi, rho),
            "b_eff": c["b_hi"] * kappa(theta_hi, rho),
            "supercritical": bool(c["b_hi"] * kappa(theta_hi, rho) > 6.0 / 7.0),
            "oracle_derivative_bound": certificate,
            "certificate_holds": bool(certificate <= 1.0 - c["b_hi"]),   # needs sup D' <= 1 - b_hi
        })
    return rows


def euler_states(denoiser, sigmas: np.ndarray, u: np.ndarray) -> list:
    """Euler trajectory of the quantile ensemble without sorting, so each latent keeps its identity."""
    state = float(sigmas[0]) * ndtri(u)
    states = [state]
    for j in range(len(sigmas) - 1):
        a = (sigmas[j] - sigmas[j + 1]) / sigmas[j]
        state = (1.0 - a) * state + a * denoiser(state, float(sigmas[j]))[0]
        states.append(state)
    return states


def synchronous_rows(mix: Mixture1D, denoisers: dict, cfg: dict, M: int) -> list:
    c = cfg["synchronous"]
    window = c["sigma_window"]
    u = midpoint_levels(M)
    rows = []
    for name, denoiser in denoisers.items():
        for K in c["K"]:
            sigmas = edm_sigmas(K, cfg["sigma_max"], cfg["sigma_min"], c["rho_edm"])
            coarse = euler_states(denoiser, sigmas, u)
            fine = euler_states(denoiser, edm_sigmas(2 * K, cfg["sigma_max"], cfg["sigma_min"], c["rho_edm"]), u)
            sync_linear = sync_quadratic = 0.0
            for j in range(K):
                sigma = float(sigmas[j])
                zeta = fine[2 * j] - coarse[j]                  # same latent, two grids
                energy = float(np.mean(zeta ** 2))
                if sigma > window or energy <= 0.0:
                    continue
                dv = ((fine[2 * j] - denoiser(fine[2 * j], sigma)[0])
                      - (coarse[j] - denoiser(coarse[j], sigma)[0])) / sigma
                ell = sigma - float(sigmas[j + 1])
                sync_linear += ell * max(-float(np.mean(zeta * dv)) / energy, 0.0)
                sync_quadratic += 0.5 * ell ** 2 * float(np.mean(dv ** 2)) / energy
            ot = [r for r in audit_grid({name: denoiser}, mix, sigmas, M, cfg["sigma_data"]) if r["sigma"] <= window]
            rows.append({
                "arm": name, "K": K, "M": M, "sigma_window": window, "n_levels": len(ot),
                "sync_linear": sync_linear, "sync_quadratic": sync_quadratic,
                "ot_linear": sum(r["ell"] * r["E_OT_plus"] for r in ot),
                "ot_quadratic": sum(0.5 * (r["ell"] * r["Q_OT"]) ** 2 for r in ot),
                "Lambda_window": log_amplification(ot),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny settings that only exercise the code")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = SMOKE if args.smoke else CONFIG
    out = args.out or RUNS_DIR / ("toy_controls_smoke" if args.smoke else "toy_controls")
    started = time.time()

    mix = Mixture1D.standardized(cfg["law"]["weights"], cfg["law"]["centers"], cfg["law"]["width"], cfg["sigma_data"])
    wrong = misspecified(mix, **cfg["misspec"])
    M = cfg["M"]
    clock = clock_rows(mix, cfg, M)
    write_csv(out / "clock.csv", clock)
    write_csv(out / "starting_noise.csv", starting_noise_rows(mix, cfg, M))
    write_csv(out / "synchronous_calibration.csv",
              synchronous_rows(mix, {"oracle": mix.posterior_mean, "misspec": wrong.posterior_mean}, cfg, M))

    # The finest clock grids again with 2M quantiles: smaller differences are not interpreted.
    K = max(cfg["clock"]["K"])
    quadrature_check = {}
    for rho_edm in cfg["clock"]["rho_edm"]:
        coarse = next(r for r in clock if r["rho_edm"] == rho_edm and r["K"] == K)["final_error"]
        fine = recursion_decomposition(exact_oracle_rows(
            mix, edm_sigmas(K, cfg["sigma_max"], cfg["sigma_min"], rho_edm), M * cfg["check_factor"]))["final_error"]
        quadrature_check[f"clock_rho{rho_edm:g}"] = {
            "K": K, "M": M, "M_check": M * cfg["check_factor"], "final_error_M": coarse,
            "final_error_M_check": fine, "final_error_rel_diff": abs(coarse - fine) / max(abs(fine), 1e-300)}
    write_summary(out, "toy_controls", cfg, {"quadrature_check": quadrature_check}, started)


if __name__ == "__main__":
    main()

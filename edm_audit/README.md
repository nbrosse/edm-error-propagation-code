# Numerical experiments

Code and frozen results for the numerical experiments of

> Nicolas Brosse, Arnak S. Dalalyan. *Universal Local Error and Realized Amplification for the
> First-Order EDM Predictor.* [arXiv:2610.10190](https://arxiv.org/abs/2610.10190), 2026.

The numerical appendix of the paper is the reference for what is computed and why. The code
builds on the NVlabs EDM code at the root of this repository and is distributed under its license,
CC BY-NC-SA 4.0 (`LICENSE.txt`).

To check that every number quoted in the paper and every generated table matches the frozen
results, run from the repository root:

```bash
uv run edm-audit-figure-data --check
```

It fails if one of them is stale; without `--check` it regenerates them in `generated/data/`. The
figures are regenerated in `generated/figures/` by `uv run edm-audit-figures`; `--compact` writes
only the two compact figures (the one-row summary and the teaser).

| Module | Content |
| --- | --- |
| `common.py` | EDM grid `G_K`, EDM coefficients, CSV reader, CSV / summary writers |
| `toy.py` | 1D Gaussian mixture, its denoisers (oracle, misspecified, perturbed, MLP), the law-level quantities of one predictor step, the error recursion, accumulation |
| `cifar.py` | EDM network and probe bank, state-recording predictor, JVP / power iteration / Lanczos, synchronous and segment rates, energy-weighted aggregation, bootstrap, influence |
| `run_*.py` | one script per experiment, its parameters at the top (`CONFIG`, and `SMOKE` for a quick run) |
| `figures.py` | the four numerical figures of the paper |
| `figure_data.py` | the numbers quoted in the prose (`audit_macros.tex`), the generated tables and `PROVENANCE.json`; `--check` fails if they are stale |

## Experiments and the runs used in the paper

All runs are in `audit_runs/results/`.

| Script | Where in the paper | Run | Hardware |
| --- | --- | --- | --- |
| `run_toy_law` | 1D Λ_K by denoiser and its majorants (hierarchy fig. and table), Gaussian-start recursion (supplementary fig.) | `toy_law_2026-09-22` | CPU, 2–4 h |
| `run_oracle_refinement` | exact-start error, amplification and orders (fig. toy, refinement table, paired-order table) | `oracle_refinement_2026-09-24/{main,check}` | CPU, ~5 and ~20 min |
| `run_toy_controls` | clock, starting noise, synchronous calibration (controls table) | `toy_controls_2026-09-22` | CPU, ~5 min |
| `run_stability` | high-noise margin and regression residual (fig. CIFAR (a)–(b), margin crossings) | `stability_full` | GPU |
|  | power-iteration re-check of the margin (reliability table) | `margin_recheck_2026-09-23` | GPU (L4), ~75 min |
| `run_transport` | synchronous budget, same-segment rates, influence (fig. CIFAR (c)–(d)) | `transport_2026-09-22` | GPU (L4), ~6 h |
| `run_secant_recheck` | 65-node rerun of the `K = 272` segments (reliability table) | `secant_recheck_2026-09-23` | GPU (L4), ~40 min |

`stability_full` and the two re-checks were produced before the code was reorganized into these
scripts; the scripts compute the same quantities.

## Running

From the repository root:

```bash
uv sync --extra cpu                 # --extra cu116 on the GPU machine
uv run --with pytest python -m pytest edm_audit/tests -q

uv run python -m edm_audit.run_toy_law --smoke      # every script has --smoke and --out
uv run python -m edm_audit.run_transport --smoke    # synthetic network, no download

uv run python -m edm_audit.run_oracle_refinement --M 65536 \
    --out audit_runs/results/oracle_refinement_2026-09-24/main
uv run python -m edm_audit.run_oracle_refinement --M 262144 --K 272 544 1088 \
    --out audit_runs/results/oracle_refinement_2026-09-24/check

uv run python -m edm_audit.run_stability --parts convergence    # the margin re-check only
uv run edm-audit-figures --png                                            # generated/figures/
uv run edm-audit-figure-data                                              # macros and tables; --check to verify
```

`--smoke` runs exercise the code path on tiny settings; their numbers are not estimates. Without
`--out`, a run writes to `audit_runs/<experiment>` (or `<experiment>_smoke`), never to `results/`.
The CIFAR scripts download the EDM checkpoint and read `datasets/cifar10-32x32-test.zip` (built with `dataset_tool.py`).

## Conventions

- `K` is the number of steps: `G_K` has `K + 1` levels from 80 to 0.002, uniform in
  `sigma^(1/rho_edm)` with `rho_edm = 7` (the manuscript's clock exponent is `1/rho_edm`).
  `G_K` is nested in `G_2K`, bit for bit, which is what makes the synchronous pairs exact.
- CIFAR inference is float32, as deployed; reductions over trajectories are float64. The toy is
  float64 throughout.
- Power iteration and Lanczos converge from below: the stretch `S_F` and `lambda_max` are
  underestimated, so the margin `b_probe` is optimistic. The re-checks quantify this.
- The energy-weighted aggregates are untrimmed: a zero displacement has zero weight; a non-finite
  value at positive weight makes the aggregate NaN and is reported by id.

## Symbol → column

| Symbol | Column | File |
| --- | --- | --- |
| *`run_toy_law`* | | |
| `e_{j,K}`, `e_{j+1,K}`, `gamma_{j,K}`, `A^OT` | `e`, `e_next`, `gamma`, `A_OT` | `steps.csv` |
| `E^OT`, `(E^OT)_+`, `Q^OT` | `E_OT`, `E_OT_plus`, `Q_OT` | `steps.csv` |
| directional term `ell (E^OT)_+ + ell^2 (Q^OT)^2 / 2` | `directional_term` | `steps.csv` |
| field maxima and margin, and their wider/finer check | `Los_pts`, `Lv_pts`, `b_pts`, `*_check`, `field_term_pts`, `field_pts_rel_change` | `steps.csv` |
| `Delta_j`, `delta_hat_j`, `r_hat_j`, `b^law_j` | `Delta_j`, `discretization_term`, `learning_term`, `b_law` | `steps.csv` |
| `Lambda_K`, `Lambda^dir_K`, `Lambda^field,pts_K` | `Lambda`, `Lambda_dir`, `Lambda_field_pts` | `summary.csv` |
| `C^init`, `C^disc`, `C^learn`, their sum | `propagated_*`, `recursion_bound`, `recursion_bound_delta` | `end_to_end.csv` |
| resolution gate | `resolved`, `quadrature_abs_diff`, `bound_over_error` | `end_to_end.csv` |
| `Lambda_2K - Lambda_K`, split over noise decades | `Lambda_increment`, `decade_{n}` | `refinement.csv` |
| rates at fixed noise | `sigma_target`, `sigma_actual`, `E_OT`, `Los_pts` | `fixed_noise.csv` |
| re-summation at each candidate threshold | `threshold`, `is_common`, `Lambda`, `Lambda_dir` | `threshold_sensitivity.csv` |
| *`run_oracle_refinement`* | | |
| local orders of `e_K`, `C^disc`, `sum delta_hat`, and `1 - Delta Lambda / log 2` | `order_error`, `order_measured_bound`, `order_local_defects`, `order_amplification_scale` | `orders.csv` |
| *`run_toy_controls`* | | |
| `theta_K`, `kappa`, defect sums | `theta_K`, `outside_mesh`, `kappa`, `sum_defect_measured`, `sum_defect_universal` | `clock.csv` |
| `theta_hi`, `kappa_hi`, `b_eff`, certificate | `theta_hi`, `kappa_hi`, `b_eff`, `supercritical`, `certificate_holds` | `starting_noise.csv` |
| synchronous vs optimal-coupling rates | `sync_linear`, `sync_quadratic`, `ot_linear`, `ot_quadratic`, `Lambda_window` | `synchronous_calibration.csv` |
| *`run_stability`* | | |
| `r_reg` | `r_reg` | `regression_residual.csv` |
| `S_F` quantiles, `b_probe` | `SF_q90`, `b_probe_q50`, `b_probe_q90` | `large_noise_margin.csv` |
| margin at 8 / 16 / 32 iterations | `power_iters`, `b_probe_q90` (and per probe and start: `SF_iter{n}`) | `margin_convergence*.csv` |
| *`run_transport`* | | |
| `A^sync`, `E^sync`, `Q^sync` and raw moments | `A_sync`, `E_sync`, `E_sync_plus`, `Q_sync`, `mean_*` | `synchronous_transport.csv` |
| `lambda^par`, `lambda^rand`, `lambda^max` (3, 5, 9 nodes) | `lambda_par_9`, `lambda_rand_9`, `lambda_max_9`, … | `segment_max.csv` |
| secant check, Lanczos diagnostics | `lambda_par_integral`, `secant_integral_gap`, `lanczos_half_drift`, `lanczos_residual` | `segment_max.csv` |
| concentration, validity, bootstrap | `ess`, `max_weight`, `valid`, `invalid_ids`, `lambda_*_9_boot_lo/hi` | `segment_max.csv` |
| accumulated synchronous rate and its influence range | `budget_linear`, `budget_quadratic`, `budget_min`, `budget_max` | `budget.csv` |
| budget without each trajectory | `class`, `latent_index`, `budget_without` | `influence.csv` |
| per-trajectory moments | `K{K}_delta2`, `K{K}_cross`, `K{K}_sigma_j`, `K{K}_ell_j` | `moments/class_{c}.npz` |

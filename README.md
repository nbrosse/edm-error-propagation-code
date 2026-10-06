# edm-error-propagation-code

Code and frozen results for the numerical experiments of

> Nicolas Brosse, Arnak S. Dalalyan. *Universal Local Error and Realized Amplification for the
> First-Order EDM Predictor.*

The experiments live in [`edm_audit/`](edm_audit/README.md), which documents each script, the run
behind every figure and table of the paper, and the meaning of every output column. They are built
on the official EDM implementation of Karras et al. (2022),
[NVlabs/edm](https://github.com/NVlabs/edm), whose files (`dnnlib/`, `torch_utils/`, `training/`,
`*.py` at the root, `docs/`) are included unmodified; its original README is
[`README_EDM.md`](README_EDM.md).

## Layout

- `edm_audit/`: the experiments (one `run_*.py` script per experiment), the figures, the
  generated numbers and tables, and the unit tests.
- `audit_runs/results/`: the frozen runs every number, figure and table of the paper comes from.
- `generated/data/`: the numbers and tables derived from those runs, with the sha256 of their
  inputs in `PROVENANCE.json`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.9.

```bash
uv sync --extra cpu                                  # --extra cu116 on a GPU machine
uv run --with pytest python -m pytest edm_audit/tests -q
uv run edm-audit-figure-data --check                 # generated numbers match the frozen runs
uv run edm-audit-figures --png                       # figures in generated/figures/
```

The one-dimensional experiments run on CPU; the CIFAR-10 experiments need a GPU, download the
pretrained EDM checkpoint, and read `datasets/cifar10-32x32-test.zip` (built with
`dataset_tool.py`, see `README_EDM.md`). See `edm_audit/README.md` for the commands and run times.

## License

This repository is a derivative of NVlabs/edm, copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES,
and is distributed under the same license, Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (`LICENSE.txt`).

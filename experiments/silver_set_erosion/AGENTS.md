# Repository Guidelines

## Project Structure & Module Organization
- Core pipeline code lives in the repo root: `main.py` (orchestrator), `banks.py`, `features.py`, `knn.py`, `xdboost.py`, `crf_utils.py`, `shadow_filter.py`, `metrics_utils.py`, `io_utils.py`, `plotting.py`.
- Configuration is centralized in `config.py`.
- Data assets live under `data/` (tiles, labels, cached DINO features).
- Outputs go under `output/run_XXX/` (plots, shapes, logs).
- Experimental notes and docs: `KB.md`, `Implementation.md`.
- Tests: `test/` contains an alternate experimental pipeline (`test/main_2.py`), not a formal test suite.

## Build, Test, and Development Commands
- Run the full pipeline:
  - `python main.py`
  - Uses `SOURCE_TILES`, `VAL_TILES`, and `HOLDOUT_TILES` from `config.py`.
- (Optional) SLURM/HPC run script: `silver_set.sh`.
- No build step is required beyond installing dependencies.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation, snake_case for functions/variables, and CapWords for classes.
- Keep changes localized and add small comments only when logic is non-obvious.
- Logging is preferred over `print`; logs go to `output/run_XXX/run.log`.

## Testing Guidelines
- There is no formal test framework in this repo.
- If you modify core logic, validate by running `python main.py` and inspecting:
  - `output/run_XXX/plots/` for visual sanity
  - `output/run_XXX/shapes/` for shapefiles

## Commit & Pull Request Guidelines
- No commit message convention is documented in this repo.
- For PRs, include a short description of changes and note any updated config defaults.
- Every code change must be recorded in `journal.md` under a new `## Change N` entry.

## Configuration & Runtime Notes
- Resolution control: `RESAMPLE_FACTOR` in `config.py` (3 = 0.2m/px → 0.6m/px).
- Validation strategy: tune on `VAL_TILES`, infer on `HOLDOUT_TILES` with fixed settings.
- Feature cache uses metadata; stale caches are auto-recomputed if `RESAMPLE_FACTOR` or patch size changes.

# Journal

## Change 1: Resampling + cache metadata + eval workflow updates
- Date: 2026-01-26
- Author: Codex
- Summary: Added RESAMPLE_FACTOR support, cache validation via metadata, and updated main workflow for validation/holdout inference with updated plotting.
- Files touched: `config.py`, `io_utils.py`, `features.py`, `banks.py`, `knn.py`, `xdboost.py`, `main.py`, `plotting.py`, `AGENTS.md`
- Notes: Clear stale feature caches if running with prior resolution settings.

## Change 2: Bank cache auto-cleanup
- Date: 2026-01-26
- Author: Codex
- Summary: Added automatic cleanup of stale bank caches keyed by patch size, context radius, and resample factor.
- Files touched: `banks.py`
- Notes: Removes outdated `*_pos_bank.npy` / `*_neg_bank.npy` files for the same image_id.

## Change 3: Tile-specific logging
- Date: 2026-01-26
- Author: Codex
- Summary: Added tile-aware logging for validation tuning and holdout inference steps.
- Files touched: `main.py`
- Notes: Logs now include the tile path during loading, scoring, CRF, and shadow stages.

## Change 4: GPU selection logging
- Date: 2026-01-26
- Author: Codex
- Summary: Log CUDA_VISIBLE_DEVICES and GPU name; use cuda:0 within the visible set.
- Files touched: `main.py`
- Notes: This respects Slurm device assignment.

## Change 5: CRF worker cap
- Date: 2026-01-26
- Author: Codex
- Summary: Set CRF_NUM_WORKERS to 16 for safer multi-core usage.
- Files touched: `config.py`
- Notes: Adjust with SLURM CPU allocation if needed.

## Change 6: Slurm CPU allocation
- Date: 2026-01-26
- Author: Codex
- Summary: Added --cpus-per-task=16 to the Slurm script to match CRF workers.
- Files touched: `silver_set.sh`
- Notes: Keep in sync with CRF_NUM_WORKERS for best utilization.

## Change 7: Bank labeling sensitivity
- Date: 2026-01-26
- Author: Codex
- Summary: Lowered POS_FRAC_THRESH and added BANK_EROSION_RADIUS to preserve thin positives after resampling.
- Files touched: `config.py`, `banks.py`
- Notes: Use BANK_EROSION_RADIUS=0 for 0.6m/px runs.

## Change 8: Plot layout cleanup
- Date: 2026-01-26
- Author: Codex
- Summary: Fixed save_plot layout to avoid empty panels by using a 2x3 grid when labels or shadow are present.
- Files touched: `plotting.py`

## Change 9: Skip empty source tiles
- Date: 2026-01-26
- Author: Codex
- Summary: Skip source tiles with zero positives instead of aborting the run.
- Files touched: `banks.py`, `main.py`

## Change 10: Skip empty XGB tiles
- Date: 2026-01-26
- Author: Codex
- Summary: Skip source tiles with zero XGB positives and error if the combined dataset is empty.
- Files touched: `xdboost.py`, `main.py`

## Change 11: Mask shape alignment guard
- Date: 2026-01-26
- Author: Codex
- Summary: Added safety resizing when reprojected labels or GT masks do not match the downsampled image shape.
- Files touched: `io_utils.py`, `main.py`

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

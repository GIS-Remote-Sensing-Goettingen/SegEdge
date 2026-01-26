# Journal

## Change 1: Resampling + cache metadata + eval workflow updates
- Date: 2026-01-26
- Author: Codex
- Summary: Added RESAMPLE_FACTOR support, cache validation via metadata, and updated main workflow for validation/holdout inference with updated plotting.
- Files touched: `config.py`, `io_utils.py`, `features.py`, `banks.py`, `knn.py`, `xdboost.py`, `main.py`, `plotting.py`, `AGENTS.md`
- Notes: Clear stale feature caches if running with prior resolution settings.

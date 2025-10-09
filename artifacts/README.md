# Artifacts

All generated assets are stored under this top-level directory:

- `checkpoints/` — Local copies of model weights (ignored by git).
- `logs/` — Structured text logs grouped by pipeline.
- `outputs/` — Binary masks, overlays, GeoPackages, and QA figures.

Pipelines create subfolders automatically. Clean up stale runs regularly to keep
the workspace lightweight.

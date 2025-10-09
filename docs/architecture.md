# SegEdge Architecture Guide

This document captures the high-level organisation of the restructured repository.  
Treat it as the canonical reference when adding new pipelines or experiments.

---

## System Overview

SegEdge hosts two primary inference pipelines:

- **Tree crown delineation** powered by DINOv3 SAT-493M (`sege.pipelines.dinov3_tree_crowns`).
- **Farmland segmentation** using SAM 2 (`sege.pipelines.sam2_farmland`) with an accompanying large-scene tiling workflow (`experiments/sam2/hpc/tiling_mask_generator.py`).

Each pipeline is implemented as a reusable Python module under `src/sege/pipelines`, with thin CLI wrappers and HPC launchers in `experiments/`. Shared artifacts (models, outputs, logs, sample data) live under `artifacts/` and `data/`.

---

## Directory Layout

| Path | Purpose |
| ---- | ------- |
| `src/sege/` | Installable Python package containing production-ready pipelines and utilities. |
| `src/sege/pipelines/` | Core inference modules (`dinov3_tree_crowns.py`, `sam2_farmland.py`). |
| `experiments/` | Experiment notebooks, HPC drivers, and prototypes grouped by model family. |
| `configs/` | YAML templates for experiment configuration (per-pipeline). |
| `requirements/` | Environment definitions (`base.txt`, `experiments/*.txt`). |
| `data/samples/` | Lightweight imagery examples for quick smoke tests. |
| `artifacts/checkpoints/` | Local checkpoints (e.g. SAM 2 weights). |
| `artifacts/outputs/` | Generated masks, GeoPackages, overlays, and QA assets. |
| `artifacts/logs/` | Persistent log output grouped by pipeline. |
| `docs/` | Architecture notes, pipeline deep-dives, references, and assets. |
| `notebooks/` | Interactive exploration notebooks (e.g. SAM 2 prompt experiments). |
| `third_party/sam2/` | Vendor drop of Meta’s SAM 2 repository for offline inference. |

---

## Execution Flow

1. **CLI / Scheduler Entry**  
   - Local runs call `python -m sege.pipelines.<pipeline> ...`.  
   - Cluster jobs submit SLURM wrappers in `experiments/<pipeline>/hpc/*.sh`, which activate the environment and run thin orchestration scripts.

2. **Pipeline Core**  
   - Handles configuration parsing, logging, device selection, and instrumentation.  
   - Invokes shared utility layers (e.g. patching, timer summaries) located alongside each module.

3. **Artifacts**  
   - All derived data is written to `artifacts/outputs/<pipeline>/...`.  
   - Logs are centralised in `artifacts/logs/<pipeline>/...`.

4. **Documentation Loop**  
   - Each pipeline maintains a deep dive at `docs/pipelines/<name>.md`.  
   - Architecture and requirements docs describe how to reproduce environments.

---

## Subsystems

- **DINOv3 Tree Crowns**  
  - Entry point: `src/sege/pipelines/dinov3_tree_crowns.py`  
  - HPC launcher: `experiments/dinov3/hpc/tiling_tree_crowns.py` + `slurm_dinov3.sh`  
  - Requirements: `requirements/experiments/dinov3.txt`

- **SAM2 Farmland Segmentation**  
  - Entry point: `src/sege/pipelines/sam2_farmland.py`  
  - Tiling driver (automatic mask generator): `experiments/sam2/hpc/tiling_mask_generator.py`  
  - Requirements: `requirements/experiments/sam2.txt`

- **Third-Party Assets**  
  - SAM 2 vendor code is vendored into `third_party/sam2`. Installable with `pip install -e third_party/sam2` if desired.

---

## Data & Artifacts Policy

- Sample imagery is stored in `data/samples/imagery`. Avoid committing large proprietary datasets; document download instructions instead.
- Intermediate results should stay under `artifacts/outputs`. Purge or archive when they become stale.
- Checkpoints belong to `artifacts/checkpoints`. Keep `.gitignore` entries aligned with this layout to prevent accidental commits of large binaries.

---

## Extending the Repo

1. Add new pipeline code to `src/sege/pipelines/` with a `main(argv=None)` signature.
2. Document the pipeline under `docs/pipelines/`.
3. Register requirements in `requirements/experiments/<name>.txt`.
4. Provide optional HPC scripts and SLURM wrappers in `experiments/<name>/`.
5. Update this architecture guide if the folder structure evolves.

---

For further background materials, browse `docs/references/` which consolidates model tables, papers, and VRAM sizing notes.

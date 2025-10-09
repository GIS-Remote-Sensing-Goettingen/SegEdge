# SegEdge

SegEdge is a collection of remote-sensing segmentation pipelines focused on tree-crown delineation and agricultural field extraction. The repository now follows a layered architecture:

1. **Installable package** (`src/sege`) containing reusable inference code.
2. **Experiments & schedulers** (`experiments/`) with SLURM-ready wrappers.
3. **Documentation & references** (`docs/`) for operational and research context.
4. **Artifacts & data** (`artifacts/`, `data/`) for checkpoints, sample imagery, and generated outputs.

---

## Pipelines

| Pipeline | Description | Entry Point | Requirements |
| -------- | ----------- | ----------- | ------------ |
| Tree crowns (DINOv3) | Unsupervised or linear-probe crown delineation using SAT-493M DINOv3 features. | `python -m sege.pipelines.dinov3_tree_crowns` | `requirements/experiments/dinov3.txt` |
| Farmland (SAM 2) | Vegetation-guided SAM 2 inference for field segmentation, with optional tiling. | `python -m sege.pipelines.sam2_farmland` | `requirements/experiments/sam2.txt` |

Detailed deep-dives live under `docs/pipelines/`.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements/experiments/dinov3.txt

# Tree crown segmentation (unsupervised)
python -m sege.pipelines.dinov3_tree_crowns \
  --input data/samples/imagery/1084-1391.tif \
  --device cuda \
  --out_dir artifacts/outputs/dinov3/samples

# Farmland segmentation
pip install -r requirements/experiments/sam2.txt
python -m sege.pipelines.sam2_farmland \
  --input data/samples/imagery/1084-1393.tif \
  --output-dir artifacts/outputs/sam2/samples
```

> **Note**: When running in offline environments, download the required SAM 2 and DINOv3 weights beforehand and set `TRANSFORMERS_OFFLINE=1`. Place checkpoints under `artifacts/checkpoints/`.

For a single-step install with optional dependencies:

```bash
pip install -e .[dinov3,sam2]
```

---

## Repository Layout

```
SegEdge/
├── README.md
├── src/sege/pipelines/        # Production pipelines
├── experiments/               # HPC runners, prototypes, notebooks
├── configs/                   # Pipeline configuration templates
├── requirements/              # Environment definitions
├── data/samples/imagery/      # Lightweight sample GeoTIFFs
├── artifacts/
│   ├── checkpoints/           # Local model weights
│   ├── logs/                  # Structured logs per pipeline
│   └── outputs/               # Generated masks & QA overlays
├── docs/                      # Architecture notes, deep dives, references
├── notebooks/                 # Interactive exploration notebooks
└── third_party/sam2/          # Vendored SAM 2 repository
```

---

## HPC Usage

- **DINOv3 crowns**  
  - Submit `experiments/dinov3/hpc/slurm_dinov3.sh`.  
  - Overrides: export `INPUT_PATH`, `OUTPUT_DIR`, `DEVICE`, etc.  
  - Driver: `experiments/dinov3/hpc/tiling_tree_crowns.py` (YAML-aware).

- **SAM 2 tiling**  
  - Submit `experiments/sam2/hpc/slurm_sam2.sh`.  
  - Overrides: export `IMAGE_PATH`, `CHECKPOINT`, `MODEL_CONFIG`, `OUTPUT_DIR`, `PATCH_SIZE`, `OVERLAP`.  
  - Driver: `experiments/sam2/hpc/tiling_mask_generator.py`.

Ensure the cluster environment has access to the vendored SAM 2 code by adding `third_party/sam2` to the `PYTHONPATH`, or perform an editable install (`pip install -e third_party/sam2`).

---

## Documentation

- `docs/architecture.md` — Overall system design and directory conventions.
- `docs/pipelines/*.md` — Pipeline-specific overviews, requirements, and tuning tips.
- `docs/references/` — Model tables, bibliography, and VRAM sizing handbooks.

---

## Contributing

1. Add new code under `src/sege/` with `main(argv=None)` style entry points.
2. Document the change in `docs/`.
3. Update `requirements/` if the dependency surface changes.
4. Keep artifacts (`artifacts/`) free of sensitive or heavyweight data where possible.

---

## Third-Party Licences

The repository vendors Meta’s SAM 2 implementation under `third_party/sam2/`. Refer to `third_party/sam2/LICENSE` for licensing terms. All other code is released under the licence in the project root.

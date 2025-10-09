# SAM2 Farmland Segmentation

**Module**: `sege.pipelines.sam2_farmland`  
**HPC tiler**: `experiments/sam2/hpc/tiling_mask_generator.py`

---

## Quick Overview
- Segments agricultural fields from multispectral GeoTIFFs using SAM 2 (Hiera-L) inference via Hugging Face.
- Seeds SAM 2 with vegetation-driven positive/negative point prompts and optional bounding boxes derived from ExG/NDWI cues.
- Supports full-scene processing or tile-based chunking for large rasters.
- Produces a binary mask and colour overlay for QA; the tiling driver additionally writes label maps and high-quality PNG exports.

---

## Requirements

### Python packages
- `torch>=2.1`
- `transformers>=4.37`
- `tifffile`
- `Pillow`
- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib` (tiling driver)

Install with:

```bash
pip install -r requirements/experiments/sam2.txt
```

### Model weights
- Hugging Face model ID (`facebook/sam2.1-hiera-large` by default) is downloaded automatically.
- For offline runs, pre-fetch weights with `huggingface-cli download` and set `TRANSFORMERS_OFFLINE=1`.
- The tiling driver can also consume local checkpoints under `artifacts/checkpoints/sam2/`.

### Data prerequisites
- Multispectral GeoTIFF with at least three bands. Defaults assume Sentinel-2 band ordering.
- Vegetation-rich scenes are ideal; NDWI-based filtering suppresses water bodies.

---

## Inputs & CLI Flags

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--input` | GeoTIFF path. | `data/samples/imagery/1084-1389.tif` |
| `--output-dir` | Folder for the binary mask & overlay. | `artifacts/outputs/sam2/inference` |
| `--bands` | Comma-separated 1-based indices mapped to RGB. | `2,4,6` |
| `--positive-points` / `--negative-points` | Number of ExG-ranked points used as prompts. | `20` / `80` |
| `--veg-percentiles` | Percentiles used when sampling vegetation regions. | `90,85,80` |
| `--max-objects` | Max candidate regions when prompting SAM 2. | `150` |
| `--model-id` | Hugging Face identifier for weights. | `facebook/sam2.1-hiera-large` |
| `--min-area` | Remove connected components below this size. | `5000` |
| `--overlay-alpha` | Blend ratio for QA overlay. | `0.6` |
| `--device` | `auto`, `cpu`, or `cuda`. | `auto` |
| `--tile-size` / `--tile-overlap` | Partition large rasters into overlapping windows. | `1024` / `64` |

All arguments are also accepted by `main(argv)` for programmatic invocations.

---

## Processing Pipeline
1. **Argument parsing** — Resolves defaults relative to the repository root and ensures output directories exist.
2. **Data loading** — Reads the GeoTIFF via `tifffile`, normalises channels, and enforces `CHW` layout.
3. **Composite creation** — Builds an RGB composite from selected bands with percentile stretching.
4. **Vegetation heuristics** — Computes ExG and NDWI to propose positive/negative prompts and bounding boxes.
5. **Prompt generation** — `compute_prompts_and_boxes` selects candidate regions, falling back to percentile sampling when vegetation heuristics fail.
6. **SAM 2 inference** — Uses `Sam2Processor` + `Sam2Model` to predict masks for each prompt; optional tiling mitigates memory.
7. **Mask post-processing** — Morphological opening/closing removes noise and fills holes; `min_area` filtering keeps plausible fields.
8. **Artifact saving** — Writes a binary PNG mask plus overlay to `--output-dir` and prints vegetation coverage statistics.

---

## Outputs
- `<stem>_farmland_mask.png` — Binary mask (`0` background, `1` farmland).
- `<stem>_sam2_overlay.png` — RGB overlay combining the composite and coloured mask.
- Coverage metrics emitted via stdout.

The tiling driver additionally produces:
- `comparison_grid.png` — Side-by-side original vs. annotation.
- `annotated.png`, `original.png`, `labels.tif` — High-quality artefacts suitable for reports.

---

## Usage Examples

### Local CLI
```bash
python -m sege.pipelines.sam2_farmland \
  --input data/samples/imagery/1084-1393.tif \
  --bands 4,3,2 \
  --device cuda \
  --output-dir artifacts/outputs/sam2/samples
```

### HPC tiling workflow
```bash
sbatch experiments/sam2/hpc/slurm_sam2.sh \
  --export=IMAGE_PATH=/scratch/farms/scene.tif,OUTPUT_DIR=/scratch/outputs/scene
```
The SLURM script forwards parameters to `tiling_mask_generator.py`, which will automatically create unique run folders under `OUTPUT_DIR`.

---

## Observability & Debugging
- Colourized logging highlights VRAM usage, patch progress, and save operations.
- VRAM snapshots (`print_vram_usage`) are triggered after each patch assignment.
- Intermediate prompts and bounding boxes can be inspected by instrumenting `compute_prompts_and_boxes`.

---

## Failure Modes & Remedies
- **No prompts generated** — Lower `--veg-percentiles` or increase `--positive-points`.
- **Over-segmentation of water bodies** — Adjust NDWI threshold within the script or post-process with land masks.
- **CUDA OOM** — Reduce `--points-per-batch`, enable tiling (`--tile-size`), or switch to `sam2.1-hiera-base-plus`.
- **Crushed small parcels** — Lower `--min-area` or reduce morphological kernel sizes within `clean_mask`.

---

## References
- Kirillov et al., *Segment Anything 2* (2024)  
- Woebbecke et al., *Color Indices for Weed Identification* (ExG formulation)  
- McFeeters, *NDWI for delineating open water features*

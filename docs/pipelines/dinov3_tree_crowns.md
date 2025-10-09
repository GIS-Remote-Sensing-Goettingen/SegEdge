# DINOv3 Tree Crown Segmentation

**Module**: `sege.pipelines.dinov3_tree_crowns`  
**Config template**: `configs/dinov3/default.yaml`

---

## Quick Overview
- Detects individual tree crowns in high-resolution (Geo)TIFF imagery using the SAT-493M DINOv3 backbone.
- Supports fully unsupervised clustering or a 1×1 convolutional linear probe for lightly supervised sites.
- Emits raster masks, GeoPackage vector layers, centroid points, and QA overlays for downstream GIS work.
- Designed as a reusable inference pipeline; the CLI can be driven directly or via the HPC helper at `experiments/dinov3/hpc/tiling_tree_crowns.py`.

---

## Requirements

### Core Python stack
- `torch>=2.1`
- `torchvision` (for weight compatibility)
- `timm`
- `transformers`
- `rasterio`
- `numpy`
- `scikit-image`
- `scikit-learn`
- `scipy`
- `opencv-python`
- `geopandas`
- `shapely`
- `tqdm`

Install with:

```bash
pip install -r requirements/experiments/dinov3.txt
```

### Optional
- `pyproj` (pulled in automatically by `geopandas`).
- Hugging Face CLI if weights are not already cached offline.

### Data prerequisites
- A georeferenced, multi-band GeoTIFF with sufficient spatial resolution to delineate crown edges (≤1 m GSD recommended).
- If running in `linear` mode, provide a serialized 1×1 conv head (produced offline).

### Hardware
- GPU strongly recommended. The GPU k-means path keeps memory under ~4 GiB by sampling/chunking.
- CPU fallback works but runs substantially slower; expects ≥32 GB RAM for large tiles.

---

## Inputs & Configuration

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--input` | Path to the source GeoTIFF (required). | — |
| `--bands` | Comma-separated 1-based bands for pseudo-RGB. Empty uses first three bands. | `""` |
| `--mode` | `unsup` (k-means) or `linear` (requires `--head`). | `unsup` |
| `--head` | Path to a serialized linear probe head (`torch.save`). | `None` |
| `--max-edge` | Downscale the raster so its largest edge ≤ value. | `4096` |
| `--patch` / `--stride` | Tiling parameters for dense feature extraction. | `896` / `640` |
| `--min-area` / `--max-area` | Tree crown pixel area bounds during morphology. | `25` / `10000` |
| `--min-distance` | Watershed seed separation in pixels. | `10` |
| `--device` | `auto`, `cuda`, or `cpu`. | `auto` |
| `--log-file` | Log destination. | `artifacts/logs/dinov3/main.log` |
| `--out_dir` | Artifact output directory. | `artifacts/outputs/dinov3/inference` |

Configuration values can also be expressed in YAML (`configs/dinov3/*.yaml`) and consumed via the HPC driver.

---

## Pipeline Stages
1. **Argument parsing & logging**  
   Configures paths, device selection, and initializes a shared `StageProfiler`.
2. **GeoTIFF ingestion** (`load_geotiff`)  
   Selects requested bands, optionally downsamples oversized rasters, and normalises values to `[0, 1]`.
3. **Dense feature extraction** (`Dinov3SatDenseExtractor`, `extract_dense_features`)  
   Tiles the image, runs DINOv3 forward passes (FP16 on CUDA), and stitches a dense feature cube back to full resolution.
4. **Segmentation head**  
   - **Unsupervised** (`segment_unsupervised`): GPU-aware k-means with scoring heuristics that consider coverage, texture variance, feature magnitude, and greenness. Automatically flips clusters if post-processing yields empty masks.  
   - **Linear probe** (`segment_linear_probe`): Applies a saved 1×1 conv head to produce logits.
5. **Morphological cleanup** (`postprocess_trees`)  
   Opening/closing operations and area filtering to suppress noise and oversized blobs; falls back to complementary clusters if needed.
6. **Instance delineation** (`segment_individual_trees`)  
   Uses a distance transform + watershed to split touching crowns.
7. **Vectorisation** (`extract_tree_polygons`, `extract_tree_points`)  
   Converts labels to GeoDataFrames for crowns, bounding boxes, and centroid points.
8. **QA overlay** (`save_visualizations`)  
   Renders a PNG overlay showing masks, watershed boundaries, and bounding boxes.
9. **Artifact persistence & telemetry**  
   Writes GeoTIFFs/GeoPackages/PNGs to `--out_dir`, logs locations, and prints profiling summary.

---

## Outputs
- `<stem>_tree_mask.tif` — Binary canopy raster.
- `<stem>_tree_labels.tif` — Watershed instance labels.
- `<stem>_tree_crowns.gpkg` — Crown polygons with per-tree attributes.
- `<stem>_tree_bboxes.gpkg` — Axis-aligned bounding boxes per crown.
- `<stem>_tree_points.gpkg` — Centroid points.
- `<stem>_overlay.png` — Visual QA overlay (mask + watershed boundaries + optional boxes).
- `<out_dir>/main.log` — Structured log (if `--log-file` points into the directory).

---

## Usage Examples

### Local CLI
```bash
pip install -e .[dinov3]
python -m sege.pipelines.dinov3_tree_crowns \
  --input data/samples/imagery/1084-1391.tif \
  --bands 4,3,2 \
  --device cuda \
  --out_dir artifacts/outputs/dinov3/samples
```

### Cluster / HPC
```bash
sbatch experiments/dinov3/hpc/slurm_dinov3.sh \
  --export=INPUT_PATH=/scratch/tiles/site1.tif,OUTPUT_DIR=/scratch/runs/site1
```
The SLURM wrapper loads `configs/dinov3/default.yaml` and pushes parameters through the shared pipeline via `experiments/dinov3/hpc/tiling_tree_crowns.py`.
It also exports `PYTHONPATH=$PWD/src` so the `sege` package resolves without a full install. Recreate that step if you run the driver manually (`PYTHONPATH=$PWD/src python experiments/dinov3/hpc/tiling_tree_crowns.py ...`).

---

## Observability
- Timings reported by `StageProfiler.summary()` highlight hotspots.
- `[CUDA]` log lines snapshot VRAM allocations after expensive steps.
- Cluster statistics (`[SEG]`) include coverage, heuristic score, and selected cluster ID.
- Warnings emit when too few crowns survive morphology or heuristics trigger fallback paths.

---

## Failure Modes & Tuning Tips
- **All-zero masks**: decrease `--min-area`, review `--bands`, or supply vegetation-friendly indices.
- **Over-segmentation**: tighten `--max-area` or increase `--min-distance`.
- **Background chosen as trees**: provide `--bands` with vegetation-sensitive channels or use the linear probe mode.
- **Memory pressure**: reduce `--patch`/`--stride` or lower `--max-edge`.
- **CRS warnings**: ensure the input raster is georeferenced; otherwise set CRS manually in post-processing.

---

## Extending the Pipeline
- Swap in alternative transformers by subclassing `Dinov3SatDenseExtractor`.
- Replace clustering with supervised heads by implementing a `segment_*` function and wiring it into the CLI.
- Integrate additional outputs (e.g. height estimates) by augmenting the artifact save section near the end of the module.

---

## References
- Meta AI — *DINOv3: Learning robust visual features without supervision*
- Watershed segmentation (Vincent & Soille, 1991)
- Vegetation indices review (Woebbecke et al., 1995)

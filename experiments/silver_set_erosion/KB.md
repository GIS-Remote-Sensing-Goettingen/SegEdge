# Project Knowledge Base (for a new LLM)

This document gives a complete, self-contained description of the SegEdge zero-shot LWF segmentation pipeline: objectives, data, implementation, configs, performance, known issues, and reproduction steps. It assumes no prior knowledge.

---

## 1) Scope & Audience
- Audience: a new LLM/engineer with zero context.
- Goal: explain what the pipeline does, how it’s structured, how to run/evaluate it, and what has failed or been fixed.
- Source tree: `/home/mak/PycharmProjects/SegEdge/experiments/silver_set_erosion`.
- Style: practical “how it works + how to run + what to avoid”, not marketing.
- Environment: see Section 12 for dependencies/versions and run recipes.

---

## 2) High-Level Overview
- Objective: Zero-shot segmentation of linear woody features (LWFs) on a target orthoimage (Image B) by transferring semantics from a labeled reference image (Image A).
- Core idea: Use DINOv3 patch embeddings to represent texture/structure; build positive/negative patch banks from A; score B via kNN or a learned XGBoost patch classifier; refine with CRF and a shadow filter.
- Pipeline stages:
  1. Load imagery + SH_2022 raster + GT vector labels.
  2. Buffer SH_2022 to constrain predictions.
  3. Extract/cached DINO features (patch size 16).
  4. Build banks (positives/negatives) from A.
  5. Score B with kNN (pos − α·neg) over k grid; sweep thresholds.
  6. Optional XGBoost classifier on patch features; sweep thresholds.
  7. Choose champion (best IoU between kNN and XGB).
  8. Median filter champion mask; CRF search around champion.
  9. Shadow filtering (RGB weighted dark-pixel removal).
  10. Export plots, shapefiles, and best-settings YAML.
- Design choices: zero-shot (no fine-tuning of backbone), aggressive caching, grid-search for self-calibration, post-hoc refinements (CRF, shadow) to clean edges/dark areas.
- Backbone detail: DINOv3 ViT-L/16 (sat493m pretrain), embedding dim ~1024, patch grid size Hp×Wp where Hp = H/16, Wp = W/16 on each tile.

---

## 3) Data & Labeling Details
- Inputs (config.py):
  - Image A: `data/dop20_593000_5979000_1km_20cm.tif`
  - Image B: `data/dop20_592000_5982000_1km_20cm.tif`
  - SH_2022 raster: `data/planet_labels_2022.tif`
  - GT vector (B): `data/labels_final.shp`
- Reprojection: `io_utils.reproject_labels_to_image` aligns raster labels to image grids; `rasterize_vector_labels` rasterizes GT.
- Buffer: `build_sh_buffer_mask` dilates SH_2022 by `BUFFER_M` meters (converted to pixels). Optional `CLIP_GT_TO_BUFFER` masks GT outside buffer to allow 100% IoU inside the allowed region.
- Tiling: default `TILE_SIZE=1024`, `STRIDE=512`; tiles cropped to multiples of patch size (16).
- Patch labeling for A: `labels_to_patch_masks` marks a patch positive if FG fraction ≥ `POS_FRAC_THRESH` (default 0.1); negative if zero FG.
- CRS: Vector CRS is reprojected to raster CRS when needed; if vector CRS missing, assumes EPSG:4326 with a warning.
- All downstream masks can be clipped to the SH buffer to enforce spatial priors.
- Typical buffer: 8 m → at 0.2 m/pixel, buffer_pixels ≈ 40 (configurable).
- Label erosion: banks use a slight erosion (disk radius 2) to avoid noisy boundaries when picking positives.
- Pos/neg counts: After subsampling, typical banks: ~6–10k positives, up to `MAX_NEG_BANK` negatives (default 8k).

---

## 4) Model Components
- **DINOv3**: HuggingFace `facebook/dinov3-vitl16-pretrain-sat493m`; patch size 16; features L2-normalized; optional FP16 matmul for kNN.
- **kNN scorer** (`knn.py`):
  - Score = mean top-k pos sims − α·mean top-k neg sims (`NEG_ALPHA`).
  - FP16 matmul optional; negative bank subsampling (`MAX_NEG_BANK`).
  - Threshold sweep via GPU/CPU batch metrics.
- **XGBoost classifier** (`xdboost.py`):
  - Trains on patch features from A (pos/neg); optional hyperparam search by IoU on B (`hyperparam_search_xgb_iou`).
  - Params grid in `config.XGB_PARAM_GRID`; uses logloss for training/early stop but selection by IoU on B after threshold sweep.
- **Champion selection**: Pick the model (kNN or XGB) with best IoU on B; its score map/threshold drive CRF.
- **Median filter**: 3×3 applied after thresholding (kNN and XGB) to remove speckle.
- **Bank sizes (typical)**: Positives ~6–10k patches; negatives capped at `MAX_NEG_BANK` (default 8000) to control memory.
- **DINO feature storage**: per-tile `.npy` (Hp×Wp×1024), consolidated full arrays for A/B for analysis.
- **Champion rule**: Compare best IoU from kNN (post-median) vs XGB (post-median); the higher IoU becomes champion for CRF and onward.

---

## 5) Post-Processing & Refinement
- **Thresholding**: Coarse grid over `THRESHOLDS`; fine-tune locally for kNN; median filter (size=3) applied to masks (kNN and XGB) to reduce speckle.
- **CRF** (`crf_utils.py`):
  - Unary = logistic(score − thr_center)/prob_softness; respects SH buffer (outside forced near-zero FG).
  - Pairwise Gaussian + bilateral (color/XY).
  - Grid search over softness/weights/sigmas; optional downsample; process-based parallelism.
- **Shadow filter** (`shadow_filter.py`):
  - Under the CRF mask, compute weighted RGB sums; sweep thresholds to drop dark pixels; select best IoU.
- **Resize strategy**: Patch scores are resized to pixel grid with bilinear (order=1, anti_aliasing=True) before accumulation; overlaps averaged by weights.

---

## 6) Configuration & Runtime Controls (config.py)
- Paths: `SOURCE_TILE`, `TARGET_TILE`, `SOURCE_LABEL_RASTER`, `EVAL_GT_VECTOR`/`EVAL_GT_VECTORS`, `FEATURE_DIR`, `BANK_CACHE_DIR`, `PLOT_DIR`, `BEST_SETTINGS_PATH`.
- Geometry: `BUFFER_M`, `TILE_SIZE`, `STRIDE`, `PATCH_SIZE`.
- Banks/Scoring: `NEG_ALPHA`, `POS_FRAC_THRESH`, `MAX_NEG_BANK`.
- kNN grid: `K_VALUES`, `THRESHOLDS`, `USE_FP16_KNN`, `USE_GPU_THRESHOLD_METRICS`, batch sizes.
- CRF grid: softness/weights/sigmas, `CRF_NUM_WORKERS`, `CRF_MAX_CONFIGS` (from main).
- XGBoost: `XGB_USE_GPU`, `XGB_PARAM_GRID`, `XGB_NUM_BOOST_ROUND`, `XGB_EARLY_STOP`, `XGB_VERBOSE_EVAL`, `XGB_VAL_FRACTION`.
- Shadow: `SHADOW_WEIGHT_SETS`, `SHADOW_THRESHOLDS`.
- Evaluation: `CLIP_GT_TO_BUFFER`.
- Plots: saved to `PLOT_DIR`, with overlays for kNN vs XGB vs GT, champion pre-CRF, and raw/CRF/shadow panels.
- kNN neg weighting: `NEG_ALPHA` controls subtraction of negative bank similarity; set to 1.0 by default; lowering reduces penalty from negatives.

---

## 7) Code Architecture (modules)
- `main.py`: Orchestrator—loads data, builds banks, runs kNN + XGB, selects champion, plots, CRF, shadow filter, exports shapefiles/YAML.
- `config.py`: All tunable paths/hyperparams.
- `timing_utils.py`: Timing helpers + debug flags.
- `features.py`: Tiling, cropping, DINO feature extraction, prefetch, normalization.
- `banks.py`: Build/load/save positive/negative banks; subsample negatives.
- `knn.py`: kNN scoring, threshold sweep, fine-tune threshold.
- `metrics_utils.py`: Metrics (IoU/F1/P/R) and batched GPU/CPU threshold eval; oracle upper bound.
- `crf_utils.py`: DenseCRF refine and grid search.
- `io_utils.py`: I/O (load images, reproject, rasterize, buffer, shapefile export, feature consolidation, best-settings YAML).
- `plotting.py`: Plots for raw/CRF/shadow and GT/kNN/XGB overlays.
- `shadow_filter.py`: Weighted-sum dark-pixel filtering under mask.
- `xdboost.py`: Build XGB dataset, train, hyperparam search by IoU on B, score image B.
- `KB.md`: This document.

---

## 8) Caching, Performance, Memory
- Tile features cached under `FEATURE_DIR`; banks cached under `BANK_CACHE_DIR`.
- Prefetch features for B to avoid repeated disk I/O.
- FP16 matmul for kNN to speed up GPU.
- CPU/GPU fallbacks for threshold metrics; CRF parallelized with process pool; optional downsample for CRF search.
- Median filtering reduces tiny artifacts before CRF.
- Memory cleanup before CRF: delete large intermediates and `torch.cuda.empty_cache()`.
- Known bottlenecks: resizing per tile, CRF iterations; potential GPU OOM with XGB `gpu_hist` or large CRF grids.
- Threshold metrics can fallback to CPU if GPU OOM; CRF search can be downsampled to reduce load.
- Disk I/O considerations: feature caches can be large (Hp×Wp×1024 per tile); consolidated feature files can reach hundreds of MB—ensure sufficient disk.
- XGB GPU support may be absent; code auto-falls back to CPU hist; training time rises but still workable with reduced rounds/depth.
- Cache hygiene: If features become inconsistent, delete per-tile caches under `FEATURE_DIR` for the affected image(s) and rerun; banks will rebuild.
- SH buffer sizing: Too small → drop valid GT; too large → allow spurious FPs; tune `BUFFER_M` relative to pixel size.

---

## 9) Experiments & Failures (notable issues encountered)
- XGBoost `gpu_hist` unsupported on this build → auto-fallback to `hist`.
- Earlier XGB shape mismatch (inconsistent tile sizes) fixed by cropping to patch multiples and padding.
- Superpixels were removed (not helpful, caused errors).
- CRF initially single-core; now process-based grid search.
- IoU sometimes zero when SH buffer or CRS mismatch; GT clipping option added.
- Median filter added post-threshold to reduce speckle.
- Plot titles were clipped; fixed with `bbox_inches="tight"`.
- Memory pressure before CRF addressed by deleting banks/prefetch caches and emptying CUDA cache.
- Shadow filtering added to drop dark artifacts after CRF; weights/thresholds tunable.
- Observed: resize time dominates kNN stage more than matmul; FP16 matmul yields small speedup but resize remains the bottleneck.
- Observed: CRF can take ~30–40s per config on full-res; downsample=2 reduces cost substantially at slight accuracy loss.
- Observed: kNN IoU improves with larger k up to a point; XGB can surpass kNN when param grid tuned; champion selection handles this automatically.
- Observed: If SH buffer is misaligned (CRS mismatch), IoU collapses to ~0; verify CRS logs and buffer stats early.
- Observed: Sparse GT outside buffer can cap max IoU; enable `CLIP_GT_TO_BUFFER` to get meaningful scores.

---

## 10) How to Reproduce & Evaluate
1. Ensure deps installed (PyTorch, transformers, xgboost, pydensecrf, rasterio, skimage, scipy).
2. Set paths/hyperparams in `config.py` as needed.
3. Run `python main.py` from the repo root.
4. Outputs:
   - Plots in `data/plots/` (raw vs CRF vs shadow; GT vs kNN vs XGB; champion pre-CRF).
   - Shapefiles: `*_pred_mask_best_raw.shp`, `*_pred_mask_best_crf.shp`, `*_pred_mask_best_shadow.shp`.
   - Best settings YAML: `BEST_SETTINGS_PATH`.
   - Consolidated features: `FEATURE_DIR/{image_id}_features_full.npy`.
5. Evaluation metric: IoU (primary), plus F1/P/R—computed on B; if `CLIP_GT_TO_BUFFER=True`, GT is masked to SH buffer so max IoU can reach 1.0.
6. Champion selection: compare best IoU from kNN vs XGB (after median filter); champion feeds CRF; shadow filter runs after CRF.
- Plots to inspect:
   - `*_knn_vs_xgb.png`: GT, kNN overlay, XGB overlay.
   - `*_champion_pre_crf.png`: champion vs GT before CRF.
   - `*_raw_crf.png`: raw vs CRF vs shadow panels.
- Shapefiles to consume: `_pred_mask_best_raw`, `_pred_mask_best_crf`, `_pred_mask_best_shadow`.
- Best-settings YAML records champion configs and context (paths, buffer, pixel size, tiling, neg_alpha, pos_frac).
- Typical run flow in main: build banks → prefetch B → kNN grid → fine-tune → median filter → XGB IoU search → overlays → CRF → shadow → exports.
- Logs: Main stdout includes timing, kNN evals, XGB search logs, CRF evals; plots show overlays; YAML captures configs.

---

## 11) Quick Reference (key functions)
- Feature extraction: `features.extract_patch_features_single_scale`, `prefetch_features_single_scale_image`.
- Banks: `banks.build_banks_single_scale`.
- kNN: `knn.zero_shot_knn_single_scale_B_with_saliency`, `grid_search_k_threshold`, `fine_tune_threshold`.
- XGB: `xdboost.build_xgb_dataset`, `hyperparam_search_xgb_iou`, `xgb_score_image_b`.
- Metrics: `metrics_utils.compute_metrics_batch_gpu/cpu`, `compute_metrics`.
- CRF: `crf_utils.crf_grid_search`, `refine_with_densecrf`.
- Shadow: `shadow_filter.shadow_filter_grid`.
- Plots: `plotting.save_plot`, `save_knn_xgb_gt_plot`, `save_best_model_plot`.
- Exports: `io_utils.export_mask_to_shapefile`, `export_best_settings`.

---

## 12) Environment, Dependencies, and Run Recipes
- Python: 3.12 (current environment). Key libs (install via pip/conda as available, no network inside sandbox):
  - torch (with CUDA if available), transformers, xgboost, pydensecrf, rasterio, fiona, shapely, pyproj, skimage, scipy, matplotlib, numpy.
  - If xgboost lacks GPU, it will auto-fall back to CPU (`tree_method=hist`).
- Run recipes:
  - Full pipeline (kNN + XGB + CRF + shadow): `python main.py`
  - Disable XGB search: set `XGB_PARAM_GRID = []` or `None` in `config.py`.
  - Force CPU for XGB: `XGB_USE_GPU = False`.
  - Speed CRF: increase `downsample_factor` (in `crf_grid_search` call) or reduce `CRF_MAX_CONFIGS`.
  - kNN-only comparison: set `XGB_PARAM_GRID = []` and inspect plots/shapefiles.
- Outputs to expect:
  - Plots in `PLOT_DIR`: `*_knn_vs_xgb.png`, `*_champion_pre_crf.png`, `*_raw_crf.png`.
  - Shapefiles: `_pred_mask_best_raw.shp`, `_pred_mask_best_crf.shp`, `_pred_mask_best_shadow.shp`.
- YAML: `BEST_SETTINGS_PATH` with champion configs + context.
- Consolidated features: `{image_id}_features_full.npy` under `FEATURE_DIR`.

---

## 13) Tuning Tips
- kNN: Increase k to smooth noise, but watch IoU; adjust `NEG_ALPHA` (lower to reduce neg penalty) and `POS_FRAC_THRESH` (raise to tighten positives, lower to include more).
- XGB: Reduce `max_depth`/`num_boost_round` if overfitting or slow; increase `colsample_bytree` if underfitting; shrink grid for speed.
- CRF: Use `downsample_factor=2` for speed; tune `prob_softness` for sharper/softer unary; reduce bilateral weights if over-smoothing colors.
- Buffer: Tune `BUFFER_M` to balance FP suppression vs missing off-buffer GT; enable `CLIP_GT_TO_BUFFER` when GT outside buffer is sparse/noisy.
- Shadow: Adjust `SHADOW_WEIGHT_SETS`/`SHADOW_THRESHOLDS` if dark regions are over-removed or under-removed.

---

## 14) Failure/Edge Case Handling
- CRS issues: If IoU ~0, verify CRS logs (vector reproject warning) and SH buffer alignment.
- Cache corruption: Delete affected tile features in `FEATURE_DIR` and rerun; banks will rebuild.
- GPU OOM: Switch XGB to CPU (`XGB_USE_GPU=False`), reduce `num_boost_round`/`max_depth`; downsample CRF and reduce `CRF_MAX_CONFIGS`; ensure FP16 matmul is optional.
- Missing features: If a tile .npy is absent, XGB scorer skips it; recompute features for completeness.

---

## 15) Results Snapshot (fill in with actual runs)
- Maintain a small table of recent runs (config hash/date):
  - kNN best IoU/F1, XGB best IoU/F1, CRF IoU/F1, Shadow IoU/F1.
  - Note key params: BUFFER_M, NEG_ALPHA, K_VALUES range, XGB grid summary, CRF downsample, SHADOW weights.

---

## 16) Exports & Schemas
- Shapefiles: Polygons of predicted FG; schema has a single `id` int field; CRS matches the reference raster of Image B.
- Best-settings YAML: Captures `best_raw_config`, `best_crf_config`, optional `best_shadow_config`, model name, image paths, buffer_m, pixel_size_m, tiling/patch sizes, neg_alpha, pos_frac, and champion params.
- Feature consolidations: `{image_id}_features_full.npy` are (N_patches × C) arrays; useful for offline analysis or alternate classifiers.

---

## 17) Logging & Visualization Conventions
- Stdout logs timing blocks, kNN eval lines `[eval-raw]`, XGB search logs `[xgb-search-iou]`, CRF evals `[crf-eval]`, and warnings (GPU fallbacks, CRS info).
- Plot colors: GT overlay uses blue (0,0,255); kNN uses green (0,255,0); XGB uses red (255,0,0); champion pre-CRF uses red overlay; shadow overlay in `save_plot` uses yellow (255,255,0).
- Plots are saved with `bbox_inches="tight"` to avoid clipping titles.

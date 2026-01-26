# DINOv3Seg Experiments

This repository provides a research-grade segmentation pipeline that keeps a frozen **DINOv3** backbone and swaps different decoder heads (classic U-Net, SPM/FAPM-enhanced U-Net, and a MaskFormer-style transformer). The pipeline now runs entirely from a YAML configuration file, supports structured verbosity with timestamps, and records durations for every major phase.

## Quick Start

1. **Copy the example config** and tailor it to your environment:
   ```bash
   cp config.example.yml config.yml
   ```
   Update the paths (raw imagery, labels, cache directory), toggle phases (`prepare`, `verify`, `train`, `inference`), and adjust hyperparameters or decoder selection under the `model` section.

2. **Run the pipeline**:
   ```bash
   python main.py config.yml
   ```
   For multi-GPU training set `resources.distributed: true` and launch with e.g.
   ```bash
   torchrun --standalone --nproc_per_node=4 main.py config.yml
   ```
   Only rank 0 prints logs and runs inference; validation metrics are computed on rank 0 and broadcast to the others. If no argument is provided the script checks the first CLI argument, then `$DINOV3SEG_CONFIG`, and finally searches upward for `config.yml`.

3. **Observe logs**: The logger honors three verbosity levels (`error`, `info`, `debug`), can print timestamps, and optionally mirrors output to a log file. Configure it via the `logging` block.

## Configuration Reference

The YAML file drives everything. Each section mirrors a phase and shares defaults if a specific value is missing.

```yaml
resources:
  seed: 1337
  omp_threads: 4
  matmul_precision: high
  distributed: false        # set true when launching with torchrun
  dist_backend: nccl        # backend for DDP

logging:
  level: info
  timestamps: true
  file: logs/run.log

paths:
  raw_images_dir: /path/to/imagery
  label_path: /path/to/labels.tif
  processed_dir: /path/to/cache

dataset:
  augmentations:
    enable: true
    hflip: true
    vflip: true
    rotate90: true
  splits:
    train_list: splits/train.txt
    val_list: splits/val.txt

model:
  backbone: facebook/dinov3-vitl16-pretrain-sat493m
  layers: [5, 11, 17, 23]
  head: unet_v2          # unet | unet_v2 | maskformer
  num_classes: 2
  dino_channels: 1024

prepare:
  enable: true
  tile_size: 512
  device: cuda

verify:
  enable: true
  workers: 8

train:
  enable: true
  processed_dir: /path/to/cache
  weights_dir: weights
  batch_size: 4
  epochs: 30
  muon_lr: 0.02
  adamw_lr: 0.001
  momentum: 0.95
  patience: 10
  val_fraction: 0.2
  num_workers: 4
  grad_accum_steps: 1
  compile: false
  ema_decay: 0.0
  loss:
    ce_weight: 1.0
    dice_weight: 1.0
    aux_weight: 0.4

inference:
  enable: false
  input_tif: /path/to/scene.tif
  output_tif: test/output_prediction.tif
  checkpoint: weights/unet_v2_best.pth
  tile_size: 512
  overlap: 0.25
  tta:
    horizontal_flip: true
    vertical_flip: false
```

Set `enable: true` for any section you want to run. The `paths` block provides base directories shared across phases, while individual sections can override them (e.g., use a different `processed_dir` for training vs. verification).

## Logging & Timing

- The custom `VerbosityLogger` prints `[LEVEL] message` lines with optional timestamps and can also tee logs to disk (`logging.file`).
- Every phase (`prepare`, `verify`, `train`, `inference`) runs inside a `TimedBlock`, so you see start/finish messages and durations.
- Inner loops log progress periodically (e.g., every 10 training batches or every 50 inference tiles) when verbosity permits.

## Distributed Training

- Set `resources.distributed: true` in the config and launch via `torchrun --standalone --nproc_per_node=<gpus> main.py config.yml`.
- Training uses `DistributedDataParallel` with `DistributedSampler`; rank 0 handles logging, validation loops, checkpointing, and inference while other ranks stay silent and focus on SGD.
- Validation metrics (loss, mIoU) and early-stopping signals are broadcast to every rank so they can exit cleanly at the same epoch. Inference automatically runs only on rank 0 to avoid duplicate outputs.

## Decoder Registry

The `model.head` key selects one of the decoders registered under `models/`:

| Head        | File             | Highlights                                                        |
|-------------|------------------|-------------------------------------------------------------------|
| `unet`      | `models/unet.py` | Baseline DinoUNet with stacked UpBlocks and raw-image skip.       |
| `unet_v2`   | `models/unet_v2.py` | Adds Spatial Prior Module + Fidelity-Aware projections + deep supervision. |
| `maskformer`| `models/maskformer.py` | Pixel decoder fused with transformer mask head (MaskFormer style).       |

Adding a new decoder only requires implementing `SegmentationHead`, registering it in `models/__init__.py`, and referencing it via `model.head`.

## Utilities

- `utils/data.py` handles tiling, label alignment, feature extraction, cache verification, and the `PrecomputedDataset`. It also applies optional train-time augmentations (flips/rotations) that keep cached features & labels aligned and supports region-based splits.
- `utils/losses.py` implements the combined CE + Dice loss used for the main head and auxiliary deep supervision.
- `utils/metrics.py` accumulates per-class IoU/Dice; we early-stop on validation mIoU instead of loss.
- `utils/optim.py` contains the Muon optimizer (matrix-aware momentum with orthogonalization), AdamW handling, and a configurable EarlyStopping helper that works for min/max metrics.
- `utils/logging.py` exposes the verbosity logger (`stdout` + optional file) and `TimedBlock` context manager.
- `config.py` reads the YAML file, honors the `$DINOV3SEG_CONFIG` override, and searches upward from the working directory if no path is provided.

- **Training extras:** gradient accumulation, optional `torch.compile`, Muon+AdamW with OneCycleLR, model EMA, CE+Dice loss, and validation metrics (mIoU/mDice) that drive early stopping.
- **Inference extras:** sliding-window streaming directly from disk, configurable overlap with probability blending, AMP, and optional flip-based test-time augmentation.

## Testing

Every function ships with doctests to keep behavior well documented. Run them with:

```bash
python -m doctest main.py
PYTHONPATH=. python - <<'PY'
import doctest, importlib
for mod in [
    "models.base",
    "models.unet",
    "models.unet_v2",
    "models.maskformer",
    "models.__init__",
    "utils.data",
    "utils.optim",
    "utils.logging",
    "utils.losses",
    "utils.metrics",
    "config",
]:
    doctest.testmod(importlib.import_module(mod))
PY
```

## Dependencies

- PyTorch (CUDA build recommended)
- `transformers`
- Geospatial stack: `rasterio`, `tifffile`, `shapely`
- Misc: `numpy`, `tqdm`, `PyYAML`

Install via:

```bash
pip install torch torchvision transformers rasterio tifffile shapely tqdm pyyaml
```

## Notes

- Large imagery and label rasters should share a CRS; the tiling pipeline reprojects labels when needed.
- Cache verification deletes unreadable `.pt` files, so rerun `prepare` if the dataset was partially generated.
- Inference now streams tiles from disk, supports overlapping windows with probability blending, runs under AMP, and can average flip-based TTA predictions.
- Cached tiles are used for training while inference recomputes DINO features on the fly.

With the YAML-driven approach, you can version-control experiment configs, schedule recurring training jobs, and keep logs consistent across runs.

"""
Config-driven training and inference entry point with pluggable heads.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import random
import copy
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import yaml
from transformers import AutoImageProcessor, AutoModel

from models import build_head
from utils import (
    EarlyStopping,
    Muon,
    PrecomputedDataset,
    SegmentationLoss,
    SegmentationMetrics,
    TimedBlock,
    VerbosityLogger,
    extract_multiscale_features,
    load_config,
    prepare_data_tiles,
    verify_and_clean_dataset_fast,
)
# Prevent CPU hangs and memory fragmentation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_RAW_IMAGES_DIR = "//mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt/folder_1/"
DEFAULT_LABEL_PATH = "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
DEFAULT_PROCESSED_DIR = "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/folder_1/"
DEFAULT_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
DEFAULT_LAYERS = [5, 11, 17, 23]
DEFAULT_HEAD = "unet"
DEFAULT_NUM_CLASSES = 2
DEFAULT_DINO_CHANNELS = 1024
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DistContext:
    def __init__(self, enabled=False, rank=0, world_size=1, local_rank=0):
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def is_main(self) -> bool:
        return not self.enabled or self.rank == 0


def setup_distributed(resources_cfg: dict) -> DistContext:
    dist_flag = resources_cfg.get("distributed", False)
    ctx = DistContext()
    if not dist_flag:
        return ctx
    if not dist.is_available():
        raise RuntimeError("distributed training requested but torch.distributed unavailable")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        raise RuntimeError("Distributed mode requires torchrun/launch to set RANK and WORLD_SIZE")
    backend = resources_cfg.get("dist_backend", "nccl")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    ctx.enabled = True
    ctx.rank = rank
    ctx.world_size = world_size
    ctx.local_rank = local_rank
    return ctx


def cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def apply_resource_config(config: dict) -> None:
    """
    Apply thread, seed, and precision settings from the config.
    """

    res_cfg = config.get("resources", {})
    threads = res_cfg.get("omp_threads")
    if threads:
        for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[env_var] = str(threads)
    precision = res_cfg.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(precision)
    seed = res_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = res_cfg.get("cudnn_benchmark", True)


def build_logger(config: dict, enabled: bool = True) -> VerbosityLogger:
    """
    Create a VerbosityLogger using the configuration's logging section.

    >>> logger = build_logger({"logging": {"level": "debug", "timestamps": False}})
    >>> logger.debug("configured")
    [DEBUG] configured
    """

    logging_cfg = config.get("logging", {})
    level = logging_cfg.get("level", "info")
    timestamps = logging_cfg.get("timestamps", True)
    log_file = logging_cfg.get("file")
    return VerbosityLogger(level=level, timestamps=timestamps, log_file=log_file, enabled=enabled)


def section_enabled(config: dict, name: str) -> bool:
    """Return True if the named section has enable=true."""

    section = config.get(name, {})
    return bool(section.get("enable", False))


def resolve_path(config: dict, section: dict, key: str, fallback: str) -> str:
    """
    Resolve a path from a section, falling back to global paths or defaults.

    >>> cfg = {"paths": {"processed_dir": "/tmp/proc"}}
    >>> resolve_path(cfg, {"processed_dir": "/custom"}, "processed_dir", "/default")
    '/custom'
    """

    paths_cfg = config.get("paths", {})
    return section.get(key) or paths_cfg.get(key) or fallback


def get_model_config(config: dict) -> dict:
    """Ensure model sub-config always exists with defaults."""

    model_cfg = config.get("model", {})
    return {
        "backbone": model_cfg.get("backbone", DEFAULT_MODEL_NAME),
        "layers": model_cfg.get("layers", DEFAULT_LAYERS),
        "head": model_cfg.get("head", DEFAULT_HEAD),
        "num_classes": model_cfg.get("num_classes", DEFAULT_NUM_CLASSES),
        "dino_channels": model_cfg.get("dino_channels", DEFAULT_DINO_CHANNELS),
    }


def _file_stem(path: str) -> str:
    return Path(path).stem


def _read_name_list(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split list not found: {path}")
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yml", ".yaml", ".json"}:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            combined = []
            for value in data.values():
                if isinstance(value, list):
                    combined.extend(value)
            return [str(item).strip() for item in combined if str(item).strip()]
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_dataset_splits(
    processed_dir: str,
    split_cfg: dict,
    val_fraction: float,
    logger: VerbosityLogger,
) -> tuple[List[str], List[str]]:
    all_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    if not all_files:
        raise ValueError(f"No cached tiles found in {processed_dir}")
    if split_cfg.get("train_list"):
        train_names = set(_read_name_list(split_cfg["train_list"]))
        train_files = [f for f in all_files if _file_stem(f) in train_names]
        if split_cfg.get("val_list"):
            val_names = set(_read_name_list(split_cfg["val_list"]))
            val_files = [f for f in all_files if _file_stem(f) in val_names]
        else:
            val_files = [f for f in all_files if f not in train_files]
        if not train_files or not val_files:
            raise ValueError("Split lists produced empty train/val subsets.")
        return train_files, val_files
    files = all_files.copy()
    random.shuffle(files)
    split_idx = max(1, int(len(files) * (1 - val_fraction)))
    train_files = files[:split_idx]
    val_files = files[split_idx:] or files[-1:]
    logger.info(
        f"Using random split with {len(train_files)} train and {len(val_files)} validation tiles."
    )
    return train_files, val_files


class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.decay = decay

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_params = dict(self.ema_model.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
            ema_buffers = dict(self.ema_model.named_buffers())
            for name, buf in model.named_buffers():
                if name in ema_buffers:
                    ema_buffers[name].copy_(buf)


def prepare_phase(config: dict, logger: VerbosityLogger) -> None:
    """
    Run the tiling and feature-caching phase if enabled.

    This phase is conceptually:
      - INPUT:  raw imagery (big GeoTIFFs) + label raster
      - OUTPUT: small .pt files with:
                  * RGB tiles
                  * labels per tile
                  * precomputed DINO multiscale features
      - PURPOSE: expensive I/O + DINO forward passes are done once,
                 so train/inference can be fast and CPU/GPU friendly.
    """

    # Check feature toggle in the config:
    # if `prepare.enable: false`, we simply skip this phase.
    if not section_enabled(config, "prepare"):
        logger.debug("Prepare phase disabled.")
        return

    # `section` contains only the "prepare" subsection of the config.
    section = config.get("prepare", {})

    # Model config is needed because we must know which backbone and layers
    # to use when extracting features during tiling.
    model_cfg = get_model_config(config)

    # Resolve paths for:
    #   - input image directory (raw patches or large rasters)
    #   - label raster (same CRS/resolution as imagery)
    #   - output directory for cached tiles (.pt)
    img_dir = resolve_path(config, section, "img_dir", DEFAULT_RAW_IMAGES_DIR)
    label_path = resolve_path(config, section, "label_path", DEFAULT_LABEL_PATH)
    output_dir = resolve_path(config, section, "output_dir", DEFAULT_PROCESSED_DIR)

    # Device for feature extraction. Typically `cuda` for speed, but can be
    # configured to `cpu` for systems without GPU.
    device = torch.device(section.get("device", DEFAULT_DEVICE))
    if dist_ctx.enabled:
        device = torch.device(f"cuda:{dist_ctx.local_rank}")

    # TimedBlock is a small profiling/logging helper so we can see how long
    # the preparation phase takes (useful on large datasets).
    with TimedBlock(logger, "Preparation phase"):
        # Core worker that does:
        #   - sliding window over imagery
        #   - alignment with labels
        #   - backbone forward for feature extraction
        #   - serializing everything into .pt files on disk
        prepare_data_tiles(
            img_dir=img_dir,
            label_path=label_path,
            output_dir=output_dir,
            model_name=model_cfg["backbone"],
            layers=model_cfg["layers"],
            device=device,
            tile_size=section.get("tile_size", 512),
            logger=logger,
        )



def verify_phase(config: dict, logger: VerbosityLogger) -> None:
    """
    Run cache verification if enabled.

    This phase is conceptually:
      - INPUT:  directory of cached .pt tiles
      - OUTPUT: same directory, but with problematic tiles removed/fixed
      - PURPOSE: catch corrupt files, shape mismatches, NaNs, etc.
                 before training, so training loops don't crash after
                 hours of work.
    """

    # Again, controlled via `verify.enable` in the config. Skipped if off.
    if not section_enabled(config, "verify"):
        logger.debug("Verify phase disabled.")
        return

    # Local section for "verify" config knobs.
    section = config.get("verify", {})

    # Where cached tiles live. This usually matches the output_dir used in
    # the preparation phase, but can be overridden.
    processed_dir = resolve_path(config, section, "processed_dir", DEFAULT_PROCESSED_DIR)

    # Wrap the verification step in a timer for logging and profiling.
    with TimedBlock(logger, "Verification phase"):
        # This helper will typically:
        #   - iterate over .pt files
        #   - try loading them
        #   - validate basic invariants (shapes, dtypes, presence of keys)
        #   - remove or log any broken tiles
        verify_and_clean_dataset_fast(
            processed_dir,
            num_workers=section.get("workers"),
            logger=logger,
        )



def create_dataloaders(
    processed_dir: str,
    dataset_cfg: dict,
    train_cfg: dict,
    batch_size: int,
    logger: VerbosityLogger,
    dist_ctx: DistContext,
) -> tuple[DataLoader, Optional[DistributedSampler], Optional[DataLoader]]:
    augment_cfg = dataset_cfg.get("augmentations", {})
    split_cfg = dataset_cfg.get("splits", {})
    val_fraction = train_cfg.get("val_fraction", 0.2)
    train_files, val_files = resolve_dataset_splits(processed_dir, split_cfg, val_fraction, logger)
    train_dataset = PrecomputedDataset(
        processed_dir,
        augmentation_cfg=augment_cfg,
        file_subset=train_files,
    )
    train_sampler = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
        )
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if (not dist_ctx.enabled) or dist_ctx.is_main:
        val_dataset = PrecomputedDataset(
            processed_dir,
            augmentation_cfg={"enable": False},
            file_subset=val_files,
        )
        val_workers = train_cfg.get("val_workers", max(1, num_workers // 2))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            persistent_workers=val_workers > 0,
        )
    return train_loader, train_sampler, val_loader


def move_features_to_device(features: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    """
    Clone and push cached feature tensors to the target device.

    >>> feats = [torch.ones(1, 2, 2, 2)]
    >>> move_features_to_device(feats, torch.device("cpu"))[0].device.type
    'cpu'
    """

    return [f.to(device) for f in features]


def align_labels_to_logits(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Align label tensor spatial dimensions with logits using nearest interpolation.

    >>> y = torch.zeros(1, 2, 2).long()
    >>> logits = torch.zeros(1, 2, 4, 4)
    >>> align_labels_to_logits(y, logits).shape
    torch.Size([1, 4, 4])
    """

    if y.ndim == 2:
        y = y.unsqueeze(0)
    if logits.shape[-2:] == y.shape[-2:]:
        return y
    y_expanded = y.unsqueeze(1).float()
    aligned = F.interpolate(y_expanded, size=logits.shape[-2:], mode="nearest")
    return aligned.squeeze(1).long()


def split_params_for_muon(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Split parameters into Muon-compatible (>=2D) and AdamW (1D) tensors.

    >>> module = torch.nn.Linear(4, 4)
    >>> muon_params, adamw_params = split_params_for_muon(module)
    >>> all(p.ndim >= 2 for p in muon_params)
    True
    """

    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []
    for _, p in model.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def evaluate(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    loss_fn: SegmentationLoss,
    device: torch.device,
    use_amp: bool,
    logger: VerbosityLogger | None = None,
    num_classes: int = 2,
) -> tuple[float, dict]:
    """
    Evaluate the model on the validation set.

    >>> class Dummy(torch.nn.Module):
    ...     def forward(self, img, feats):
    ...         return torch.zeros(img.size(0), 2, img.size(2), img.size(3))
    >>> dummy_model = Dummy()
    >>> class DummyLoader:
    ...     def __iter__(self):
    ...         yield torch.zeros(1,3,4,4), [torch.zeros(1,1,1,1) for _ in range(4)], torch.zeros(1,4,4).long()
    ...     def __len__(self): return 1
    >>> loss_fn = SegmentationLoss(num_classes=2)
    >>> loss, metrics = evaluate(dummy_model, DummyLoader(), loss_fn, torch.device("cpu"), False, None, 2)
    >>> loss >= 0 and "miou" in metrics
    True
    """

    if loader is None:
        zeros = torch.zeros(num_classes)
        return 0.0, {"per_class_iou": zeros, "per_class_dice": zeros, "miou": 0.0, "mdice": 0.0}
    model.eval()
    total = 0.0
    metrics = SegmentationMetrics(num_classes)
    autocast = torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()
    with torch.no_grad():
        for batch_idx, (img, features, y) in enumerate(loader, 1):
            img = img.to(device)
            y = y.to(device)
            feats = move_features_to_device(features, device)
            with autocast:
                if hasattr(model, "forward_with_aux"):
                    logits, aux_logits = model.forward_with_aux(img, feats)
                else:
                    logits = model(img, feats)
                    aux_logits = None
                target_main = align_labels_to_logits(y, logits)
                target_aux = align_labels_to_logits(y, aux_logits) if aux_logits is not None else None
                loss = loss_fn(logits, target_main, aux_logits=aux_logits, aux_targets=target_aux)
            total += loss.item()
            preds = logits.argmax(dim=1)
            metrics.update(preds.cpu(), target_main.cpu())
            if logger and batch_idx % 10 == 0:
                logger.debug(
                    f"[Val] batch {batch_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"running mIoU={metrics.compute()['miou']:.4f}"
                )
    avg_loss = total / len(loader)
    metric_summary = metrics.compute()
    if logger:
        logger.debug(
            f"Validation summary :: loss={avg_loss:.4f}, "
            f"mIoU={metric_summary['miou']:.4f}, mDice={metric_summary['mdice']:.4f}"
        )
    return avg_loss, metric_summary


class TTATransform:
    def __init__(self, name: str):
        self.name = name

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.name == "hflip":
            return np.flip(image, axis=1).copy()
        if self.name == "vflip":
            return np.flip(image, axis=0).copy()
        return image

    def invert_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.name == "hflip":
            return torch.flip(logits, dims=(3,))
        if self.name == "vflip":
            return torch.flip(logits, dims=(2,))
        return logits


def build_tta_transforms(cfg: dict) -> List[TTATransform]:
    transforms = [TTATransform("none")]
    if cfg.get("horizontal_flip"):
        transforms.append(TTATransform("hflip"))
    if cfg.get("vertical_flip"):
        transforms.append(TTATransform("vflip"))
    return transforms


def train_phase(config: dict, logger: VerbosityLogger) -> None:
    """
    Train the configured segmentation head if enabled.

    High-level:
      - INPUT:  directory of cached tiles (RGB + features + labels)
      - MODEL:  frozen DINO backbone, trainable segmentation head
      - OUTPUT: .pth checkpoints in `weights_dir`
      - PURPOSE: optimize only the segmentation head, using:
                   * mixed loss (CE + Dice)
                   * Muon optimizer for matrix params + AdamW for 1D params
                   * OneCycleLR schedule
                   * EMA and early stopping on mIoU
    """

    # Feature toggle: only run if `train.enable: true`.
    if not section_enabled(config, "train"):
        logger.debug("Train phase disabled.")
        return

    # `section` contains training-specific hyperparameters (batch size,
    # learning rates, num_workers, etc.).
    section = config.get("train", {})

    # Dataset-specific configuration:
    #   - random vs explicit splits
    #   - augmentation settings
    dataset_cfg = config.get("dataset", {})

    # Backbone + head configuration (model name, layers, head type...).
    model_cfg = get_model_config(config)

    # Location of cached tiles; falls back to global path or default if not
    # explicitly set in `train`.
    processed_dir = resolve_path(config, section, "processed_dir", DEFAULT_PROCESSED_DIR)

    # Directory to store model weights (.pth). Creates it if missing.
    weights_dir = section.get("weights_dir", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Compute device (typically `cuda`).
    device = torch.device(section.get("device", DEFAULT_DEVICE))

    # Batch size for training. Can be small if GPU memory is constrained;
    # gradient accumulation is used to simulate larger effective batch size.
    batch_size = section.get("batch_size", 4)

    # Build DataLoaders over cached tiles (train/val). These load precomputed
    # tensors from disk instead of decoding GeoTIFFs at runtime.
    train_loader, train_sampler, val_loader = create_dataloaders(
        processed_dir, dataset_cfg, section, batch_size, logger, dist_ctx
    )
    logger.info(f"Dataset split: {len(train_loader.dataset)} train tiles.")
    if val_loader is not None:
        logger.info(f"Validation tiles: {len(val_loader.dataset)}")

    # Instantiate the segmentation head (UNet, DinoUNet, etc.) according to
    # config. The backbone is not created here because features are cached.
    model = build_head(
        model_cfg["head"],
        num_classes=model_cfg["num_classes"],
        dino_channels=model_cfg["dino_channels"],
    ).to(device)

    # Optional compilation via torch.compile (where available), which can
    # speed up training at the cost of a longer first iteration.
    if section.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    if dist_ctx.enabled:
        model = DDP(
            model,
            device_ids=[dist_ctx.local_rank],
            output_device=dist_ctx.local_rank,
            find_unused_parameters=False,
        )

    # Log number of trainable parameters.
    total_params = sum(p.numel() for p in (model.module if dist_ctx.enabled else model).parameters())
    logger.info(f"Initialized head '{model_cfg['head']}' with {total_params:,} parameters.")

    # Split parameters by dimensionality:
    #   - >=2D tensors (matrices/conv kernels) go to Muon optimizer
    #   - 1D tensors (biases, LayerNorm weights) go to AdamW
    base_model = model.module if dist_ctx.enabled else model
    muon_params, adamw_params = split_params_for_muon(base_model)

    # Construct the hybrid Muon + AdamW optimizer. Muon handles "matrix-like"
    # parameters, AdamW handles 1D params. LR for each group can be tuned
    # separately via config.
    optimizer = Muon(
        muon_params,
        lr=section.get("muon_lr", 0.02),
        momentum=section.get("momentum", 0.95),
        adamw_params=adamw_params,
        adamw_lr=section.get("adamw_lr", 1e-3),
    )

    # Compute how many optimizer steps happen per epoch, accounting for
    # gradient accumulation. OneCycleLR needs this to shape the LR schedule.
    steps_per_epoch = math.ceil(len(train_loader) / max(1, section.get("grad_accum_steps", 1)))

    # OneCycleLR: high initial LR ramp that decays over time, often working
    # well with Muon/AdamW. Max LR reused from muon_lr for simplicity.
    scheduler = OneCycleLR(
        optimizer,
        max_lr=section.get("muon_lr", 0.02),
        epochs=section.get("epochs", 30),
        steps_per_epoch=steps_per_epoch,
    )

    # Loss configuration:
    #   - CE + Dice weighted combination
    #   - optional class_weights, ignore_index, aux loss weight, etc.
    loss_cfg = section.get("loss", {})
    loss_fn = SegmentationLoss(
        num_classes=model_cfg["num_classes"],
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        aux_weight=loss_cfg.get("aux_weight", 0.4),
        class_weights=loss_cfg.get("class_weights"),
        ignore_index=loss_cfg.get("ignore_index"),
    ).to(device)

    # Automatic mixed precision (AMP) on GPU to speed up training and reduce
    # memory; disabled on CPU for simplicity.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    autocast = torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()

    # Path where the best model (by mIoU) will be stored.
    best_path = os.path.join(weights_dir, f"{model_cfg['head']}_best.pth")

    # Early stopping monitors a metric (mIoU) and restores best checkpoint
    # when no improvement beyond `min_delta` is observed for `patience` epochs.
    early_stopping = EarlyStopping(
        patience=section.get("patience", 10),
        min_delta=0.005,
        path=best_path,
        mode="max",
    )

    # Optional EMA: maintain a smoothed copy of the model parameters, which
    # often yields more stable validation performance.
    ema_decay = section.get("ema_decay", 0.0)
    ema = ModelEMA(base_model, ema_decay) if ema_decay > 0 else None

    # Core training hyperparameters from config.
    epochs = section.get("epochs", 30)
    grad_accum = max(1, section.get("grad_accum_steps", 1))

    logger.info(f"Training for up to {epochs} epochs on device {device}.")

    # The entire training loop is profiled with TimedBlock.
    with TimedBlock(logger, "Training phase"):
        global_step = 0
        for epoch in range(epochs):
            # Time each epoch separately for finer-grained profiling.
            with TimedBlock(logger, f"Epoch {epoch + 1}"):
                model.train()
                train_loss = 0.0

                # Zero gradients at the start of each epoch. With gradient
                # accumulation, we call optimizer.step() less frequently.
                optimizer.zero_grad()

                # tqdm progress bar purely for user feedback during training.
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

                for batch_idx, (img, features, y) in enumerate(pbar, 1):
                    # Move batched tiles + labels to the target device.
                    img = img.to(device)
                    y = y.to(device)
                    feats = move_features_to_device(features, device)

                    # Use autocast for mixed precision when on GPU.
                    with autocast:
                        # Some heads expose `forward_with_aux` for deep
                        # supervision. If available, we use both main and aux.
                        if hasattr(model, "forward_with_aux"):
                            logits, aux_logits = model.forward_with_aux(img, feats)
                        else:
                            logits = model(img, feats)
                            aux_logits = None

                        # Align label resolution to current logits (handles
                        # heads that produce logits at lower resolution).
                        target_main = align_labels_to_logits(y, logits)
                        target_aux = (
                            align_labels_to_logits(y, aux_logits) if aux_logits is not None else None
                        )

                        # Compute composite loss (main + optional aux).
                        # Divide by grad_accum so that effective gradient after
                        # accumulating `grad_accum` steps matches the full batch.
                        loss = loss_fn(logits, target_main, aux_logits=aux_logits, aux_targets=target_aux)
                        loss = loss / grad_accum

                    # Backprop: route through GradScaler if AMP is active.
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Perform an optimizer step every `grad_accum` batches
                    # or at the very end of the epoch.
                    if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        # Reset gradients for the next accumulation window.
                        optimizer.zero_grad()

                        # Advance LR schedule (OneCycle).
                        scheduler.step()

                        # Update EMA weights if enabled.
                        if ema:
                            ema.update(model.module if dist_ctx.enabled else model)

                        global_step += 1

                    # Accumulate un-normalized loss for reporting.
                    train_loss += loss.item() * grad_accum

                    # Optional debug logging every N batches to monitor loss
                    # and current learning rate during training.
                    if batch_idx % 10 == 0:
                        logger.debug(
                            f"Epoch {epoch + 1}, batch {batch_idx}/{len(train_loader)} "
                            f"loss={loss.item() * grad_accum:.4f}, lr={scheduler.get_last_lr()[0]:.5f}"
                        )

                # Average training loss over all batches.
                avg_train_loss = train_loss / len(train_loader)

                # Use EMA model for evaluation if present; otherwise use
                # the raw model. This usually yields smoother validation metrics.
                eval_model = ema.ema_model if ema else (model.module if dist_ctx.enabled else model)

                # Run full validation pass on cached tiles.
                val_loss, val_metrics = evaluate(
                    eval_model,
                    val_loader,
                    loss_fn,
                    device,
                    use_amp,
                    logger if dist_ctx.is_main else None,
                    model_cfg["num_classes"],
                )

                if dist_ctx.enabled:
                    loss_tensor = torch.tensor([val_loss, val_metrics["miou"]], device=device)
                    dist.broadcast(loss_tensor, src=0)
                    val_loss = loss_tensor[0].item()
                    val_metrics["miou"] = loss_tensor[1].item()

                logger.info(
                    f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['miou']:.4f}"
                )

                # Save a per-epoch checkpoint with val loss and mIoU encoded
                # in the filename for easy manual inspection/comparison.
                epoch_ckpt = os.path.join(
                    weights_dir,
                    f"{model_cfg['head']}_VALLOSS_{val_loss:.4f}_MIOU_{val_metrics['miou']:.4f}_EPOCH_{epoch + 1}.pth",
                )
                if dist_ctx.is_main:
                    torch.save(eval_model.state_dict(), epoch_ckpt)

                # Update early stopping using the monitored metric (mIoU).
                # It will also keep track of and write the best model to `best_path`.
                stop_flag = False
                if dist_ctx.is_main:
                    early_stopping(val_metrics["miou"], eval_model)
                    stop_flag = early_stopping.early_stop
                if dist_ctx.enabled:
                    flag_tensor = torch.tensor(1 if stop_flag else 0, device=device)
                    dist.broadcast(flag_tensor, src=0)
                    stop_flag = bool(flag_tensor.item())
                if stop_flag:
                    if dist_ctx.is_main:
                        logger.info("Early stopping triggered.")
                    break

    if dist_ctx.is_main:
        logger.info(f"Training finished. Best weights saved to {best_path}")


def inference_phase(config: dict, logger: VerbosityLogger) -> None:
    """
    Run sliding-window inference if enabled.

    High-level:
      - INPUT:   large input GeoTIFF (`input_tif`)
      - MODEL:   DINO backbone + trained head (loaded from `checkpoint`)
      - PROCESS: sliding-window tiling with optional TTA + probability fusion
      - OUTPUT:  single-band prediction raster (`output_tif`)
      - PURPOSE: scalable inference on large scenes that do not fit into
                 GPU memory at once, while leveraging cached DINO features
                 per tile (computed on the fly here).
    """

    # Support both `inference` and legacy `infer` config keys. If neither is
    # enabled, we exit early.
    infer_cfg = config.get("inference", config.get("infer", {}))
    if not infer_cfg or not infer_cfg.get("enable", False):
        logger.debug("Inference phase disabled.")
        return

    # Model configuration (backbone name, feature layers, head type, etc.).
    model_cfg = get_model_config(config)

    # Device for inference; often `cuda`, but can be overridden in config.
    device = torch.device(infer_cfg.get("device", DEFAULT_DEVICE))

    # Hugging Face processor + backbone for DINO. We only need the encoder
    # (AutoModel), not the classification head.
    processor = AutoImageProcessor.from_pretrained(model_cfg["backbone"])
    backbone = AutoModel.from_pretrained(model_cfg["backbone"]).eval().to(device)

    # Build the segmentation head with the same interface used in training.
    head = build_head(
        model_cfg["head"],
        num_classes=model_cfg["num_classes"],
        dino_channels=model_cfg["dino_channels"],
    ).to(device)

    # Load the trained weights; `checkpoint` is typically the best model
    # selected by early stopping, but can be any .pth path.
    checkpoint = infer_cfg["checkpoint"]
    logger.info(f"Loading checkpoint {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=device)
    # `strict=False` allows missing/extra keys (e.g. EMA buffers, different
    # deep supervision heads) without crashing.
    head.load_state_dict(state_dict, strict=False)
    head.eval()

    # Input and output rasters:
    #   - input_tif: large RGB image
    #   - output_tif: predictions (1 band: class index per pixel)
    input_tif = infer_cfg["input_tif"]
    output_tif = infer_cfg["output_tif"]

    # Tile size for sliding window. Only the tile must fit in GPU memory,
    # not the whole image.
    tile_size = infer_cfg.get("tile_size", 512)

    # Patch size of the ViT backbone; affects feature extraction grid.
    # For sat-493m, `vitl16` usually implies 16, `vitl14` implies 14.
    ps = 14 if "vitl14" in model_cfg["backbone"] else 16

    # Overlap configuration:
    #   - if < 1, interpret as fraction of tile_size
    #   - if >=1, interpret as absolute pixel overlap
    overlap_cfg = infer_cfg.get("overlap", 0.0)
    overlap_px = int(tile_size * overlap_cfg) if overlap_cfg < 1 else int(overlap_cfg)

    # Stride between tile origins. Smaller stride → more overlap →
    # smoother boundaries but higher compute.
    stride = max(1, tile_size - overlap_px)

    # Build test-time augmentation (TTA) transforms (hflip/vflip).
    # TTA averages predictions over augmented versions to reduce variance.
    tta_transforms = build_tta_transforms(infer_cfg.get("tta", {}))

    # Read image metadata once to allocate full-size accumulators.
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        channels = src.count

    # For now, the pipeline assumes 3-band imagery.
    assert channels == 3, "Expected 3-band imagery."

    # Pre-allocate probability and count accumulators over full scene:
    #   - prob_accum[c, y, x] stores summed probabilities for each class.
    #   - count_accum[y, x]   counts how many tiles have covered each pixel.
    prob_accum = np.zeros((model_cfg["num_classes"], height, width), dtype=np.float32)
    count_accum = np.zeros((height, width), dtype=np.float32)

    # Purely for logging progress.
    total_tiles = math.ceil(height / stride) * math.ceil(width / stride)
    logger.info(f"Running inference on {total_tiles} tiles with stride {stride}.")

    # Main tiling loop wrapped in TimedBlock for profiling.
    with rasterio.open(input_tif) as src, TimedBlock(logger, "Inference phase"):
        tile_counter = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile_counter += 1

                # Compute the tile window in [y, x] space, clamped to image bounds.
                y_max = min(y + tile_size, height)
                x_max = min(x + tile_size, width)
                window = Window.from_slices((y, y_max), (x, x_max))

                # Read current tile (C, H_tile, W_tile) and convert to HWC.
                img_tile = src.read(window=window, boundless=True)
                img_tile = np.transpose(img_tile, (1, 2, 0))

                # Skip entirely empty tiles (all zeros) to save compute.
                if np.max(img_tile) == 0:
                    continue

                orig_h, orig_w = img_tile.shape[:2]
                pad_h = max(0, tile_size - orig_h)
                pad_w = max(0, tile_size - orig_w)
                if pad_h or pad_w:
                    img_tile = np.pad(
                        img_tile,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect",
                    )

                # Accumulator for per-tile class probabilities. Averaged over TTA.
                tile_probs = np.zeros(
                    (model_cfg["num_classes"], orig_h, orig_w), dtype=np.float32
                )

                # Loop over TTA transforms (none/hflip/vflip).
                for transform in tta_transforms:
                    # Apply augmentation on numpy array (H, W, C).
                    aug_img = transform.apply(img_tile)

                    # Simple normalization: scale uint8 [0, 255] → float32 [0, 1].
                    img_tile_norm = (aug_img.astype(np.float32) / 255.0).astype(np.float32)

                    # Convert to PyTorch tensor (B, C, H, W).
                    img_t = torch.from_numpy(img_tile_norm).permute(2, 0, 1).unsqueeze(0).to(device)

                    # Extract multiscale DINO features for this augmented tile.
                    feats = extract_multiscale_features(
                        aug_img.astype(np.float32),
                        backbone,
                        processor,
                        device,
                        model_cfg["layers"],
                        ps=ps,
                    )
                    # Add batch dimension to each feature map and move to device.
                    feats_batched = [f.to(device).unsqueeze(0) for f in feats]

                    # Forward pass through the head under AMP, invert the TTA
                    # augmentation on logits, and resize logits back to tile size
                    # if needed.
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                        logits = head(img_t, feats_batched)
                        logits = transform.invert_logits(logits)

                        # Some heads may output logits at slightly different
                        # spatial resolution; interpolate back to tile size.
                        if logits.shape[-2:] != img_t.shape[-2:]:
                            logits = F.interpolate(
                                logits,
                                size=img_t.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                        # Convert logits to probabilities via softmax and to numpy.
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    # Crop to original (unpadded) tile size before accumulating.
                    probs = probs[:, :orig_h, :orig_w]
                    tile_probs += probs

                # Average over TTA samples.
                tile_probs /= len(tta_transforms)

                # Accumulate into global buffers and update coverage counts.
                prob_accum[:, y:y_max, x:x_max] += tile_probs
                count_accum[y:y_max, x:x_max] += 1

                # Periodic progress logging so long runs are visible.
                if tile_counter % 50 == 0 or tile_counter == total_tiles:
                    logger.info(f"Inference progress: {tile_counter}/{total_tiles} tiles.")

    # Avoid division by zero in uncovered pixels (should be rare if tiling is correct).
    count_accum[count_accum == 0] = 1

    # Normalize aggregated probabilities by the number of times each pixel
    # has been seen (handles overlaps).
    prob_accum /= count_accum

    # Final hard prediction: argmax over class dimension.
    pred_full = prob_accum.argmax(axis=0).astype(np.uint8)

    # Update output profile to single-band uint8 with nodata=0.
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)

    # Write prediction raster to disk.
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(pred_full, 1)

    logger.info(f"Saved prediction to {output_tif}")


def main(config_path: str | None = None) -> None:
    """
    Load a YAML configuration file and execute the enabled phases.

    Command-line usage pattern:
      python script.py path/to/config.yml

    Execution order (each phase can be individually enabled/disabled):
      1. prepare_phase  → build cached tiles and DINO features
      2. verify_phase   → sanity-check cached tiles
      3. train_phase    → train segmentation head on cached tiles
      4. inference_phase→ run sliding-window inference on large rasters
    """

    # Resolve config path:
    #   - explicit function arg (config_path), or
    #   - first CLI argument (sys.argv[1]), or
    #   - embedded default in `load_config` if none provided.
    candidate = config_path or (sys.argv[1] if len(sys.argv) > 1 else None)

    # Load configuration dict from YAML or similar. The loader also stashes
    # the resolved config path in `_config_path` for logging.
    config = load_config(candidate)

    # Apply resource settings (threads, seeds, matmul precision).
    apply_resource_config(config)
    dist_ctx = setup_distributed(config.get("resources", {}))

    # Build logger with requested verbosity, timestamps, and optional logfile.
    logger = build_logger(config, enabled=dist_ctx.is_main)

    logger.info(f"Loaded configuration from {config.get('_config_path', 'embedded dict')}")

    # Phase 1: data preparation (tiling + DINO feature cache).
    prepare_phase(config, logger)

    # Phase 2: verification of the cached dataset.
    verify_phase(config, logger)

    # Phase 3: training of the segmentation head.
    train_phase(config, logger, dist_ctx)

    # Phase 4: sliding-window inference over large rasters.
    if dist_ctx.is_main:
        inference_phase(config, logger)

    logger.info("All enabled phases completed.")
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()

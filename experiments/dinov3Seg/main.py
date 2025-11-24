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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
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

DEFAULT_RAW_IMAGES_DIR = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/"
DEFAULT_LABEL_PATH = "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
DEFAULT_PROCESSED_DIR = "/mnt/OS/processed_tiles_1024/"
DEFAULT_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
DEFAULT_LAYERS = [5, 11, 17, 23]
DEFAULT_HEAD = "unet"
DEFAULT_NUM_CLASSES = 2
DEFAULT_DINO_CHANNELS = 1024
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def build_logger(config: dict) -> VerbosityLogger:
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
    return VerbosityLogger(level=level, timestamps=timestamps, log_file=log_file)


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
    """

    if not section_enabled(config, "prepare"):
        logger.debug("Prepare phase disabled.")
        return
    section = config.get("prepare", {})
    model_cfg = get_model_config(config)
    img_dir = resolve_path(config, section, "img_dir", DEFAULT_RAW_IMAGES_DIR)
    label_path = resolve_path(config, section, "label_path", DEFAULT_LABEL_PATH)
    output_dir = resolve_path(config, section, "output_dir", DEFAULT_PROCESSED_DIR)
    device = torch.device(section.get("device", DEFAULT_DEVICE))
    with TimedBlock(logger, "Preparation phase"):
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
    """

    if not section_enabled(config, "verify"):
        logger.debug("Verify phase disabled.")
        return
    section = config.get("verify", {})
    processed_dir = resolve_path(config, section, "processed_dir", DEFAULT_PROCESSED_DIR)
    with TimedBlock(logger, "Verification phase"):
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
) -> tuple[DataLoader, DataLoader]:
    augment_cfg = dataset_cfg.get("augmentations", {})
    split_cfg = dataset_cfg.get("splits", {})
    val_fraction = train_cfg.get("val_fraction", 0.2)
    train_files, val_files = resolve_dataset_splits(processed_dir, split_cfg, val_fraction, logger)
    train_dataset = PrecomputedDataset(
        processed_dir,
        augmentation_cfg=augment_cfg,
        file_subset=train_files,
    )
    val_dataset = PrecomputedDataset(
        processed_dir,
        augmentation_cfg={"enable": False},
        file_subset=val_files,
    )
    num_workers = train_cfg.get("num_workers", 4)
    val_workers = train_cfg.get("val_workers", max(1, num_workers // 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    return train_loader, val_loader


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
    loader: DataLoader,
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

    model.eval()
    total = 0.0
    metrics = SegmentationMetrics(num_classes)
    autocast = torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()
    with torch.no_grad():
        for img, features, y in loader:
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
    avg_loss = total / len(loader)
    metric_summary = metrics.compute()
    if logger:
        logger.debug(f"Validation loss: {avg_loss:.4f} | mIoU: {metric_summary['miou']:.4f}")
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
    """

    if not section_enabled(config, "train"):
        logger.debug("Train phase disabled.")
        return
    section = config.get("train", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = get_model_config(config)
    processed_dir = resolve_path(config, section, "processed_dir", DEFAULT_PROCESSED_DIR)
    weights_dir = section.get("weights_dir", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    device = torch.device(section.get("device", DEFAULT_DEVICE))
    batch_size = section.get("batch_size", 4)
    train_loader, val_loader = create_dataloaders(processed_dir, dataset_cfg, section, batch_size, logger)
    logger.info(
        f"Dataset split: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val tiles."
    )
    model = build_head(
        model_cfg["head"],
        num_classes=model_cfg["num_classes"],
        dino_channels=model_cfg["dino_channels"],
    ).to(device)
    if section.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialized head '{model_cfg['head']}' with {total_params:,} parameters.")
    muon_params, adamw_params = split_params_for_muon(model)
    optimizer = Muon(
        muon_params,
        lr=section.get("muon_lr", 0.02),
        momentum=section.get("momentum", 0.95),
        adamw_params=adamw_params,
        adamw_lr=section.get("adamw_lr", 1e-3),
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, section.get("grad_accum_steps", 1)))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=section.get("muon_lr", 0.02),
        epochs=section.get("epochs", 30),
        steps_per_epoch=steps_per_epoch,
    )
    loss_cfg = section.get("loss", {})
    loss_fn = SegmentationLoss(
        num_classes=model_cfg["num_classes"],
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        aux_weight=loss_cfg.get("aux_weight", 0.4),
        class_weights=loss_cfg.get("class_weights"),
        ignore_index=loss_cfg.get("ignore_index"),
    ).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    autocast = torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()
    best_path = os.path.join(weights_dir, f"{model_cfg['head']}_best.pth")
    early_stopping = EarlyStopping(
        patience=section.get("patience", 10),
        min_delta=0.005,
        path=best_path,
        mode="max",
    )
    ema_decay = section.get("ema_decay", 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    epochs = section.get("epochs", 30)
    grad_accum = max(1, section.get("grad_accum_steps", 1))
    logger.info(f"Training for up to {epochs} epochs on device {device}.")
    with TimedBlock(logger, "Training phase"):
        global_step = 0
        for epoch in range(epochs):
            with TimedBlock(logger, f"Epoch {epoch + 1}"):
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
                for batch_idx, (img, features, y) in enumerate(pbar, 1):
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
                        loss = loss / grad_accum
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        if ema:
                            ema.update(model)
                        global_step += 1
                    train_loss += loss.item() * grad_accum
                    if batch_idx % 10 == 0:
                        logger.debug(
                            f"Epoch {epoch + 1}, batch {batch_idx}/{len(train_loader)} "
                            f"loss={loss.item() * grad_accum:.4f}, lr={scheduler.get_last_lr()[0]:.5f}"
                        )
                avg_train_loss = train_loss / len(train_loader)
                eval_model = ema.ema_model if ema else model
                val_loss, val_metrics = evaluate(
                    eval_model,
                    val_loader,
                    loss_fn,
                    device,
                    use_amp,
                    logger,
                    model_cfg["num_classes"],
                )
                logger.info(
                    f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['miou']:.4f}"
                )
                epoch_ckpt = os.path.join(
                    weights_dir,
                    f"{model_cfg['head']}_VALLOSS_{val_loss:.4f}_MIOU_{val_metrics['miou']:.4f}_EPOCH_{epoch + 1}.pth",
                )
                torch.save(eval_model.state_dict(), epoch_ckpt)
                early_stopping(val_metrics["miou"], eval_model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break
    logger.info(f"Training finished. Best weights saved to {best_path}")


def inference_phase(config: dict, logger: VerbosityLogger) -> None:
    """
    Run sliding-window inference if enabled.
    """

    infer_cfg = config.get("inference", config.get("infer", {}))
    if not infer_cfg or not infer_cfg.get("enable", False):
        logger.debug("Inference phase disabled.")
        return
    model_cfg = get_model_config(config)
    device = torch.device(infer_cfg.get("device", DEFAULT_DEVICE))
    processor = AutoImageProcessor.from_pretrained(model_cfg["backbone"])
    backbone = AutoModel.from_pretrained(model_cfg["backbone"]).eval().to(device)
    head = build_head(model_cfg["head"], num_classes=model_cfg["num_classes"], dino_channels=model_cfg["dino_channels"]).to(device)
    checkpoint = infer_cfg["checkpoint"]
    logger.info(f"Loading checkpoint {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=device)
    head.load_state_dict(state_dict, strict=False)
    head.eval()
    input_tif = infer_cfg["input_tif"]
    output_tif = infer_cfg["output_tif"]
    tile_size = infer_cfg.get("tile_size", 512)
    ps = 14 if "vitl14" in model_cfg["backbone"] else 16
    overlap_cfg = infer_cfg.get("overlap", 0.0)
    overlap_px = int(tile_size * overlap_cfg) if overlap_cfg < 1 else int(overlap_cfg)
    stride = max(1, tile_size - overlap_px)
    tta_transforms = build_tta_transforms(infer_cfg.get("tta", {}))
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        channels = src.count
    assert channels == 3, "Expected 3-band imagery."
    prob_accum = np.zeros((model_cfg["num_classes"], height, width), dtype=np.float32)
    count_accum = np.zeros((height, width), dtype=np.float32)
    total_tiles = math.ceil(height / stride) * math.ceil(width / stride)
    logger.info(f"Running inference on {total_tiles} tiles with stride {stride}.")
    with rasterio.open(input_tif) as src, TimedBlock(logger, "Inference phase"):
        tile_counter = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile_counter += 1
                y_max = min(y + tile_size, height)
                x_max = min(x + tile_size, width)
                window = Window.from_slices((y, y_max), (x, x_max))
                img_tile = src.read(window=window, boundless=True)
                img_tile = np.transpose(img_tile, (1, 2, 0))
                if np.max(img_tile) == 0:
                    continue
                tile_probs = np.zeros((model_cfg["num_classes"], y_max - y, x_max - x), dtype=np.float32)
                for transform in tta_transforms:
                    aug_img = transform.apply(img_tile)
                    img_tile_norm = (aug_img.astype(np.float32) / 255.0).astype(np.float32)
                    img_t = torch.from_numpy(img_tile_norm).permute(2, 0, 1).unsqueeze(0).to(device)
                    feats = extract_multiscale_features(
                        aug_img.astype(np.float32),
                        backbone,
                        processor,
                        device,
                        model_cfg["layers"],
                        ps=ps,
                    )
                    feats_batched = [f.to(device).unsqueeze(0) for f in feats]
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                        logits = head(img_t, feats_batched)
                        logits = transform.invert_logits(logits)
                        if logits.shape[-2:] != img_t.shape[-2:]:
                            logits = F.interpolate(logits, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    tile_probs += probs[:, : y_max - y, : x_max - x]
                tile_probs /= len(tta_transforms)
                prob_accum[:, y:y_max, x:x_max] += tile_probs
                count_accum[y:y_max, x:x_max] += 1
                if tile_counter % 50 == 0 or tile_counter == total_tiles:
                    logger.info(f"Inference progress: {tile_counter}/{total_tiles} tiles.")
    count_accum[count_accum == 0] = 1
    prob_accum /= count_accum
    pred_full = prob_accum.argmax(axis=0).astype(np.uint8)
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(pred_full, 1)
    logger.info(f"Saved prediction to {output_tif}")


def main(config_path: str | None = None) -> None:
    """
    Load a YAML configuration file and execute the enabled phases.

    >>> main("config.example.yml")  # doctest: +SKIP
    """

    candidate = config_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    config = load_config(candidate)
    apply_resource_config(config)
    logger = build_logger(config)
    logger.info(f"Loaded configuration from {config.get('_config_path', 'embedded dict')}")
    prepare_phase(config, logger)
    verify_phase(config, logger)
    train_phase(config, logger)
    inference_phase(config, logger)
    logger.info("All enabled phases completed.")


if __name__ == "__main__":
    main()

"""
Unified training and inference entry point with pluggable segmentation heads.
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import rasterio
from transformers import AutoImageProcessor, AutoModel

from models import build_head, available_heads
from utils import (
    EarlyStopping,
    Muon,
    PrecomputedDataset,
    extract_multiscale_features,
    prepare_data_tiles,
    verify_and_clean_dataset_fast,
)


# System-wide knobs to stop CPU hangs and fragmentation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser with sub-commands for prep, verify, train, and infer.

    >>> parser = build_arg_parser()
    >>> isinstance(parser, argparse.ArgumentParser)
    True
    """

    parser = argparse.ArgumentParser(description="DINOv3 segmentation toolbox.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Tile GeoTIFFs and cache features.")
    prep.add_argument("--img-dir", required=True)
    prep.add_argument("--label-path", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--model-name", default="facebook/dinov3-vitl16-pretrain-sat493m")
    prep.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    prep.add_argument("--tile-size", type=int, default=512)
    prep.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    verify = subparsers.add_parser("verify", help="Check cached feature files.")
    verify.add_argument("--processed-dir", required=True)
    verify.add_argument("--workers", type=int, default=None)

    train = subparsers.add_parser("train", help="Train a segmentation head.")
    train.add_argument("--processed-dir", required=True)
    train.add_argument("--weights-dir", required=True)
    train.add_argument("--head", choices=list(available_heads().keys()), default="unet")
    train.add_argument("--num-classes", type=int, default=2)
    train.add_argument("--dino-channels", type=int, default=1024)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--muon-lr", type=float, default=0.02)
    train.add_argument("--adamw-lr", type=float, default=1e-4)
    train.add_argument("--momentum", type=float, default=0.95)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--patience", type=int, default=10)

    infer = subparsers.add_parser("infer", help="Run sliding-window inference.")
    infer.add_argument("--input-tif", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output-tif", required=True)
    infer.add_argument("--head", choices=list(available_heads().keys()), default="unet")
    infer.add_argument("--num-classes", type=int, default=2)
    infer.add_argument("--dino-channels", type=int, default=1024)
    infer.add_argument("--model-name", default="facebook/dinov3-vitl16-pretrain-sat493m")
    infer.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    infer.add_argument("--tile-size", type=int, default=512)
    infer.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the program.

    >>> args = parse_args(["verify", "--processed-dir", "/tmp"])
    >>> args.command
    'verify'
    """

    parser = build_arg_parser()
    return parser.parse_args(argv)


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


def build_dataloaders(dataset: Dataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Split the dataset 80/20 and build PyTorch data loaders.

    >>> class Dummy(Dataset):
    ...     def __len__(self): return 10
    ...     def __getitem__(self, idx): return torch.zeros(3, 4, 4), [torch.zeros(1,1,1) for _ in range(4)], torch.zeros(4,4).long()
    >>> ds = Dummy()
    >>> train_loader, val_loader = build_dataloaders(ds, batch_size=2)
    >>> len(train_loader) > 0 and len(val_loader) > 0
    True
    """

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_losses(
    logits: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    aux_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the main loss and optionally add auxiliary supervision.

    >>> crit = torch.nn.CrossEntropyLoss()
    >>> logits = torch.randn(2, 2, 4, 4)
    >>> labels = torch.zeros(2, 4, 4).long()
    >>> compute_losses(logits, labels, crit).shape
    torch.Size([])
    """

    y_aligned = align_labels_to_logits(y, logits)
    main_loss = criterion(logits, y_aligned)
    if aux_logits is None:
        return main_loss
    aux_target = align_labels_to_logits(y, aux_logits)
    return main_loss + 0.4 * criterion(aux_logits, aux_target)


def prepare_tiles_command(args: argparse.Namespace) -> None:
    """
    Entry point for the prepare sub-command.

    >>> ns = argparse.Namespace(
    ...     img_dir="/tmp",
    ...     label_path="/tmp/labels.tif",
    ...     output_dir="/tmp/out",
    ...     model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    ...     layers=[5, 11, 17, 23],
    ...     tile_size=512,
    ...     device="cpu",
    ... )
    >>> prepare_tiles_command(ns)  # doctest: +SKIP
    """

    device = torch.device(args.device)
    prepare_data_tiles(
        img_dir=args.img_dir,
        label_path=args.label_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        layers=args.layers,
        device=device,
        tile_size=args.tile_size,
    )


def verify_command(args: argparse.Namespace) -> None:
    """
    Entry point for verifying cached dataset files.

    >>> ns = argparse.Namespace(processed_dir="/tmp", workers=1)
    >>> verify_command(ns)  # doctest: +SKIP
    """

    verify_and_clean_dataset_fast(args.processed_dir, args.workers)


def train_command(args: argparse.Namespace) -> None:
    """
    Train the requested segmentation head.

    >>> ns = argparse.Namespace(
    ...     processed_dir="/tmp",
    ...     weights_dir="/tmp",
    ...     head="unet",
    ...     num_classes=2,
    ...     dino_channels=1024,
    ...     batch_size=1,
    ...     epochs=1,
    ...     muon_lr=0.02,
    ...     adamw_lr=1e-3,
    ...     momentum=0.95,
    ...     device="cpu",
    ...     patience=1,
    ... )
    >>> train_command(ns)  # doctest: +SKIP
    """

    device = torch.device(args.device)
    os.makedirs(args.weights_dir, exist_ok=True)
    dataset = PrecomputedDataset(args.processed_dir)
    train_loader, val_loader = build_dataloaders(dataset, args.batch_size)
    model = build_head(args.head, num_classes=args.num_classes, dino_channels=args.dino_channels).to(device)
    muon_params, adamw_params = split_params_for_muon(model)
    optimizer = Muon(
        muon_params,
        lr=args.muon_lr,
        momentum=args.momentum,
        adamw_params=adamw_params,
        adamw_lr=args.adamw_lr,
    )
    scheduler = OneCycleLR(optimizer, max_lr=args.muon_lr, total_steps=len(train_loader) * args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    autocast = torch.amp.autocast(device_type=device.type) if use_amp else nullcontext()
    ckpt_path = os.path.join(args.weights_dir, f"{args.head}_best.pth")
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.005, path=ckpt_path)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for img, features, y in pbar:
            img = img.to(device)
            y = y.to(device)
            feats = move_features_to_device(features, device)
            optimizer.zero_grad()
            with autocast:
                if hasattr(model, "forward_with_aux"):
                    logits, aux_logits = model.forward_with_aux(img, feats)
                else:
                    logits = model(img, feats)
                    aux_logits = None
                loss = compute_losses(logits, y, criterion, aux_logits)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        avg_train_loss = train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    print(f"Training finished. Best weights saved to {ckpt_path}")


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
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
    >>> loss = evaluate(dummy_model, DummyLoader(), torch.nn.CrossEntropyLoss(), torch.device("cpu"), False)
    >>> loss >= 0
    True
    """

    model.eval()
    total = 0.0
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
                loss = compute_losses(logits, y, criterion, aux_logits)
            total += loss.item()
    return total / len(loader)


def infer_command(args: argparse.Namespace) -> None:
    """
    Sliding-window inference command entry point.

    >>> ns = argparse.Namespace(
    ...     input_tif="/tmp/input.tif",
    ...     checkpoint="/tmp/ckpt.pth",
    ...     output_tif="/tmp/out.tif",
    ...     head="unet",
    ...     num_classes=2,
    ...     dino_channels=1024,
    ...     model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    ...     layers=[5, 11, 17, 23],
    ...     tile_size=512,
    ...     device="cpu",
    ... )
    >>> infer_command(ns)  # doctest: +SKIP
    """

    device = torch.device(args.device)
    head = build_head(args.head, num_classes=args.num_classes, dino_channels=args.dino_channels).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(state_dict, strict=False)
    head.eval()

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name).eval().to(device)

    with rasterio.open(args.input_tif) as src:
        profile = src.profile.copy()
        img_full = src.read()
        img_full = np.transpose(img_full, (1, 2, 0))
        height, width, channels = img_full.shape
    assert channels == 3, "Expected 3-band imagery."
    pred_full = np.zeros((height, width), dtype=np.uint8)
    ps = 14 if "vitl14" in args.model_name else 16
    for y in tqdm(range(0, height, args.tile_size), desc="Inferring"):
        for x in range(0, width, args.tile_size):
            y_min, x_min = y, x
            y_max, x_max = y + args.tile_size, x + args.tile_size
            if y_max > height:
                y_min, y_max = height - args.tile_size, height
            if x_max > width:
                x_min, x_max = width - args.tile_size, width
            img_tile = img_full[y_min:y_max, x_min:x_max, :]
            if np.max(img_tile) == 0:
                continue
            img_tile_norm = (img_tile.astype(np.float32) / 255.0).astype(np.float32)
            img_t = torch.from_numpy(img_tile_norm).permute(2, 0, 1).unsqueeze(0).to(device)
            feats = extract_multiscale_features(
                img_tile.astype(np.float32),
                backbone,
                processor,
                device,
                args.layers,
                ps=ps,
            )
            feats_batched = [f.to(device).unsqueeze(0) for f in feats]
            with torch.no_grad():
                logits = head(img_t, feats_batched)
                if logits.shape[-2:] != img_t.shape[-2:]:
                    logits = F.interpolate(logits, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
                pred_tile = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_full[y_min:y_max, x_min:x_max] = pred_tile
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    os.makedirs(os.path.dirname(args.output_tif) or ".", exist_ok=True)
    with rasterio.open(args.output_tif, "w", **profile) as dst:
        dst.write(pred_full, 1)
    print(f"Saved prediction to {args.output_tif}")


def main(argv: List[str] | None = None) -> None:
    """
    Program entry point: dispatch sub-command.

    >>> main(["verify", "--processed-dir", "/tmp"])  # doctest: +SKIP
    """

    args = parse_args(argv)
    if args.command == "prepare":
        prepare_tiles_command(args)
    elif args.command == "verify":
        verify_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

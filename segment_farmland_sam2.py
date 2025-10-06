"""Segment farmland fields in a satellite GeoTIFF using SAM2 Tiny via Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi
from transformers import Sam2Model, Sam2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("images/1084-1389.tif"),
        help="Path to the multispectral GeoTIFF to segment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the mask and overlay will be written.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="2,4,6",
        help="Comma separated 1-based band indices to build the RGB composite.",
    )
    parser.add_argument(
        "--positive-points",
        type=int,
        default=20,
        help="Number of high vegetation points sampled as positive prompts for SAM2.",
    )
    parser.add_argument(
        "--negative-points",
        type=int,
        default=80,
        help="Number of low vegetation points sampled as negative prompts for SAM2 (fallback).",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=150,
        help="Maximum number of candidate field regions to prompt SAM2 with.",
    )
    parser.add_argument(
        "--veg-percentiles",
        type=str,
        default="90,85,80",
        help="Comma-separated vegetation percentiles used to seed prompts (highest first).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id for SAM2 weights.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5_000,
        help="Minimum pixel area for a connected component to be kept in the mask.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="Alpha blending factor for overlaying the mask (0=no overlay, 1=full solid colour).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to run SAM2 inference on (default auto).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for chunked processing (0 processes the full image at once).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Overlap between adjacent tiles when chunking is enabled.",
    )
    return parser.parse_args()


def load_multispectral(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    array = tifffile.imread(path)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.shape[0] < array.shape[-1]:
        # assume (C, H, W)
        return array.astype(np.float32)
    # assume (H, W, C)
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def stretch_channel(channel: np.ndarray) -> np.ndarray:
    low, high = np.percentile(channel, (2, 98))
    if high <= low:
        denom = channel.max() or 1.0
        return np.clip(channel / denom, 0.0, 1.0)
    stretched = (channel - low) / (high - low)
    return np.clip(stretched, 0.0, 1.0)


def build_rgb_composite(multispectral: np.ndarray, band_indices: list[int]) -> np.ndarray:
    channels = []
    max_index = multispectral.shape[0]
    for idx in band_indices:
        if not 1 <= idx <= max_index:
            raise ValueError(f"Band index {idx} is out of range 1..{max_index}")
        channel = multispectral[idx - 1]
        channels.append(stretch_channel(channel))
    composite = np.stack(channels, axis=0)
    return composite


def compute_prompts_and_boxes(
    vegetation: np.ndarray,
    percentiles: list[float],
    max_objects: int,
    fallback_pos: int,
    fallback_neg: int,
    min_area: int,
) -> tuple[list[list[list[float]]], list[list[list[int]]], list[list[list[float]]] | None]:
    vegetation = np.nan_to_num(vegetation, nan=0.0, posinf=0.0, neginf=0.0)
    height, width = vegetation.shape
    point_prompts: list[list[float]] = []
    label_prompts: list[list[int]] = []
    box_prompts: list[list[float]] = []

    visited = np.zeros_like(vegetation, dtype=bool)

    for percentile in percentiles:
        threshold = np.percentile(vegetation, percentile)
        candidate = vegetation > threshold
        candidate = ndi.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
        candidate &= ~visited
        labeled, num_labels = ndi.label(candidate)
        if num_labels == 0:
            continue
        slices = ndi.find_objects(labeled)
        component_sizes = ndi.sum(candidate, labeled, index=range(1, num_labels + 1))
        component_sizes = np.asarray(component_sizes, dtype=np.float64)
        if component_sizes.ndim == 0:
            component_sizes = component_sizes[None]
        order = np.argsort(component_sizes)[::-1]
        min_prompt_area = max(min_area // 6, 250)
        margin = max(int(min(height, width) * 0.01), 6)

        for idx in order:
            area = float(component_sizes[idx])
            if area < min_prompt_area:
                continue
            sl = slices[idx]
            if sl is None:
                continue
            y_slice, x_slice = sl
            y0, y1 = y_slice.start, y_slice.stop
            x0, x1 = x_slice.start, x_slice.stop
            cy = (y0 + y1 - 1) / 2.0
            cx = (x0 + x1 - 1) / 2.0

            x0_pad = max(0, x0 - margin)
            y0_pad = max(0, y0 - margin)
            x1_pad = min(width, x1 + margin)
            y1_pad = min(height, y1 + margin)

            point_prompts.append([[float(cx), float(cy)]])
            label_prompts.append([1])
            box_prompts.append([float(x0_pad), float(y0_pad), float(x1_pad), float(y1_pad)])

            visited[y0_pad:y1_pad, x0_pad:x1_pad] = True

            if len(point_prompts) >= max_objects:
                break
        if len(point_prompts) >= max_objects:
            break

    if not point_prompts:
        h, w = vegetation.shape
        flat = vegetation.reshape(-1)
        num_pos = min(fallback_pos, flat.size)
        num_neg = min(fallback_neg, flat.size)

        pos_idx = np.argpartition(flat, -num_pos)[-num_pos:]
        neg_idx = np.argpartition(flat, num_neg)[:num_neg]

        pos_coords = np.stack(np.unravel_index(pos_idx, (h, w)), axis=1)
        neg_coords = np.stack(np.unravel_index(neg_idx, (h, w)), axis=1)

        points: list[list[float]] = []
        labels: list[int] = []

        for y, x in pos_coords:
            points.append([float(x), float(y)])
            labels.append(1)
        for y, x in neg_coords:
            points.append([float(x), float(y)])
            labels.append(0)

        return [[points]], [[labels]], None

    return [point_prompts], [label_prompts], [box_prompts]


# Utilities for tiling-based segmentation


def compute_window_slices(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    if tile_size <= 0 or tile_size >= length:
        return [(0, length)]
    step = tile_size - overlap
    if step <= 0:
        step = tile_size
    positions = list(range(0, max(1, length - tile_size), step))
    positions.append(length - tile_size)

    slices: list[tuple[int, int]] = []
    for start in positions:
        if start < 0:
            start = 0
        end = start + tile_size
        if end > length:
            start = length - tile_size
            end = length
        window = (start, end)
        if not slices or slices[-1] != window:
            slices.append(window)
    return slices


def segment_patch(
    rgb_patch: np.ndarray,
    vegetation_patch: np.ndarray,
    processor: Sam2Processor,
    model: Sam2Model,
    device: torch.device,
    veg_percentiles: list[float],
    args: argparse.Namespace,
) -> np.ndarray:
    if rgb_patch.size == 0:
        return np.zeros_like(vegetation_patch, dtype=bool)

    image = Image.fromarray(rgb_patch)
    (
        input_points,
        input_labels,
        input_boxes,
    ) = compute_prompts_and_boxes(
        vegetation_patch,
        veg_percentiles,
        args.max_objects,
        args.positive_points,
        args.negative_points,
        args.min_area,
    )

    processor_kwargs = dict(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    if input_boxes is not None:
        processor_kwargs["input_boxes"] = input_boxes

    inputs = processor(**processor_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    pred_masks = outputs.pred_masks.cpu()
    iou_scores = outputs.iou_scores.cpu()

    post_masks = processor.post_process_masks(pred_masks, inputs["original_sizes"])[0]
    best_mask = select_best_masks(post_masks, iou_scores[0])

    return best_mask.cpu().numpy().astype(bool)


def segment_with_tiles(
    rgb_uint8: np.ndarray,
    vegetation_index: np.ndarray,
    processor: Sam2Processor,
    model: Sam2Model,
    device: torch.device,
    veg_percentiles: list[float],
    args: argparse.Namespace,
) -> np.ndarray:
    height, width, _ = rgb_uint8.shape
    y_slices = compute_window_slices(height, args.tile_size, args.tile_overlap)
    x_slices = compute_window_slices(width, args.tile_size, args.tile_overlap)

    combined = np.zeros((height, width), dtype=bool)

    for y0, y1 in y_slices:
        for x0, x1 in x_slices:
            rgb_patch = rgb_uint8[y0:y1, x0:x1]
            veg_patch = vegetation_index[y0:y1, x0:x1]
            if rgb_patch.size == 0 or veg_patch.size == 0:
                continue
            patch_mask = segment_patch(
                rgb_patch,
                veg_patch,
                processor,
                model,
                device,
                veg_percentiles,
                args,
            )
            combined[y0:y1, x0:x1] |= patch_mask

    return combined


def select_best_masks(post_masks: torch.Tensor, iou_scores: torch.Tensor) -> torch.Tensor:
    """Pick the highest scoring mask per object and collapse them into a single bool tensor."""
    # post_masks: (num_objects, num_masks, H, W), bool
    # iou_scores: (num_objects, num_masks)
    if post_masks.shape[0] == 0:
        raise ValueError("SAM2 did not return any masks.")

    mask_list = []
    for obj_idx in range(post_masks.shape[0]):
        scores = iou_scores[obj_idx]
        best_idx = int(torch.argmax(scores).item())
        mask_list.append(post_masks[obj_idx, best_idx])
    combined = torch.stack(mask_list, dim=0).any(dim=0)
    return combined


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndi.binary_fill_holes(mask)

    labeled, num_labels = ndi.label(mask)
    if num_labels == 0:
        return mask
    component_sizes = ndi.sum(mask, labeled, index=range(1, num_labels + 1))
    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[1:] = component_sizes >= min_area
    cleaned = keep[labeled]
    return cleaned


def save_outputs(rgb_uint8: np.ndarray, mask: np.ndarray, out_dir: Path, stem: str, alpha: float) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_path = out_dir / f"{stem}_farmland_mask.png"
    mask_img.save(mask_path)

    overlay = rgb_uint8.copy()
    colour = np.array([46, 204, 113], dtype=np.float32)  # pleasant green
    overlay_mask = mask.astype(bool)
    if overlay_mask.any():
        blended = (1.0 - alpha) * overlay[overlay_mask].astype(np.float32) + alpha * colour
        overlay[overlay_mask] = np.clip(blended, 0, 255).astype(np.uint8)
    overlay_path = out_dir / f"{stem}_sam2_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    return mask_path, overlay_path


def main() -> None:
    args = parse_args()

    multispectral = load_multispectral(args.input)
    band_indices = [int(idx.strip()) for idx in args.bands.split(",") if idx.strip()]
    if len(band_indices) != 3:
        raise ValueError("Exactly three band indices are required to build an RGB composite.")

    composite = build_rgb_composite(multispectral, band_indices)
    rgb_uint8 = np.transpose((composite * 255.0).astype(np.uint8), (1, 2, 0))

    b, g, r = composite[0], composite[1], composite[2]  # B,G,R
    exg = 2.0 * g - r - b  # ExG (Woebbecke et al.)
    mn, mx = exg.min(), exg.max()
    vegetation_index = ((exg - mn) / (mx - mn + 1e-6)).astype(np.float32)
    vegetation_index = np.nan_to_num(vegetation_index, nan=0.0)

    green_band = multispectral[3]  # 1-based 4 (Green)
    nir_band = multispectral[7]  # 1-based 8 (NIR)
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-6)
    vegetation_index[ndwi > 0.05] = -1.0  # suppress water; try 0.0–0.1

    height, width = vegetation_index.shape

    veg_percentiles = [float(p.strip()) for p in args.veg_percentiles.split(',') if p.strip()]
    veg_percentiles = [p for p in veg_percentiles if 0.0 < p < 100.0]
    if not veg_percentiles:
        raise ValueError("Provide at least one vegetation percentile between 0 and 100.")

    veg_percentiles.sort(reverse=True)

    processor = Sam2Processor.from_pretrained(args.model_id)
    model = Sam2Model.from_pretrained(args.model_id)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = args.device
        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            requested = "cpu"
        device = torch.device(requested)

    model.to(device)
    model.eval()

    tile_size = args.tile_size
    use_tiles = bool(tile_size and tile_size > 0 and tile_size < max(height, width))

    if use_tiles:
        mask_bool = segment_with_tiles(
            rgb_uint8,
            vegetation_index,
            processor,
            model,
            device,
            veg_percentiles,
            args,
        )
    else:
        mask_bool = segment_patch(
            rgb_uint8,
            vegetation_index,
            processor,
            model,
            device,
            veg_percentiles,
            args,
        )

    cleaned_mask = clean_mask(mask_bool.astype(bool), args.min_area)

    mask_path, overlay_path = save_outputs(rgb_uint8, cleaned_mask, args.output_dir, args.input.stem, args.overlay_alpha)

    coverage = cleaned_mask.mean() * 100.0
    print(f"Saved binary mask to {mask_path}")
    print(f"Saved overlay image to {overlay_path}")
    print(f"Farmland coverage: {coverage:.2f}% of pixels")


if __name__ == "__main__":
    main()



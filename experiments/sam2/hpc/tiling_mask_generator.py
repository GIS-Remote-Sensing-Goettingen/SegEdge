import argparse
import gc
import glob
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM2_REPO = PROJECT_ROOT / "third_party" / "sam2"
if SAM2_REPO.exists():
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "sam2" / "models" / "sam2_hiera_large.pt"
DEFAULT_MODEL_CONFIG = SAM2_REPO / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml"
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "samples" / "imagery" / "1084-1393.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "outputs" / "sam2" / "hpc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile-based SAM2 automatic mask generator for ultra-large scenes."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image to segment.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM2 checkpoint path.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="SAM2 model configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory where outputs will be stored.",
    )
    parser.add_argument("--patch-size", type=int, default=2048, help="Patch size for tiling.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between patches.")
    parser.add_argument("--points-per-side", type=int, default=64, help="SAM2 mask generator parameter.")
    parser.add_argument("--points-per-batch", type=int, default=32, help="SAM2 mask generator parameter.")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.5, help="Minimum predicted IoU to keep a mask.")
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.92,
        help="Minimum stability score when accepting masks.",
    )
    parser.add_argument("--stability-offset", type=float, default=0.7, help="Stability offset hyperparameter.")
    parser.add_argument("--box-nms-thresh", type=float, default=0.7, help="Box NMS threshold.")
    parser.add_argument("--min-mask-area", type=int, default=1000, help="Minimum mask region area (pixels).")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="run",
        help="Prefix used when auto-incrementing output folders.",
    )
    return parser.parse_args()

# Color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and tags"""

    # Tag colors
    TAG_COLORS = {
        'VRAM': Colors.CYAN,
        'PATCH': Colors.MAGENTA,
        'SAVE': Colors.GREEN,
        'MODEL': Colors.BLUE,
        'IMAGE': Colors.YELLOW,
        'MEMORY': Colors.BRIGHT_CYAN,
        'TIME': Colors.BRIGHT_GREEN,
    }

    # Level colors
    LEVEL_COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.WHITE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        # Color the level name
        level_color = self.LEVEL_COLORS.get(record.levelname, Colors.WHITE)
        colored_level = f"{level_color}{record.levelname}{Colors.RESET}"

        # Format timestamp
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        colored_timestamp = f"{Colors.BRIGHT_BLACK}{timestamp}{Colors.RESET}"

        # Check if message has a tag
        message = record.getMessage()
        colored_message = message

        # Apply tag colors
        for tag, color in self.TAG_COLORS.items():
            tag_pattern = f"[{tag}]"
            if tag_pattern in message:
                colored_tag = f"{color}{Colors.BOLD}[{tag}]{Colors.RESET}"
                colored_message = colored_message.replace(tag_pattern, colored_tag)

        return f"{colored_timestamp} - {colored_level} - {colored_message}"


class FileFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)"""
    def format(self, record):
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        return f"{timestamp} - {record.levelname} - {record.getMessage()}"


# Setup logging with colored console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter())

# File handler without colors
file_handler = logging.FileHandler('sam2_processing.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(FileFormatter())

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_next_output_folder(base_dir: Path, prefix: str = "output_job") -> Path:
    """
    Find the next available output folder name in ``base_dir``.

    Args:
        base_dir: Directory where run folders are created.
        prefix: Folder prefix (defaults to ``output_job``).

    Returns:
        Path: Next available folder path (e.g. ``.../output_job_3``).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_folders = sorted(base_dir.glob(f"{prefix}_*"))
    if not existing_folders:
        return base_dir / f"{prefix}_0"

    indices = []
    for folder in existing_folders:
        try:
            num = int(folder.name.split("_")[-1])
            indices.append(num)
        except ValueError:
            continue

    next_index = (max(indices) + 1) if indices else 0
    return base_dir / f"{prefix}_{next_index}"


def generate_patches(image, patch_size=1024, overlap=128):
    """
    Split image into overlapping patches.

    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch
        overlap: Overlap between adjacent patches

    Returns:
        List of (patch, x_offset, y_offset) tuples
    """
    h, w = image.shape[:2]

    # Validate minimum size
    if h < overlap or w < overlap:
        raise ValueError(f"Image too small ({h}x{w}) for overlap {overlap}")

    stride = patch_size - overlap
    patches = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            # Extract patch
            patch = image[y:y_end, x:x_end]

            # Pad if needed
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch.copy()
                patch = padded

            patches.append((patch, x, y))

    logger.info(f"[PATCH] Generated {len(patches)} patches of size {patch_size}x{patch_size} with {overlap}px overlap")
    return patches


def process_patches_to_labelmap(patches, mask_generator, image_shape, overlap=128):
    """
    Process patches and build a single label map incrementally.
    This avoids creating full-image arrays for each mask.

    Args:
        patches: List of (patch, x_offset, y_offset) tuples
        mask_generator: SAM2AutomaticMaskGenerator instance
        image_shape: (H, W) of original image
        overlap: Overlap size for overlap region handling

    Returns:
        labels: H×W int32 array where each pixel has a unique mask ID (0=background)
    """
    H, W = image_shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_id = 1

    logger.info(f"[PATCH] Processing {len(patches)} patches into label map")

    for idx, (patch_rgb, ox, oy) in enumerate(patches):
        logger.info(f"[PATCH] Processing patch {idx + 1}/{len(patches)} at offset ({ox}, {oy})")
        t_patch = time.time()

        # Use scoped inference_mode + autocast per patch to prevent fragmentation
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            patch_masks = mask_generator.generate(patch_rgb)

        logger.debug(f"[PATCH] Patch {idx + 1} generated {len(patch_masks)} masks in {time.time() - t_patch:.2f}s")

        # Sort by area (small first) so tiny regions aren't overpainted by large ones
        patch_masks.sort(key=lambda m: m.get("area", 0))

        painted_count = 0
        for m in patch_masks:
            seg = m["segmentation"].astype(bool)  # (ph, pw)
            ph, pw = seg.shape

            # Calculate bounds in global image space
            y0, y1 = oy, min(oy + ph, H)
            x0, x1 = ox, min(ox + pw, W)
            seg_crop = seg[:y1 - y0, :x1 - x0]

            # Paint only where label==0 (no overlap, first-come-first-served)
            # For overlap regions, you could add IoU check here if needed
            paint_region = labels[y0:y1, x0:x1]
            paint_mask = (paint_region == 0) & seg_crop

            if paint_mask.any():
                paint_region[paint_mask] = next_id
                next_id += 1
                painted_count += 1

        logger.debug(f"[PATCH] Painted {painted_count}/{len(patch_masks)} masks from patch {idx + 1}")

        print_vram_usage()

        # Free memory immediately after each patch
        del patch_masks
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Periodic garbage collection
        if (idx + 1) % 5 == 0:
            gc.collect()
            logger.info(f"[MEMORY] Memory cleanup after patch {idx + 1}")

    unique_labels = np.unique(labels)
    logger.info(f"[PATCH] Label map complete: {len(unique_labels) - 1} unique masks (excluding background)")
    return labels


def labelmap_to_colored_mask(labels, image_bgr):
    """
    Convert label map to a colored visualization overlay.

    Args:
        labels: H×W int32 label map
        image_bgr: Original BGR image

    Returns:
        Annotated BGR image with colored masks
    """
    logger.info("[IMAGE] Creating colored visualization from label map")

    # Create a color map
    num_labels = labels.max()
    if num_labels == 0:
        logger.warning("[IMAGE] No masks found in label map")
        return image_bgr.copy()

    # Generate random colors for each label
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Create colored mask
    colored_mask = colors[labels]

    # Blend with original image
    alpha = 0.5
    annotated = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)

    return annotated


def print_vram_usage(device=0):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_allocated(device)
    logger.info(f"[VRAM] allocated: {alloc/1024**2:.2f} MiB, reserved: {reserved/1024**2:.2f} MiB, "
                f"free: {free/1024**2:.2f} MiB, total: {total/1024**2:.2f} MiB, peak: {peak/1024**2:.2f} MiB")


def plot_images_and_save(images, grid_size, titles, save_path, dpi=300):
    """
    Save images in a grid with high quality.

    Args:
        images: List of images to display
        grid_size: (nrows, ncols) tuple
        titles: List of titles for each image
        save_path: Path to save the output
        dpi: DPI for output image (default 300 for high quality)
    """
    logger.debug(f"[SAVE] plot_images_and_save called")
    logger.debug(f"[SAVE] grid_size: {grid_size}, #images: {len(images)}")
    for i, img in enumerate(images):
        logger.debug(f"[SAVE] image {i} shape: {img.shape if hasattr(img, 'shape') else type(img)}")

    t0 = time.time()
    nrows, ncols = grid_size

    # Calculate figure size based on actual image dimensions for better quality
    if len(images) > 0 and hasattr(images[0], 'shape'):
        img_h, img_w = images[0].shape[:2]
        # Calculate figure size to maintain aspect ratio
        # Use inches: divide by dpi to get proper size
        fig_w = (img_w * ncols) / dpi
        fig_h = (img_h * nrows) / dpi
        figsize = (fig_w, fig_h)
    else:
        figsize = (8 * ncols, 8 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]
            # assume BGR images
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"[SAVE] cvtColor failed for image {idx}: {e}")
                rgb = img
            ax.imshow(rgb, interpolation='bilinear')
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    dt = time.time() - t0
    logger.info(f"[SAVE] Saved grid image to {save_path} in {dt:.3f}s at {dpi} DPI")


def save_individual_images(image_bgr, annotated_image, labels, output_dir):
    """
    Save individual high-quality images without matplotlib compression.

    Args:
        image_bgr: Original BGR image
        annotated_image: Annotated BGR image with masks
        labels: Label map array
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[SAVE] Saving individual high-quality images to {output_dir}/")

    # Save original image
    cv2.imwrite(f"{output_dir}/original.png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save annotated image
    cv2.imwrite(f"{output_dir}/annotated.png", annotated_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save label map as 16-bit TIFF
    cv2.imwrite(f"{output_dir}/labels.tif", labels.astype(np.uint16))

    logger.info(f"[SAVE] Saved:")
    logger.info(f"[SAVE]   - {output_dir}/original.png")
    logger.info(f"[SAVE]   - {output_dir}/annotated.png")
    logger.info(f"[SAVE]   - {output_dir}/labels.tif")


if __name__ == '__main__':
    cli_args = parse_args()
    logger.info("=" * 80)
    logger.info(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}SAM2 Segmentation Pipeline Started{Colors.RESET}")
    logger.info("=" * 80)
    t_start = time.time()

    # GPU setup - but DO NOT enter global autocast context
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"[MODEL] GPU device properties: {props}")
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[MODEL] TF32 enabled for faster computation")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[MODEL] Device: {DEVICE}")
    print_vram_usage()

    checkpoint_path = Path(cli_args.checkpoint).expanduser().resolve()
    config_path = Path(cli_args.model_config).expanduser().resolve()
    image_path = Path(cli_args.image).expanduser().resolve()
    output_root = Path(cli_args.output_dir).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("[MODEL] Building SAM2 model")

    from pathlib import Path
    from hydra import compose, initialize_config_dir


    def build_sam2(config_name_or_path: str, checkpoint: str, device="cuda", apply_postprocessing=False):
        p = Path(config_name_or_path)
        if p.suffix in (".yaml", ".yml") and p.exists():
            # it’s a file path → use initialize_config_dir
            with initialize_config_dir(version_base=None, config_dir=str(p.parent)):
                cfg = compose(config_name=p.stem, overrides=[])
        else:
            # treat as config name in Hydra search path
            cfg = compose(config_name=config_name_or_path, overrides=[])
        # then proceed to build the model

        model = build_sam2(cfg, checkpoint=checkpoint, device=device, apply_postprocessing=apply_postprocessing)



        return model


    sam2_model = build_sam2(str(config_path), str(checkpoint_path), device=DEVICE, apply_postprocessing=False)

    logger.info("[MODEL] Instantiating mask generator")
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=cli_args.points_per_side,
        points_per_batch=cli_args.points_per_batch,
        pred_iou_thresh=cli_args.pred_iou_thresh,
        stability_score_thresh=cli_args.stability_score_thresh,
        stability_score_offset=cli_args.stability_offset,
        crop_n_layers=0,  # Set to 0 since we're already tiling
        box_nms_thresh=cli_args.box_nms_thresh,
        min_mask_region_area=cli_args.min_mask_area,
    )

    logger.info(f"[IMAGE] Reading image from {image_path}")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image at {image_path}")
    logger.debug(f"[IMAGE] image_bgr shape: {image_bgr.shape}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Generate patches
    patches = generate_patches(image_rgb, patch_size=cli_args.patch_size, overlap=cli_args.overlap)

    # Process patches into a single label map (avoids RAM blow-up)
    labels = process_patches_to_labelmap(
        patches,
        mask_generator_2,
        image_shape=image_rgb.shape[:2],
        overlap=cli_args.overlap
    )

    print_vram_usage()

    # Create colored visualization from label map
    annotated_image = labelmap_to_colored_mask(labels, image_bgr)

    # Get next available output folder
    output_dir = get_next_output_folder(output_root, prefix=cli_args.output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SAVE] Using output folder: {output_dir}")

    logger.info("[SAVE] Calling plot_images_and_save ...")
    plot_images_and_save(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image'],
        save_path=output_dir / "comparison_grid.png"
    )

    # Save individual high-quality images
    save_individual_images(image_bgr, annotated_image, labels, output_dir=output_dir)

    logger.info("=" * 80)
    logger.info(f"[TIME] Total script time: {time.time() - t_start:.2f}s")
    logger.info(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}Pipeline Complete!{Colors.RESET}")
    logger.info("=" * 80)
    print_vram_usage()

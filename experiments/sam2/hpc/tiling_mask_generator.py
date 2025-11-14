"""
SAM2 Tiling-Based Segmentation Pipeline

This module implements a tiling-based approach for segmenting ultra-large images
using the SAM2 (Segment Anything Model 2) architecture. It handles multi-channel
imagery and uses a patch-based processing strategy to manage memory efficiently.

Design Patterns Used:
    - Strategy Pattern: Multiple image loading strategies (OpenCV, rasterio, tifffile)
    - Factory Pattern: Dynamic image loader selection
    - Builder Pattern: Incremental label map construction

Author: mak8427
Date: 2025-10-28
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from matplotlib import pyplot as plt

# Project structure setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAM2_REPO = PROJECT_ROOT / "third_party" / "sam2"
if SAM2_REPO.exists():
    sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Default paths
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "sam2" / "models" / "sam2_hiera_large.pt"
DEFAULT_MODEL_CONFIG = SAM2_REPO / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml"
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "samples" / "imagery" / "1084-1393.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "outputs" / "sam2" / "hpc"


# ============================================================================
# CONFIGURATION AND ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the segmentation pipeline.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - image (Path): Input image path
            - checkpoint (Path): SAM2 model checkpoint path
            - model_config (Path): SAM2 configuration YAML path
            - output_dir (Path): Output directory for results
            - patch_size (int): Size of each patch for tiling
            - overlap (int): Overlap between adjacent patches
            - points_per_side (int): SAM2 parameter for prompt density
            - points_per_batch (int): SAM2 batch size for prompts
            - pred_iou_thresh (float): IoU threshold for mask filtering
            - stability_score_thresh (float): Stability threshold for masks
            - stability_offset (float): Stability calculation offset
            - box_nms_thresh (float): NMS threshold for bounding boxes
            - min_mask_area (int): Minimum mask area in pixels
            - output_prefix (str): Prefix for output folder naming
    """
    parser = argparse.ArgumentParser(
        description="Tile-based SAM2 automatic mask generator for ultra-large scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Input image to segment."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="SAM2 checkpoint path."
    )
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

    # Tiling parameters
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2048,
        help="Patch size for tiling (pixels)."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between patches (pixels)."
    )

    # SAM2 parameters
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=256,
        help="SAM2 mask generator parameter: points per side of grid."
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help="SAM2 mask generator parameter: batch size for point processing."
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.5,
        help="Minimum predicted IoU to keep a mask."
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.92,
        help="Minimum stability score when accepting masks.",
    )
    parser.add_argument(
        "--stability-offset",
        type=float,
        default=0.7,
        help="Stability offset hyperparameter."
    )
    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="Box NMS threshold."
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=1000,
        help="Minimum mask region area (pixels)."Enjoy your weekend!
    )

    # Output naming
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="run",
        help="Prefix used when auto-incrementing output folders.",
    )

    return parser.parse_args()


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class Colors:
    """ANSI color codes for terminal output formatting."""

    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Standard foreground colors
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
    """
    Custom logging formatter with ANSI color codes and semantic tags.

    Provides colored output for different log levels and highlights
    semantic tags like [MODEL], [VRAM], [PATCH], etc.
    """

    # Tag-specific colors for semantic highlighting
    TAG_COLORS = {
        'VRAM': Colors.CYAN,
        'PATCH': Colors.MAGENTA,
        'SAVE': Colors.GREEN,
        'MODEL': Colors.BLUE,
        'IMAGE': Colors.YELLOW,
        'MEMORY': Colors.BRIGHT_CYAN,
        'TIME': Colors.BRIGHT_GREEN,
    }

    # Log level colors
    LEVEL_COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.WHITE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colors and semantic tag highlighting.

        Args:
            record (logging.LogRecord): Log record to format

        Returns:
            str: Formatted log string with ANSI color codes
        """
        # Color the level name
        level_color = self.LEVEL_COLORS.get(record.levelname, Colors.WHITE)
        colored_level = f"{level_color}{record.levelname}{Colors.RESET}"

        # Format timestamp
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        colored_timestamp = f"{Colors.BRIGHT_BLACK}{timestamp}{Colors.RESET}"

        # Get message and apply tag colors
        message = record.getMessage()
        colored_message = message

        # Apply semantic tag colors
        for tag, color in self.TAG_COLORS.items():
            tag_pattern = f"[{tag}]"
            if tag_pattern in message:
                colored_tag = f"{color}{Colors.BOLD}[{tag}]{Colors.RESET}"
                colored_message = colored_message.replace(tag_pattern, colored_tag)

        return f"{colored_timestamp} - {colored_level} - {colored_message}"


class FileFormatter(logging.Formatter):
    """Plain text formatter for file output (no ANSI color codes)."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as plain text.

        Args:
            record (logging.LogRecord): Log record to format

        Returns:
            str: Formatted log string without colors
        """
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        return f"{timestamp} - {record.levelname} - {record.getMessage()}"


def setup_logger() -> logging.Logger:
    """
    Configure and return the application logger.

    Sets up dual output: colored console output and plain file output.

    Returns:
        logging.Logger: Configured logger instance with handlers attached
    """
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

    return logger


# Initialize logger
logger = setup_logger()


# ============================================================================
# FILE SYSTEM UTILITIES
# ============================================================================

def get_next_output_folder(base_dir: Path, prefix: str = "output_job") -> Path:
    """
    Find the next available output folder name with auto-incrementing suffix.

    Scans the base directory for existing folders matching the pattern
    "{prefix}_{number}" and returns the next available number.

    Args:
        base_dir (Path): Directory where output folders are created
        prefix (str): Folder prefix (default: "output_job")

    Returns:
        Path: Next available folder path (e.g., .../output_job_3)

    Example:
        >>> import tempfile
        >>> temp_dir = Path(tempfile.mkdtemp())
        >>> folder = get_next_output_folder(temp_dir, "run")
        >>> folder.name
        'run_0'
        >>> folder.mkdir(exist_ok=True)
        >>> folder2 = get_next_output_folder(temp_dir, "run")
        >>> folder2.name
        'run_1'
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find existing folders matching pattern
    existing_folders = sorted(base_dir.glob(f"{prefix}_*"))
    if not existing_folders:
        return base_dir / f"{prefix}_0"

    # Extract indices from folder names
    indices = []
    for folder in existing_folders:
        try:
            num = int(folder.name.split("_")[-1])
            indices.append(num)
        except ValueError:
            continue

    # Return next available index
    next_index = (max(indices) + 1) if indices else 0
    return base_dir / f"{prefix}_{next_index}"


# ============================================================================
# IMAGE LOADING (STRATEGY PATTERN)
# ============================================================================

def load_image_opencv(image_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load image using OpenCV (fast, supports standard formats).

    Args:
        image_path (Path): Path to image file

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple of (BGR, RGB) images or None
            - BGR image: shape (H, W, 3), dtype uint8
            - RGB image: shape (H, W, 3), dtype uint8
        Returns None if loading fails.
    """
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is not None:
        logger.debug(f"[IMAGE] Loaded with OpenCV, shape: {image_bgr.shape}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_bgr, image_rgb

    return None


def normalize_to_uint8(image: np.ndarray, use_percentile: bool = True) -> np.ndarray:
    """
    Normalize image data to uint8 range [0, 255].

    Args:
        image (np.ndarray): Input image of any dtype
        use_percentile (bool): If True, use 2nd and 98th percentiles for scaling
                              to improve contrast. If False, use min/max.

    Returns:
        np.ndarray: Normalized image, shape preserved, dtype uint8

    Example:
        >>> img = np.array([[0.0, 0.5], [0.75, 1.0]])
        >>> result = normalize_to_uint8(img, use_percentile=False)
        >>> result
        array([[  0, 127],
               [191, 255]], dtype=uint8)
        >>> img_uint8 = np.array([[0, 128], [192, 255]], dtype=np.uint8)
        >>> normalize_to_uint8(img_uint8) is img_uint8
        True
    """
    if image.dtype == np.uint8:
        return image

    # Handle float images in [0, 1] range
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)

    # Scale using percentiles or min/max
    if use_percentile:
        vmin, vmax = np.percentile(image, [2, 98])
    else:
        vmin, vmax = image.min(), image.max()

    # Clip and scale to [0, 255]
    normalized = np.clip((image - vmin) / (vmax - vmin) * 255, 0, 255)
    return normalized.astype(np.uint8)


def load_image_rasterio(image_path: Path, rgb_order: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multi-channel geospatial image using rasterio.

    Handles multi-spectral imagery by selecting the first 3 channels
    as RGB. Applies percentile-based normalization for better contrast.

    Args:
        image_path (Path): Path to image file (typically GeoTIFF)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (BGR, RGB) images
            - BGR image: shape (H, W, 3), dtype uint8
            - RGB image: shape (H, W, 3), dtype uint8

    Raises:
        ValueError: If image has unsupported number of channels
        ImportError: If rasterio is not installed
    """
    import rasterio

    with rasterio.open(image_path) as src:
        logger.info(f"[IMAGE] Image has {src.count} channels, "
                   f"shape: {src.height}x{src.width}")

        # Read data: shape (channels, height, width)
        data = src.read()

        # Select appropriate channels
        if src.count >= 3:
            # Take first 3 channels as RGB
            rgb_data = data[rgb_order, :, :]
        elif src.count == 1:
            # Replicate grayscale to 3 channels
            rgb_data = np.repeat(data, 3, axis=0)
        else:
            raise ValueError(f"Cannot handle {src.count} channels")

        # Transpose to (H, W, C) format
        image_rgb = np.transpose(rgb_data, (1, 2, 0))

        # Normalize to uint8
        image_rgb = normalize_to_uint8(image_rgb, use_percentile=True)

        # Convert RGB to BGR for OpenCV compatibility
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        logger.info(f"[IMAGE] Loaded with rasterio, shape: {image_rgb.shape}")

        return image_bgr, image_rgb


def load_image_tifffile(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TIFF image using tifffile library.

    Handles various TIFF formats including multi-channel and
    multi-page TIFFs. Automatically detects channel ordering.

    Args:
        image_path (Path): Path to TIFF file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (BGR, RGB) images
            - BGR image: shape (H, W, 3), dtype uint8
            - RGB image: shape (H, W, 3), dtype uint8

    Raises:
        ValueError: If image shape is unexpected
        ImportError: If tifffile is not installed
    """
    import tifffile

    data = tifffile.imread(image_path)
    logger.info(f"[IMAGE] Loaded with tifffile, shape: {data.shape}")

    # Handle different data layouts
    if data.ndim == 2:
        # Grayscale (H, W)
        image_rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

    elif data.ndim == 3:
        # Check if channels-last (H, W, C) or channels-first (C, H, W)
        if data.shape[2] > 3 and data.shape[2] < data.shape[0]:
            # Channels-last: (H, W, C) where C > 3
            image_rgb = data[:, :, :3]
        elif data.shape[0] > 3 and data.shape[0] < 20 and data.shape[0] < data.shape[1]:
            # Channels-first: (C, H, W) where C > 3
            image_rgb = np.transpose(data[:3], (1, 2, 0))
        elif data.shape[2] <= 3:
            # Standard RGB: (H, W, 3)
            image_rgb = data
        else:
            raise ValueError(f"Unexpected image shape: {data.shape}")
    else:
        raise ValueError(f"Unexpected number of dimensions: {data.ndim}")

    # Normalize to uint8
    image_rgb = normalize_to_uint8(image_rgb, use_percentile=True)

    # Handle grayscale
    if image_rgb.ndim == 2 or (image_rgb.ndim == 3 and image_rgb.shape[2] == 1):
        image_rgb = cv2.cvtColor(image_rgb.squeeze(), cv2.COLOR_GRAY2RGB)

    # Convert to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr, image_rgb


def load_image_smart(image_path: Path, rgb_order: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smart image loader with automatic format detection and fallback.

    Uses a strategy pattern to try multiple loading methods in sequence:
    1. OpenCV (fastest, standard formats)
    2. Rasterio (geospatial, multi-channel)
    3. Tifffile (complex TIFF formats)

    Args:
        image_path (Path): Path to image file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (BGR, RGB) images
            - BGR image: shape (H, W, 3), dtype uint8, for OpenCV operations
            - RGB image: shape (H, W, 3), dtype uint8, for SAM2 model

    Raises:
        RuntimeError: If all loading strategies fail or required libraries
                     are not installed

    Note:
        This function tries multiple strategies and may log warnings
        if earlier strategies fail before succeeding with a later one.
    """
    logger.info(f"[IMAGE] Reading image from {image_path}")

    # Strategy 1: Try OpenCV first (fastest)
    result = load_image_opencv(image_path)
    if result is not None:
        return result

    # Strategy 2: Try rasterio for multi-channel imagery
    logger.info("[IMAGE] OpenCV failed, trying rasterio for multi-channel TIFF")
    try:
        return load_image_rasterio(image_path, rgb_order)
    except ImportError:
        logger.warning("[IMAGE] rasterio not available, trying tifffile")
    except Exception as e:
        logger.warning(f"[IMAGE] rasterio failed: {e}, trying tifffile")

    # Strategy 3: Try tifffile as last resort
    try:
        return load_image_tifffile(image_path)
    except ImportError:
        raise RuntimeError(
            "Failed to load image. Install rasterio or tifffile:\n"
            "  pip install rasterio tifffile"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load image at {image_path}: {e}")


# ============================================================================
# IMAGE TILING AND PATCH GENERATION
# ============================================================================

def validate_patch_parameters(image_shape: Tuple[int, int],
                              patch_size: int,
                              overlap: int) -> None:
    """
    Validate patch generation parameters against image dimensions.

    Args:
        image_shape (Tuple[int, int]): Image shape as (height, width)
        patch_size (int): Size of each patch in pixels
        overlap (int): Overlap between patches in pixels

    Raises:
        ValueError: If parameters are invalid (image too small, overlap too large, etc.)

    Example:
        >>> validate_patch_parameters((2048, 2048), 1024, 128)
        >>> validate_patch_parameters((100, 100), 1024, 128)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Image too small...
    """
    h, w = image_shape

    if h < overlap or w < overlap:
        raise ValueError(
            f"Image too small ({h}x{w}) for overlap {overlap}. "
            f"Reduce overlap or use a smaller value."
        )

    if overlap >= patch_size:
        raise ValueError(
            f"Overlap ({overlap}) must be smaller than patch_size ({patch_size})"
        )

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")


def extract_patch(image: np.ndarray,
                  x: int,
                  y: int,
                  patch_size: int) -> np.ndarray:
    """
    Extract a single patch from an image with zero-padding if needed.

    Args:
        image (np.ndarray): Source image, shape (H, W, C)
        x (int): Left coordinate of patch
        y (int): Top coordinate of patch
        patch_size (int): Size of square patch

    Returns:
        np.ndarray: Extracted patch, shape (patch_size, patch_size, C), dtype matches input
            Padded with zeros if patch extends beyond image boundaries.

    Example:
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        >>> patch = extract_patch(img, 0, 0, 50)
        >>> patch.shape
        (50, 50, 3)
        >>> patch = extract_patch(img, 80, 80, 50)
        >>> patch.shape
        (50, 50, 3)
        >>> bool(np.sum(patch == 0) > 0)  # Has padding
        True
    """
    h, w = image.shape[:2]

    # Calculate actual bounds
    y_end = min(y + patch_size, h)
    x_end = min(x + patch_size, w)

    # Extract the valid region
    patch = image[y:y_end, x:x_end]

    # Pad if necessary
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        return padded

    return patch


def generate_patches(image: np.ndarray,
                     patch_size: int = 1024,
                     overlap: int = 128) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split image into overlapping square patches using a sliding window approach.

    This function implements a sliding window strategy to divide large images
    into manageable patches. Patches are generated in row-major order (left to
    right, top to bottom). Edge patches are zero-padded to maintain consistent size.

    Args:
        image (np.ndarray): Input image, shape (H, W, C)
        patch_size (int): Size of each square patch in pixels (default: 1024)
        overlap (int): Overlap between adjacent patches in pixels (default: 128)

    Returns:
        List[Tuple[np.ndarray, int, int]]: List of tuples containing:
            - patch (np.ndarray): Image patch, shape (patch_size, patch_size, C)
            - x_offset (int): Left coordinate of patch in original image
            - y_offset (int): Top coordinate of patch in original image

    Raises:
        ValueError: If image is too small for the given overlap or invalid parameters

    Example:
        >>> image = np.zeros((2560, 3584, 3), dtype=np.uint8)
        >>> patches = generate_patches(image, patch_size=1024, overlap=512)
        >>> len(patches)
        35
        >>> patch, x, y = patches[0]
        >>> patch.shape
        (1024, 1024, 3)
        >>> x, y
        (0, 0)
        >>> all(p.shape == (1024, 1024, 3) for p, _, _ in patches)
        True
    """
    h, w = image.shape[:2]

    # Validate parameters
    validate_patch_parameters((h, w), patch_size, overlap)

    # Calculate stride (distance between patch origins)
    stride = patch_size - overlap
    patches = []

    # Generate patches in row-major order
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = extract_patch(image, x, y, patch_size)
            patches.append((patch, x, y))

    logger.info(
        f"[PATCH] Generated {len(patches)} patches of size {patch_size}x{patch_size} "
        f"with {overlap}px overlap (stride={stride})"
    )

    return patches


# ============================================================================
# MASK GENERATION AND LABEL MAP CONSTRUCTION
# ============================================================================

def generate_masks_for_patch(patch: np.ndarray,
                            mask_generator: SAM2AutomaticMaskGenerator) -> List[dict]:
    """
    Generate segmentation masks for a single patch using SAM2.

    Uses mixed precision (bfloat16) and inference mode for memory efficiency.

    Args:
        patch (np.ndarray): RGB image patch, shape (H, W, 3), dtype uint8
        mask_generator (SAM2AutomaticMaskGenerator): Initialized SAM2 mask generator

    Returns:
        List[dict]: List of mask dictionaries, each containing:
            - 'segmentation' (np.ndarray): Boolean mask, shape (H, W)
            - 'area' (int): Mask area in pixels
            - 'bbox' (List[int]): Bounding box [x, y, w, h]
            - 'predicted_iou' (float): Predicted IoU score
            - 'stability_score' (float): Mask stability score

    Note:
        This function is optimized for CUDA inference and uses automatic
        mixed precision to reduce memory usage.
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = mask_generator.generate(patch)

    return masks


def paint_masks_to_labelmap(masks: List[dict],
                            labels: np.ndarray,
                            offset_x: int,
                            offset_y: int,
                            next_id: int) -> Tuple[int, int]:
    """
    Paint segmentation masks onto a label map at specified offset.

    Implements a first-come-first-served strategy: pixels are only painted
    if they are currently labeled as background (0). This prevents overlap
    conflicts when processing adjacent patches.

    Args:
        masks (List[dict]): List of mask dictionaries from SAM2
        labels (np.ndarray): Global label map, shape (H, W), dtype int32
            Modified in-place to add new masks.
        offset_x (int): Horizontal offset of patch in global coordinates
        offset_y (int): Vertical offset of patch in global coordinates
        next_id (int): Next available label ID to assign

    Returns:
        Tuple[int, int]: Tuple containing:
            - next_id (int): Updated next available label ID
            - painted_count (int): Number of masks successfully painted

    Note:
        Masks are sorted by area (smallest first) before painting to prevent
        small objects from being overwritten by larger ones.

    Example:
        >>> labels = np.zeros((100, 100), dtype=np.int32)
        >>> mask1 = np.zeros((100, 100), dtype=bool)
        >>> mask1[10:20, 10:20] = True
        >>> masks = [{'segmentation': mask1, 'area': 100}]
        >>> next_id, count = paint_masks_to_labelmap(masks, labels, 0, 0, 1)
        >>> count
        1
        >>> next_id
        2
        >>> int(np.sum(labels > 0))
        100
    """
    H, W = labels.shape

    # Sort masks by area (small first) to preserve small objects
    masks.sort(key=lambda m: m.get("area", 0))

    painted_count = 0

    for mask_dict in masks:
        seg = mask_dict["segmentation"].astype(bool)  # shape: (ph, pw)
        ph, pw = seg.shape

        # Calculate bounds in global image space
        y0 = offset_y
        y1 = min(offset_y + ph, H)
        x0 = offset_x
        x1 = min(offset_x + pw, W)

        # Crop segmentation to valid bounds
        seg_crop = seg[:y1 - y0, :x1 - x0]

        # Get region of label map to paint
        paint_region = labels[y0:y1, x0:x1]

        # Only paint where label is currently 0 (background)
        paint_mask = (paint_region == 0) & seg_crop

        if paint_mask.any():
            paint_region[paint_mask] = next_id
            next_id += 1
            painted_count += 1

    return next_id, painted_count


def cleanup_memory(patch_idx: int, cleanup_interval: int = 5) -> None:
    """
    Perform memory cleanup operations for CUDA and Python.

    Synchronizes CUDA operations, empties the cache, and triggers
    garbage collection at regular intervals.

    Args:
        patch_idx (int): Current patch index (0-based)
        cleanup_interval (int): Perform full cleanup every N patches (default: 5)

    Note:
        Always synchronizes CUDA and empties cache. Full garbage collection
        only occurs at specified intervals to balance performance and memory.

    Example:
        >>> cleanup_memory(0)  # First patch - no GC
        >>> cleanup_memory(4)  # Fifth patch - triggers GC
    """
    # Always clean CUDA memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Periodic full cleanup
    if (patch_idx + 1) % cleanup_interval == 0:
        gc.collect()
        logger.info(f"[MEMORY] Full cleanup after patch {patch_idx + 1}")


def process_patches_to_labelmap(patches: List[Tuple[np.ndarray, int, int]],
                                mask_generator: SAM2AutomaticMaskGenerator,
                                image_shape: Tuple[int, int],
                                overlap: int = 128) -> np.ndarray:
    """
    Process image patches and build a unified segmentation label map.

    This function implements an incremental label map construction strategy
    to avoid memory explosion. It processes patches sequentially, generating
    masks for each patch and painting them onto a global label map. Memory
    is aggressively managed to handle ultra-large images.

    Algorithm:
        1. Initialize empty label map (background = 0)
        2. For each patch:
            a. Generate segmentation masks using SAM2
            b. Sort masks by area (small first)
            c. Paint masks onto label map (first-come-first-served)
            d. Clean up memory
        3. Return unified label map

    Args:
        patches (List[Tuple[np.ndarray, int, int]]): List of (patch, x, y) tuples
            from generate_patches()
        mask_generator (SAM2AutomaticMaskGenerator): Initialized SAM2 generator
        image_shape (Tuple[int, int]): Original image shape as (height, width)
        overlap (int): Overlap between patches in pixels (default: 128)
            Currently logged but not used for overlap handling.

    Returns:
        np.ndarray: Label map, shape (H, W), dtype int32
            - 0: background (unlabeled)
            - 1, 2, 3, ...: unique segment IDs
            Each positive integer represents a unique segmented region.

    Note:
        Memory usage scales with image size (O(H*W)), not with number of masks,
        making this approach suitable for ultra-large images. Peak GPU memory
        usage is logged after each patch.
    """
    H, W = image_shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_id = 1  # Start labeling from 1 (0 is background)

    logger.info(f"[PATCH] Processing {len(patches)} patches into label map")

    for idx, (patch_rgb, ox, oy) in enumerate(patches):
        logger.info(f"[PATCH] Processing patch {idx + 1}/{len(patches)} "
                   f"at offset ({ox}, {oy})")

        t_start = time.time()

        # Generate masks for this patch
        patch_masks = generate_masks_for_patch(patch_rgb, mask_generator)

        elapsed = time.time() - t_start
        logger.debug(f"[PATCH] Patch {idx + 1} generated {len(patch_masks)} masks "
                    f"in {elapsed:.2f}s")

        # Paint masks onto global label map
        next_id, painted_count = paint_masks_to_labelmap(
            patch_masks, labels, ox, oy, next_id
        )

        logger.debug(f"[PATCH] Painted {painted_count}/{len(patch_masks)} masks "
                    f"from patch {idx + 1}")

        # Log memory usage
        print_vram_usage()

        # Clean up memory
        del patch_masks
        cleanup_memory(idx, cleanup_interval=5)

    # Final statistics
    unique_labels = np.unique(labels)
    num_segments = len(unique_labels) - 1  # Exclude background
    logger.info(f"[PATCH] Label map complete: {num_segments} unique masks "
               f"(excluding background)")

    return labels


# ============================================================================
# VISUALIZATION AND OUTPUT
# ============================================================================
def create_colormap(num_labels: int, seed: int = 42) -> np.ndarray:
    """
    Create a random colormap for label visualization.

    Args:
        num_labels (int): Number of unique labels (excluding background)
        seed (int): Random seed for reproducibility (default: 42)

    Returns:
        np.ndarray: Color lookup table, shape (num_labels + 1, 3), dtype uint8
            Index 0 (background) is black [0, 0, 0].
            Indices 1+ have random RGB colors.

    Example:
        >>> colors = create_colormap(3, seed=42)
        >>> colors.shape
        (4, 3)
        >>> colors[0].tolist()
        [0, 0, 0]
        >>> colors.dtype
        dtype('uint8')
    """
    np.random.seed(seed)
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    return colors


def labelmap_to_colored_mask(labels: np.ndarray,
                             image_bgr: np.ndarray,
                             alpha: float = 0.5) -> np.ndarray:
    """
    Convert label map to colored visualization overlay.

    Each unique label is assigned a random color for visualization.
    The colored mask is alpha-blended with the original image.

    Args:
        labels (np.ndarray): Label map, shape (H, W), dtype int32
            0 represents background, positive integers are segment IDs.
        image_bgr (np.ndarray): Original BGR image, shape (H, W, 3), dtype uint8
        alpha (float): Transparency of colored overlay (default: 0.5)
            0.0 = fully transparent (original image only)
            1.0 = fully opaque (colored mask only)

    Returns:
        np.ndarray: Annotated BGR image, shape (H, W, 3), dtype uint8
            Original image blended with colored segmentation masks.

    Example:
        >>> labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
        >>> image = np.ones((2, 2, 3), dtype=np.uint8) * 128
        >>> colored = labelmap_to_colored_mask(labels, image, alpha=0.5)
        >>> colored.shape
        (2, 2, 3)
        >>> colored.dtype
        dtype('uint8')
    """
    logger.info("[IMAGE] Creating colored visualization from label map")

    num_labels = labels.max()

    if num_labels == 0:
        logger.warning("[IMAGE] No masks found in label map")
        return image_bgr.copy()

    # Create colormap and apply to labels
    colors = create_colormap(num_labels, seed=42)
    colored_mask = colors[labels]

    # Alpha blend with original image
    annotated = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)

    logger.info(f"[IMAGE] Created visualization with {num_labels} colored segments")

    return annotated


def plot_images_and_save(images: List[np.ndarray],
                        grid_size: Tuple[int, int],
                        titles: List[str],
                        save_path: Path,
                        dpi: int = 300) -> None:
    """
    Save multiple images in a grid layout with high quality.

    Creates a matplotlib figure with images arranged in a grid, each with
    an optional title. The output is saved at high DPI for publication quality.

    Args:
        images (List[np.ndarray]): List of BGR images to display
        grid_size (Tuple[int, int]): Grid dimensions as (nrows, ncols)
        titles (List[str]): Titles for each image (can be empty)
        save_path (Path): Output file path
        dpi (int): Dots per inch for output (default: 300 for high quality)

    Returns:
        None: Saves figure to disk

    Note:
        - Images are converted from BGR to RGB for display
        - Figure size is calculated to maintain aspect ratio
        - Tight layout is used to minimize whitespace
        - Figure is closed after saving to free memory
    """
    logger.debug(f"[SAVE] Creating grid: {grid_size}, images: {len(images)}")

    t_start = time.time()
    nrows, ncols = grid_size

    # Calculate figure size based on image dimensions
    if len(images) > 0 and hasattr(images[0], 'shape'):
        img_h, img_w = images[0].shape[:2]
        fig_w = (img_w * ncols) / dpi
        fig_h = (img_h * nrows) / dpi
        figsize = (fig_w, fig_h)
    else:
        figsize = (8 * ncols, 8 * nrows)

    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    # Plot each image
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]

            # Convert BGR to RGB
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"[SAVE] Color conversion failed for image {idx}: {e}")
                rgb = img

            ax.imshow(rgb, interpolation='bilinear')

            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)

        ax.axis("off")

    # Optimize layout and save
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    elapsed = time.time() - t_start
    logger.info(f"[SAVE] Saved grid to {save_path} in {elapsed:.3f}s at {dpi} DPI")


def save_individual_images(image_bgr: np.ndarray,
                          annotated_image: np.ndarray,
                          labels: np.ndarray,
                          output_dir: Path) -> None:
    """
    Save individual output images without compression.

    Saves three files:
        1. original.png - Original input image
        2. annotated.png - Image with colored segmentation overlay
        3. labels.tif - Raw label map as 16-bit TIFF

    Args:
        image_bgr (np.ndarray): Original BGR image, shape (H, W, 3), dtype uint8
        annotated_image (np.ndarray): Annotated BGR image, shape (H, W, 3), dtype uint8
        labels (np.ndarray): Label map, shape (H, W), dtype int32
        output_dir (Path): Output directory path

    Returns:
        None: Saves files to disk

    Note:
        - PNG files use no compression for maximum quality
        - Label map is converted to uint16 for TIFF compatibility
        - Directory is created if it doesn't exist
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[SAVE] Saving individual images to {output_dir}/")

    # Save original image (PNG, no compression)
    original_path = output_dir / "original.png"
    cv2.imwrite(str(original_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save annotated image (PNG, no compression)
    annotated_path = output_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save label map (16-bit TIFF)
    labels_path = output_dir / "labels.tif"
    cv2.imwrite(str(labels_path), labels.astype(np.uint16))

    logger.info(f"[SAVE] Saved:")
    logger.info(f"[SAVE]   - {original_path.name}")
    logger.info(f"[SAVE]   - {annotated_path.name}")
    logger.info(f"[SAVE]   - {labels_path.name}")


# ============================================================================
# GPU AND SYSTEM UTILITIES
# ============================================================================

def print_vram_usage(device: int = 0) -> None:
    """
    Log current CUDA memory usage statistics.

    Prints allocated, reserved, free, and total VRAM, as well as peak usage.
    Synchronizes CUDA operations before reporting for accuracy.

    Args:
        device (int): CUDA device index (default: 0)

    Returns:
        None: Logs memory statistics

    Note:
        Only executes if CUDA is available. Silently returns otherwise.
    """
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize(device)

    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_allocated(device)

    logger.info(
        f"[VRAM] allocated: {alloc/1024**2:.2f} MiB, "
        f"reserved: {reserved/1024**2:.2f} MiB, "
        f"free: {free/1024**2:.2f} MiB, "
        f"total: {total/1024**2:.2f} MiB, "
        f"peak: {peak/1024**2:.2f} MiB"
    )


def setup_gpu() -> torch.device:
    """
    Configure GPU settings and return appropriate device.

    Enables TF32 for Ampere+ GPUs (compute capability >= 8.0) for faster
    matrix operations. Logs GPU properties and memory status.

    Returns:
        torch.device: CUDA device if available, CPU otherwise

    Note:
        This function has side effects (enables TF32, logs messages).
        Output varies based on hardware availability.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"[MODEL] GPU device properties: {props}")

        # Enable TF32 for Ampere and newer GPUs
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[MODEL] TF32 enabled for faster computation")

        device = torch.device('cuda')
    else:
        logger.warning("[MODEL] CUDA not available, using CPU")
        device = torch.device('cpu')

    logger.info(f"[MODEL] Device: {device}")
    print_vram_usage()

    return device


def validate_paths(checkpoint_path: Path,
                  config_path: Path,
                  image_path: Path) -> None:
    """
    Validate that all required file paths exist.

    Args:
        checkpoint_path (Path): SAM2 checkpoint file path
        config_path (Path): SAM2 config YAML path
        image_path (Path): Input image path

    Raises:
        FileNotFoundError: If any required file is missing

    Example:
        >>> import tempfile
        >>> temp_file = Path(tempfile.NamedTemporaryFile(delete=False).name)
        >>> temp_file.touch()
        >>> validate_paths(temp_file, temp_file, temp_file)
        >>> temp_file.unlink()
        >>> validate_paths(temp_file, temp_file, temp_file)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: Checkpoint not found: ...
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_sam2_model(config_path: Path,
                         checkpoint_path: Path,
                         device: torch.device) -> torch.nn.Module:
    """
    Initialize SAM2 model with proper Hydra configuration.

    Handles Hydra's global state management by clearing any existing
    instance before initialization. Uses initialize_config_dir to point
    Hydra to the correct configuration directory.

    Args:
        config_path (Path): Path to SAM2 config YAML file (must be absolute)
        checkpoint_path (Path): Path to model checkpoint file
        device (torch.device): Device to load model on (cuda or cpu)

    Returns:
        torch.nn.Module: Initialized SAM2 model in evaluation mode

    Raises:
        FileNotFoundError: If config or checkpoint not found
        RuntimeError: If Hydra initialization fails

    Note:
        config_path must be an absolute path for Hydra's initialize_config_dir.
    """
    logger.info("[MODEL] Building SAM2 model")

    # Clear existing Hydra state
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Ensure absolute path
    config_path = config_path.resolve()
    config_name = config_path.stem  # e.g., "sam2_hiera_l"
    config_dir = str(config_path.parent)

    logger.info(f"[MODEL] Config directory: {config_dir}")
    logger.info(f"[MODEL] Config name: {config_name}")

    # Initialize Hydra and build model
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(device),
            apply_postprocessing=False
        )

    return model


def create_mask_generator(model: torch.nn.Module,
                         args: argparse.Namespace) -> SAM2AutomaticMaskGenerator:
    """
    Create SAM2 automatic mask generator with specified parameters.

    Args:
        model (torch.nn.Module): Initialized SAM2 model
        args (argparse.Namespace): Command-line arguments containing generator parameters

    Returns:
        SAM2AutomaticMaskGenerator: Configured mask generator instance

    Note:
        Sets crop_n_layers=0 since we handle tiling externally.
    """
    logger.info("[MODEL] Instantiating mask generator")

    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_offset,
        crop_n_layers=0,  # No internal cropping - we handle tiling
        box_nms_thresh=args.box_nms_thresh,
        min_mask_region_area=args.min_mask_area,
    )

    return generator


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_segmentation_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the complete SAM2 segmentation pipeline.

    Pipeline stages:
        1. Setup: GPU configuration, path validation
        2. Model initialization: Load SAM2 model and mask generator
        3. Image loading: Smart loading with format detection
        4. Patch generation: Split image into overlapping patches
        5. Segmentation: Process patches and build label map
        6. Visualization: Create colored overlay
        7. Output: Save results to disk

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        None: Saves results to disk

    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If image loading or model initialization fails
    """
    t_start = time.time()

    # ========================================================================
    # Stage 1: Setup
    # ========================================================================
    logger.info("=" * 80)
    logger.info(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}"
               f"SAM2 Segmentation Pipeline Started{Colors.RESET}")
    logger.info("=" * 80)

    # Setup GPU
    device = setup_gpu()

    # Resolve and validate paths
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.model_config.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    rgb_order = [5, 3, 1]
    validate_paths(checkpoint_path, config_path, image_path)

    # ========================================================================
    # Stage 2: Model Initialization
    # ========================================================================
    sam2_model = initialize_sam2_model(config_path, checkpoint_path, device)
    mask_generator = create_mask_generator(sam2_model, args)

    # ========================================================================
    # Stage 3: Image Loading
    # ========================================================================
    image_bgr, image_rgb = load_image_smart(image_path, rgb_order=rgb_order)
    logger.info(f"[IMAGE] Loaded image shape: {image_rgb.shape}")

    # ========================================================================
    # Stage 4: Patch Generation
    # ========================================================================
    patches = generate_patches(
        image_rgb,
        patch_size=args.patch_size,
        overlap=args.overlap
    )

    # ========================================================================
    # Stage 5: Segmentation
    # ========================================================================
    labels = process_patches_to_labelmap(
        patches,
        mask_generator,
        image_shape=image_rgb.shape[:2],
        overlap=args.overlap
    )

    print_vram_usage()

    # ========================================================================
    # Stage 6: Visualization
    # ========================================================================
    annotated_image = labelmap_to_colored_mask(labels, image_bgr, alpha=0.5)

    # ========================================================================
    # Stage 7: Output
    # ========================================================================
    output_dir = get_next_output_folder(output_root, prefix=args.output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SAVE] Using output folder: {output_dir}")

    # Save comparison grid
    plot_images_and_save(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['Source Image', 'Segmented Image'],
        save_path=output_dir / "comparison_grid.png",
        dpi=300
    )

    # Save individual images
    save_individual_images(image_bgr, annotated_image, labels, output_dir)

    # ========================================================================
    # Final Statistics
    # ========================================================================
    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"[TIME] Total pipeline time: {elapsed:.2f}s")
    logger.info(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}"
               f"Pipeline Complete!{Colors.RESET}")
    logger.info("=" * 80)
    print_vram_usage()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main entry point for the SAM2 segmentation pipeline.

    Parses command-line arguments and executes the segmentation pipeline.
    Handles top-level exceptions and ensures proper logging.
    """
    try:
        args = parse_args()
        run_segmentation_pipeline(args)
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPT] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
# SPDX-License-Identifier: GPL-2.0
#
# Single-scale zero-shot transfer from Image A -> Image B using DINOv3.
#
# Pipeline:
# - Extract DINOv3 patch features on Image A and B.
# - Build positive/negative banks from Image A using SH_2022 labels.
# - GPU kNN classification on Image B (score map + saliency).
# - Clip predictions on B using a buffered SH_2022 mask.
# - Grid search over k and threshold to pick best parameters (raw only).
# - Fine-tune the threshold around the coarse optimum (raw only).
# - DenseCRF refinement with unary calibrated around the best validation
#   threshold instead of min-max.
# - Extended: negative bank scoring (pos - alpha * neg), timing, oracle
#   upper bound for SH buffer, and CRF hyperparameter grid search.

import time
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from skimage.morphology import erosion, disk, dilation

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject, Resampling
import rasterio.features as rfeatures
from rasterio.crs import CRS

import fiona
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# Optional fast morphology backend
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from transformers import AutoImageProcessor, AutoModel
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


# ----------------------------------------------------------------------
# 0. timing utilities + global config
# ----------------------------------------------------------------------

DEBUG_TIMING = True
DEBUG_TIMING_VERBOSE = False  # set True to time lightweight helpers (log noisy)
USE_FP16_KNN = True  # set True to speed up matmul on GPU; may slightly change scores
USE_GPU_THRESHOLD_METRICS = True  # evaluate threshold grid on GPU in batch
THRESHOLD_BATCH_SIZE = 8  # chunk size for GPU threshold evaluation to avoid OOM
CRF_MAX_CONFIGS = 64  # limit CRF grid to avoid huge runtimes
THRESHOLD_CPU_BATCH_SIZE = 16  # CPU chunk size for batched threshold eval

# Negative bank config
MAX_NEG_BANK = 8000      # max number of negative patches (will subsample if larger)
NEG_ALPHA = 1.0          # score = pos_mean - NEG_ALPHA * neg_mean


def time_start():
    """Start a timing block if DEBUG_TIMING is enabled."""
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    """End a timing block and print elapsed time in seconds."""
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt:.3f} s")


# ----------------------------------------------------------------------
# 1. model setup
# ----------------------------------------------------------------------

def init_model(model_name: str):
    """
    Initialize DINOv3 model and image processor on CPU or GPU.

    We keep this logic separate so that `main` stays cleaner and we can
    easily swap the backbone by changing only `model_name`.
    """
    t0 = time_start()

    processor = AutoImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    time_end("init_model", t0)

    return model, processor, device


# ----------------------------------------------------------------------
# 2. data loading, reprojection, and basic preparation
# ----------------------------------------------------------------------

def load_dop20_image(path: str) -> np.ndarray:
    """
    Load a DOP20 GeoTIFF and return an HxWx3 RGB array.

    We rely on rasterio to read the bands in (C, H, W) and reshape to
    image layout (H, W, C). If there are more than 3 bands, we keep
    only the first three (RGB assumption).
    """
    t0 = time_start()
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    img = reshape_as_image(arr)  # (H, W, C)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    time_end(f"load_dop20_image[{os.path.basename(path)}]", t0)
    return img


def reproject_labels_to_image(ref_img_path: str, labels_path: str) -> np.ndarray:
    """
    Reproject a raster label map onto the grid of a reference image.

    We create an in-memory raster with the same transform/CRS/resolution as
    the reference image, reproject the label raster into it with nearest
    neighbor, and return the first band as a 2D array.
    """
    t0 = time_start()
    with rasterio.open(ref_img_path) as ref, rasterio.open(labels_path) as src:
        dst_meta = ref.meta.copy()
        dst_meta.update(dtype=src.dtypes[0], count=src.count)

        memfile = rasterio.io.MemoryFile()
        with memfile.open(**dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_width=ref.width,
                    dst_height=ref.height,
                    resampling=Resampling.nearest,
                )
            labels_arr = dst.read()

    labels_2d = labels_arr[0]
    time_end(
        f"reproject_labels_to_image[{os.path.basename(labels_path)} -> {os.path.basename(ref_img_path)}]",
        t0,
    )
    return labels_2d


def rasterize_vector_labels(vector_path: str,
                            ref_raster_path: str,
                            burn_value: int = 1) -> np.ndarray:
    """
    Rasterize a vector shapefile onto the grid of ref_raster_path.

    - If vector CRS is missing, we assume EPSG:4326 (lat/lon).
    - If CRS differ, geometries are reprojected to raster CRS.

    Returns:
        gt_mask: uint8, shape (H, W), values in {0, burn_value}.
    """
    t0 = time_start()

    with rasterio.open(ref_raster_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        raster_crs = src.crs

    with fiona.open(vector_path, "r") as shp:
        vec_crs = shp.crs

        # If CRS is missing/unknown, assume WGS84 lat/lon.
        if not vec_crs:
            print("[warn] vector CRS is missing/unknown; assuming EPSG:4326 (WGS84)")
            vec_crs = CRS.from_epsg(4326).to_dict()

        # Setup transformer if needed
        transformer = None
        if raster_crs and vec_crs and vec_crs != raster_crs.to_dict():
            print(f"[info] reprojecting vector geometries "
                  f"from {vec_crs} -> {raster_crs.to_dict()}")
            transformer = Transformer.from_crs(
                vec_crs,
                raster_crs.to_dict(),
                always_xy=True,
            )

        shapes = []
        for feat in shp:
            geom = feat["geometry"]
            if transformer is not None:
                geom_obj = shape(geom)
                geom_obj = shp_transform(transformer.transform, geom_obj)
                geom = mapping(geom_obj)
            shapes.append((geom, burn_value))

    gt_mask = rfeatures.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    time_end("rasterize_vector_labels", t0)
    return gt_mask


# ----------------------------------------------------------------------
# 3. helpers (normalization, tiling, label-to-patch, feature I/O)
# ----------------------------------------------------------------------

def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize feature vectors along the last dimension.

    This makes cosine similarity equivalent to dot product, which is
    what we use for kNN scoring on GPU.
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    out = feats / norms
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("l2_normalize", t0)
    return out


def tile_iterator(image_hw3: np.ndarray,
                  labels_hw: np.ndarray | None = None,
                  tile_size: int = 1024,
                  stride: int | None = None):
    """
    Generator that yields overlapping tiles (and optionally label tiles)
    over an HxWx3 image.

    This is used both for DINO feature extraction and for building the
    positive/negative banks in a memory-friendly way.
    """
    h, w = image_hw3.shape[:2]
    if stride is None:
        stride = tile_size

    y = 0
    while y < h:
        x = 0
        y_end = min(y + tile_size, h)
        while x < w:
            x_end = min(x + tile_size, w)
            img_tile = image_hw3[y:y_end, x:x_end]

            if labels_hw is not None:
                lab_tile = labels_hw[y:y_end, x:x_end]
            else:
                lab_tile = None

            yield y, x, img_tile, lab_tile
            x += stride
        y += stride


def crop_to_multiple_of_ps(img_tile_hw3: np.ndarray,
                           labels_tile_hw: np.ndarray | None,
                           ps: int):
    """
    Crop a tile so that both height and width are multiples of the
    DINO patch size `ps`. This avoids awkward incomplete patches.
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps

    img_c = img_tile_hw3[:h_eff, :w_eff]
    if labels_tile_hw is not None:
        lab_c = labels_tile_hw[:h_eff, :w_eff]
    else:
        lab_c = None

    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("crop_to_multiple_of_ps", t0)
    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(labels_tile: np.ndarray,
                          hp: int,
                          wp: int,
                          pos_frac_thresh: float = 0.1):
    """
    Aggregate pixel labels into patch-level positive/negative masks.

    - We reshape the label tile into (Hp, patch_h, Wp, patch_w) and compute
      the foreground fraction within each patch.
    - A patch is positive if foreground fraction >= pos_frac_thresh.
    - A patch is negative if there are zero foreground pixels.
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp

    labels_c = labels_tile[:hp * patch_h, :wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)

    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))  # (Hp, Wp)

    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0

    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("labels_to_patch_masks", t0)
    return pos_mask, neg_mask


def tile_feature_path(feature_dir: str,
                      image_id: str,
                      y: int,
                      x: int) -> str:
    """
    Canonical filename for a tile's features. Keeps the naming scheme
    consistent and easy to grep.
    """
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def save_tile_features(feats_tile: np.ndarray,
                       feature_dir: str,
                       image_id: str,
                       y: int,
                       x: int):
    """
    Save patch features for a single tile as a .npy file.
    """
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))


def consolidate_features_for_image(feature_dir: str,
                                   image_id: str,
                                   output_suffix: str = "_features_full.npy"):
    """
    Concatenate all per-tile feature arrays for an image into a single
    (N_patches, C) array and save it.

    This is mostly for downstream analysis/debugging and not required
    for the pipeline itself.
    """
    t0 = time_start()

    if not os.path.isdir(feature_dir):
        print(f"[warn] feature_dir does not exist: {feature_dir}")
        return None

    prefix = f"{image_id}_y"
    suffix = "_features.npy"
    files = [
        f for f in os.listdir(feature_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]

    if not files:
        print(f"[warn] no feature tiles found for image_id={image_id} in {feature_dir}")
        return None

    files = sorted(files)
    feats_list = []

    for fname in files:
        fpath = os.path.join(feature_dir, fname)
        arr = np.load(fpath)  # (Hp, Wp, C)
        feats_list.append(arr.reshape(-1, arr.shape[-1]))  # (Hp*Wp, C)

    feats_full = np.concatenate(feats_list, axis=0).astype(np.float32)

    out_path = os.path.join(feature_dir, f"{image_id}{output_suffix}")
    np.save(out_path, feats_full)

    time_end(f"consolidate_features_for_image[{image_id}]", t0)
    print(
        f"[info] consolidated {len(files)} tiles for {image_id} -> {out_path}, "
        f"shape={feats_full.shape}"
    )

    return out_path


# ----------------------------------------------------------------------
# 4. single-scale DINOv3 patch feature extraction
# ----------------------------------------------------------------------

def extract_patch_features_single_scale(image_hw3: np.ndarray,
                                        model,
                                        processor,
                                        device,
                                        ps: int = 16,
                                        aggregate_layers=None):
    """
    Extract single-scale DINOv3 patch features from an HxWx3 image.

    - We rely on the HuggingFace processor to handle preprocessing.
    - Optionally average features from multiple layers.
    - Outputs an (Hp, Wp, C) feature tensor normalized to unit length.
    """
    t0 = time_start()
    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)

    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape

    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)

    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens:, :]  # drop CLS + registers

    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps

    assert hp * wp == num_tokens, (
        f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    )

    feats = patch_tokens[0].cpu().numpy().reshape(hp, wp, dim)
    feats = l2_normalize(feats)

    time_end("extract_patch_features_single_scale", t0)
    return feats, hp, wp


# ----------------------------------------------------------------------
# 5. bank building from Image A (single-scale, with caching)
# ----------------------------------------------------------------------

def build_banks_single_scale(img_a: np.ndarray,
                             labels_a: np.ndarray,
                             model,
                             processor,
                             device,
                             ps: int = 16,
                             tile_size: int = 1024,
                             stride: int | None = None,
                             pos_frac_thresh: float = 0.1,
                             aggregate_layers=None,
                             feature_dir: str | None = None,
                             image_id: str | None = None,
                             bank_cache_dir: str | None = None):
    """
    Build positive and negative patch feature banks from Image A.

    - Erode labels to ensure positives are in the interior of SH_2022
      segments (robustness).
    - Positive patches: fraction of positive pixels >= pos_frac_thresh.
    - Negative patches: no positive pixels.
    - Optionally cache banks on disk to avoid recomputation.
    """
    t0 = time_start()

    # Shortcut: load cached banks if available
    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        pos_cache_path = os.path.join(bank_cache_dir, f"{image_id}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{image_id}_neg_bank.npy")
        if os.path.exists(pos_cache_path):
            pos_bank = np.load(pos_cache_path)
            neg_bank = np.load(neg_cache_path) if os.path.exists(neg_cache_path) else None
            time_end("build_banks_single_scale(load_cache)", t0)
            print(f"[cache] loaded banks from {bank_cache_dir}")
            return pos_bank, neg_bank

    pos_list = []
    neg_list = []

    cached_tiles = 0
    computed_tiles = 0

    # Slight erosion to avoid label noise on boundaries
    labels_eroded = erosion((labels_a > 0).astype(bool), disk(2))

    for y, x, img_tile, lab_tile in tile_iterator(img_a,
                                                  labels_eroded,
                                                  tile_size,
                                                  stride):
        img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(
            img_tile, lab_tile, ps
        )
        if h_eff < ps or w_eff < ps:
            continue

        feats_tile = None
        hp = wp = None

        if feature_dir is not None and image_id is not None:
            fpath = tile_feature_path(feature_dir, image_id, y, x)
            if os.path.exists(fpath):
                feats_tile = np.load(fpath)
                hp, wp = feats_tile.shape[:2]
                cached_tiles += 1

        if feats_tile is None:
            feats_tile, hp, wp = extract_patch_features_single_scale(
                img_c,
                model,
                processor,
                device,
                ps=ps,
                aggregate_layers=aggregate_layers,
            )
            computed_tiles += 1

            if feature_dir is not None and image_id is not None:
                save_tile_features(
                    feats_tile,
                    feature_dir=feature_dir,
                    image_id=image_id,
                    y=y,
                    x=x,
                )

        pos_mask, neg_mask = labels_to_patch_masks(
            lab_c,
            hp,
            wp,
            pos_frac_thresh=pos_frac_thresh,
        )

        pos_feats_tile = feats_tile[pos_mask]
        neg_feats_tile = feats_tile[neg_mask]

        if pos_feats_tile.size > 0:
            pos_list.append(pos_feats_tile)
        if neg_feats_tile.size > 0:
            neg_list.append(neg_feats_tile)

    if not pos_list:
        raise ValueError(
            "no positive patches found in Image A; "
            "check labels or pos_frac_thresh"
        )

    pos_bank = np.concatenate(pos_list, axis=0)
    if neg_list:
        neg_bank = np.concatenate(neg_list, axis=0)
    else:
        neg_bank = None

    print(f"Positive bank size: {len(pos_bank)} patches")
    if neg_bank is not None:
        print(f"Negative bank size: {len(neg_bank)} patches")

        # Subsample negatives if too large (helps GPU memory / speed)
        if len(neg_bank) > MAX_NEG_BANK:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(neg_bank), size=MAX_NEG_BANK, replace=False)
            neg_bank_sub = neg_bank[idx]
            print(
                f"[info] subsampled negative bank from {len(neg_bank)} "
                f"to {len(neg_bank_sub)} patches (MAX_NEG_BANK={MAX_NEG_BANK})"
            )
            neg_bank = neg_bank_sub

    time_end("build_banks_single_scale", t0)
    print(
        f"[cache] A: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )

    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        pos_cache_path = os.path.join(bank_cache_dir, f"{image_id}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{image_id}_neg_bank.npy")
        np.save(pos_cache_path, pos_bank.astype(np.float32))
        if neg_bank is not None:
            np.save(neg_cache_path, neg_bank.astype(np.float32))
        print(f"[cache] saved banks to {bank_cache_dir}")

    return pos_bank, neg_bank


# ----------------------------------------------------------------------
# 5b. Optional prefetch of Image B features (single-scale, cached features)
# ----------------------------------------------------------------------

def prefetch_features_single_scale_image(
    img_hw3: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
):
    """
    Preload / compute all tile features for a given image once, so grid-search
    over multiple k values can reuse them without repeatedly hitting disk.

    We store a simple dict keyed by (y, x) with feature arrays and shapes.
    """
    t0 = time_start()
    cache = {}
    cached_tiles = 0
    computed_tiles = 0
    skipped_tiles = 0

    for y, x, img_tile, _ in tile_iterator(img_hw3, None, tile_size, stride):
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
        if h_eff < ps or w_eff < ps:
            skipped_tiles += 1
            continue

        feats_tile = None
        hp = wp = None

        if feature_dir is not None and image_id is not None:
            fpath = tile_feature_path(feature_dir, image_id, y, x)
            if os.path.exists(fpath):
                feats_tile = np.load(fpath)
                hp, wp = feats_tile.shape[:2]
                cached_tiles += 1

        if feats_tile is None:
            feats_tile, hp, wp = extract_patch_features_single_scale(
                img_c,
                model,
                processor,
                device,
                ps=ps,
                aggregate_layers=aggregate_layers,
            )
            computed_tiles += 1

            if feature_dir is not None and image_id is not None:
                save_tile_features(
                    feats_tile,
                    feature_dir=feature_dir,
                    image_id=image_id,
                    y=y,
                    x=x,
                )

        cache[(y, x)] = {
            "feats": feats_tile,
            "h_eff": h_eff,
            "w_eff": w_eff,
            "hp": hp,
            "wp": wp,
        }

    time_end("prefetch_features_single_scale_image", t0)
    print(
        "[prefetch] tiles="
        f"{len(cache)} (cached={cached_tiles}, computed={computed_tiles}, "
        f"skipped={skipped_tiles})"
    )
    return cache


# ----------------------------------------------------------------------
# 6. GPU kNN scoring on Image B (single-scale, cached features + optional prefetch)
#     Extended to use negative bank: score = pos_mean - alpha * neg_mean
# ----------------------------------------------------------------------

def zero_shot_knn_single_scale_B_with_saliency(
    img_b: np.ndarray,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    k: int = 5,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
    neg_alpha: float = 1.0,
    prefetched_tiles: dict | None = None,
    use_fp16_matmul: bool = False,
):
    """
    Compute kNN-based zero-shot scores on Image B using DINOv3 features.

    - For each patch feature in B, we compute similarity to the positive
      and (optionally) negative banks from A.
    - The score is mean(top-k positive sims) - alpha * mean(top-k negative sims).
    - We also derive a simple saliency estimate from the positive sims.
    """
    t0 = time_start()

    h_full, w_full = img_b.shape[:2]

    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    saliency_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    cached_tiles = 0
    computed_tiles = 0

    pos_bank_t = torch.from_numpy(pos_bank.astype(np.float32)).to(device)
    pos_bank_t_half = None
    if use_fp16_matmul and device.type == "cuda":
        pos_bank_t_half = pos_bank_t.half()
    k_pos_eff = min(k, pos_bank_t.shape[0])

    if neg_bank is not None:
        neg_bank_t = torch.from_numpy(neg_bank.astype(np.float32)).to(device)
        neg_bank_t_half = None
        if use_fp16_matmul and device.type == "cuda":
            neg_bank_t_half = neg_bank_t.half()
        k_neg_eff = min(k, neg_bank_t.shape[0])
        use_neg = True
        print(f"[info] zero_shot: using negative bank with size={neg_bank_t.shape[0]}, "
              f"k_neg_eff={k_neg_eff}, alpha={neg_alpha}")
    else:
        neg_bank_t = None
        neg_bank_t_half = None
        k_neg_eff = 0
        use_neg = False
        print("[info] zero_shot: negative bank disabled (neg_bank is None)")

    matmul_time = 0.0
    resize_time = 0.0

    if prefetched_tiles is not None:
        tile_iter = sorted(prefetched_tiles.items())
        print(f"[perf] zero_shot: using prefetched features for {len(tile_iter)} tiles")
    else:
        tile_iter = tile_iterator(img_b, None, tile_size, stride)

    for tile_entry in tile_iter:
        t0_tile = time_start()
        if prefetched_tiles is not None:
            (y, x), feat_info = tile_entry
            feats_tile = feat_info["feats"]
            h_eff = feat_info["h_eff"]
            w_eff = feat_info["w_eff"]
            hp = feat_info["hp"]
            wp = feat_info["wp"]
            cached_tiles += 1  # already on disk/memory
        else:
            y, x, img_tile, _ = tile_entry

            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(
                img_tile, None, ps
            )
            if h_eff < ps or w_eff < ps:
                time_end(f"zero_shot_tile_skip(y={y},x={x})", t0_tile)
                continue

            feats_tile = None
            hp = wp = None

            if feature_dir is not None and image_id is not None:
                fpath = tile_feature_path(feature_dir, image_id, y, x)
                if os.path.exists(fpath):
                    feats_tile = np.load(fpath)
                    hp, wp = feats_tile.shape[:2]
                    cached_tiles += 1

            if feats_tile is None:
                feats_tile, hp, wp = extract_patch_features_single_scale(
                    img_c,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
                computed_tiles += 1

                if feature_dir is not None and image_id is not None:
                    save_tile_features(
                        feats_tile,
                        feature_dir=feature_dir,
                        image_id=image_id,
                        y=y,
                        x=x,
                    )

        x_feats = feats_tile.reshape(-1, feats_tile.shape[-1]).astype(np.float32)

        with torch.no_grad():
            x_feats_t = torch.from_numpy(x_feats).to(device)      # (Nb, C)
            if use_fp16_matmul and device.type == "cuda":
                x_feats_t = x_feats_t.half()
                pos_bank_local = pos_bank_t_half
                neg_bank_local = neg_bank_t_half
            else:
                pos_bank_local = pos_bank_t
                neg_bank_local = neg_bank_t

            t_matmul0 = time.perf_counter() if DEBUG_TIMING else None

            # Positive sims
            sims_pos_full = x_feats_t @ pos_bank_local.t()        # (Nb, Npos)
            sims_pos_topk, _ = torch.topk(sims_pos_full, k=k_pos_eff, dim=1)
            score_pos = sims_pos_topk.mean(dim=1)                 # (Nb,)

            if use_neg:
                sims_neg_full = x_feats_t @ neg_bank_local.t()    # (Nb, Nneg)
                sims_neg_topk, _ = torch.topk(sims_neg_full, k=k_neg_eff, dim=1)
                score_neg = sims_neg_topk.mean(dim=1)             # (Nb,)
                score_batch = score_pos - neg_alpha * score_neg   # (Nb,)
            else:
                score_batch = score_pos                           # (Nb,)

            if DEBUG_TIMING and t_matmul0 is not None:
                matmul_time += time.perf_counter() - t_matmul0

        # score map
        score_patch = score_batch.cpu().numpy().reshape(hp, wp)

        # saliency
        sims_pos = sims_pos_topk.float().cpu().numpy()            # (Nb, k_pos_eff)
        weights = sims_pos / (sims_pos.sum(axis=1, keepdims=True) + 1e-8)
        saliency_vals = (weights * sims_pos).sum(axis=1)
        saliency_patch = saliency_vals.reshape(hp, wp)

        t_resize0 = time.perf_counter() if DEBUG_TIMING else None
        score_tile = resize(
            score_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        saliency_tile = resize(
            saliency_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        score_full[y:y + h_eff, x:x + w_eff] += score_tile
        saliency_full[y:y + h_eff, x:x + w_eff] += saliency_tile
        weight_full[y:y + h_eff, x:x + w_eff] += 1.0

        if DEBUG_TIMING and t_resize0 is not None:
            resize_time += time.perf_counter() - t_resize0

        time_end(f"zero_shot_tile(y={y},x={x},k={k})", t0_tile)

    mask_nonzero = weight_full > 0.0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    saliency_full[mask_nonzero] /= weight_full[mask_nonzero]

    time_end(f"zero_shot_knn_single_scale_B_with_saliency (GPU, k={k})", t0)
    print(
        f"[cache] B: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )
    if DEBUG_TIMING:
        print(
            f"[perf] k={k} matmul_time={matmul_time:.2f}s, "
            f"resize_time={resize_time:.2f}s"
        )

    return score_full, saliency_full


# ----------------------------------------------------------------------
# 7. SH_2022 clipping + metrics + oracle upper bound
# ----------------------------------------------------------------------

def build_sh_buffer_mask(labels_sh: np.ndarray,
                         buffer_pixels: int) -> np.ndarray:
    """
    Build a binary buffer mask around SH_2022 labels.

    - labels_sh: rasterized SH_2022 map (non-zero = SH).
    - buffer_pixels: radius in pixels for morphological dilation.
    """
    t0 = time_start()
    base = labels_sh > 0
    if buffer_pixels <= 0:
        time_end("build_sh_buffer_mask", t0)
        return base
    if _HAS_CV2:
        # OpenCV is significantly faster for large structuring elements
        ksize = 2 * buffer_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        buf = cv2.dilate(base.astype(np.uint8), kernel).astype(bool)
    else:
        buf = dilation(base.astype(bool), disk(buffer_pixels))
    time_end("build_sh_buffer_mask", t0)
    return buf


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute confusion matrix and derived statistics for binary masks.
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
    }
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("compute_metrics", t0)
    return metrics


def compute_metrics_batch_gpu(score_map: np.ndarray,
                              thresholds: list[float],
                              sh_mask: np.ndarray | None,
                              gt_mask: np.ndarray,
                              device: torch.device,
                              batch_size: int = 8) -> list[dict]:
    """
    Evaluate multiple thresholds in parallel on GPU to speed up grid search.

    We flatten the score and GT arrays and run a batched comparison
    for a list of thresholds, returning per-threshold metrics.
    """
    t0 = time_start()
    score_t = torch.from_numpy(score_map.astype(np.float32)).to(device).flatten()
    gt_t = torch.from_numpy(gt_mask.astype(np.bool_)).to(device).flatten()
    if sh_mask is not None:
        sh_t = torch.from_numpy(sh_mask.astype(np.bool_)).to(device).flatten()
    else:
        sh_t = None

    metrics = []
    eps = 1e-8

    for start in range(0, len(thresholds), batch_size):
        thr_chunk = thresholds[start:start + batch_size]
        thr_t = torch.tensor(thr_chunk, device=device, dtype=torch.float32).view(-1, 1)

        mask_thr = score_t.unsqueeze(0) >= thr_t  # (T, N)
        if sh_t is not None:
            mask_thr = mask_thr & sh_t

        tp = (mask_thr & gt_t).sum(dim=1).float()
        fp = (mask_thr & (~gt_t)).sum(dim=1).float()
        fn = ((~mask_thr) & gt_t).sum(dim=1).float()
        tn = ((~mask_thr) & (~gt_t)).sum(dim=1).float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        for i, thr in enumerate(thr_chunk):
            metrics.append({
                "threshold": float(thr),
                "tp": int(tp[i].item()),
                "fp": int(fp[i].item()),
                "fn": int(fn[i].item()),
                "tn": int(tn[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "iou": float(iou[i].item()),
                "f1": float(f1[i].item()),
            })

    time_end("compute_metrics_batch_gpu", t0)
    return metrics


def compute_metrics_batch_cpu(score_map: np.ndarray,
                              thresholds: list[float],
                              sh_mask: np.ndarray | None,
                              gt_mask: np.ndarray,
                              batch_size: int = 16) -> list[dict]:
    """
    CPU batched threshold evaluation to reduce Python overhead; chunked
    to limit RAM usage.
    """
    t0 = time_start()
    flat_scores = score_map.astype(np.float32).reshape(1, -1)  # (1, N)
    flat_gt = gt_mask.astype(bool).reshape(1, -1)
    if sh_mask is not None:
        flat_sh = sh_mask.astype(bool).reshape(1, -1)
    else:
        flat_sh = None

    metrics = []
    eps = 1e-8

    for start in range(0, len(thresholds), batch_size):
        thr_chunk = np.array(thresholds[start:start + batch_size], dtype=np.float32).reshape(-1, 1)
        mask = flat_scores >= thr_chunk  # (B, N)
        if flat_sh is not None:
            mask = np.logical_and(mask, flat_sh)

        tp = np.logical_and(mask, flat_gt).sum(axis=1).astype(np.float64)
        fp = np.logical_and(mask, ~flat_gt).sum(axis=1).astype(np.float64)
        fn = np.logical_and(~mask, flat_gt).sum(axis=1).astype(np.float64)
        tn = np.logical_and(~mask, ~flat_gt).sum(axis=1).astype(np.float64)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        for i, thr in enumerate(thr_chunk[:, 0]):
            metrics.append({
                "threshold": float(thr),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "iou": float(iou[i]),
                "f1": float(f1[i]),
            })

    time_end("compute_metrics_batch_cpu", t0)
    return metrics


def compute_oracle_upper_bound(gt_mask: np.ndarray,
                               sh_mask: np.ndarray) -> dict:
    """
    Oracle IoU if you are not allowed to predict outside SH buffer:
    IoU( gt ∧ buffer , gt ).
    """
    t0 = time_start()
    oracle_mask = np.logical_and(gt_mask.astype(bool), sh_mask.astype(bool))
    metrics = compute_metrics(oracle_mask, gt_mask)
    print(
        "[oracle] SH buffer upper bound -> "
        f"IoU={metrics['iou']:.3f}, F1={metrics['f1']:.3f}, "
        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}"
    )
    time_end("oracle_upper_bound_SH_buffer", t0)
    return metrics


# ----------------------------------------------------------------------
# 9. DenseCRF refinement (logistic unary centred at threshold)
# ----------------------------------------------------------------------

def refine_with_densecrf(
    img_rgb: np.ndarray,
    score_map: np.ndarray,
    threshold_center: float,
    sh_mask: np.ndarray | None = None,
    prob_softness: float = 0.05,
    n_iters: int = 5,
    pos_w: float = 3.0,
    pos_xy_std: float = 3.0,
    bilateral_w: float = 5.0,
    bilateral_xy_std: float = 50.0,
    bilateral_rgb_std: float = 5.0,
) -> np.ndarray:
    """
    DenseCRF refinement for binary segmentation.

    - We convert the score_map to a foreground probability via a logistic
      centered at `threshold_center`.
    - SH_2022 buffer is encoded by down-weighting foreground outside.
    - CRF pairwise terms encourage spatial smoothness and color-aligned
      boundaries, improving coherence along LWFs.
    """
    t0 = time_start()
    h, w, _ = img_rgb.shape
    assert score_map.shape == (h, w), "score_map must have shape (H, W)"

    # --- 1) turn score_map into foreground probability -----------------
    s = score_map.astype(np.float32)

    # Logistic centred at threshold_center
    logits_fg = (s - threshold_center) / prob_softness
    p_fg = 1.0 / (1.0 + np.exp(-logits_fg))

    eps = 1e-6
    p_fg = np.clip(p_fg, eps, 1.0 - eps)

    # Respect SH_2022 buffer: outside buffer, force near-zero FG prob
    if sh_mask is not None:
        sh_mask = sh_mask.astype(bool)
        p_fg[~sh_mask] = eps  # almost certainly background

    p_bg = 1.0 - p_fg

    # Shape for unary_from_softmax: (C, H, W)
    probs = np.stack([p_bg, p_fg], axis=0)

    # --- 2) build DenseCRF2D model -------------------------------------
    d = dcrf.DenseCRF2D(w, h, 2)

    # Unary energy from probabilities
    unary = unary_from_softmax(probs)  # shape (2, H*W)
    d.setUnaryEnergy(unary)

    # --- 3) add pairwise terms -----------------------------------------
    # (a) spatial smoothness (XY only)
    d.addPairwiseGaussian(
        sxy=(pos_xy_std, pos_xy_std),
        compat=pos_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # (b) bilateral term in XY + RGB (aligns with color edges)
    if img_rgb.dtype != np.uint8:
        img_rgb_u8 = img_rgb.astype(np.uint8)
    else:
        img_rgb_u8 = img_rgb

    if not img_rgb_u8.flags["C_CONTIGUOUS"]:
        img_rgb_u8 = np.ascontiguousarray(img_rgb_u8)

    d.addPairwiseBilateral(
        sxy=(bilateral_xy_std, bilateral_xy_std),
        srgb=(bilateral_rgb_std, bilateral_rgb_std, bilateral_rgb_std),
        rgbim=img_rgb_u8,
        compat=bilateral_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # --- 4) run mean-field inference -----------------------------------
    Q = d.inference(n_iters)  # list of length 2, each H*W
    Q = np.array(Q).reshape(2, h, w)
    labels = np.argmax(Q, axis=0).astype(np.uint8)  # 0=bg, 1=fg

    refined_mask = labels == 1

    # (Optional) enforce SH buffer as a hard constraint as well
    if sh_mask is not None:
        refined_mask = np.logical_and(refined_mask, sh_mask)

    time_end("refine_with_densecrf", t0)
    return refined_mask


def _crf_eval_worker(args):
    """
    Helper for process-based CRF evaluation to leverage multiple CPU cores.
    Returns metrics and the config; mask is recomputed for the best config
    later at full resolution.
    """
    (img_rgb_ds,
     score_map_ds,
     sh_mask_ds,
     gt_mask_ds,
     threshold_center,
     n_iters,
     cfg) = args

    prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
    mask_crf_local = refine_with_densecrf(
        img_rgb_ds,
        score_map_ds,
        threshold_center,
        sh_mask_ds,
        prob_softness=prob_soft,
        n_iters=n_iters,
        pos_w=pos_w,
        pos_xy_std=pos_xy,
        bilateral_w=bi_w,
        bilateral_xy_std=bi_xy,
        bilateral_rgb_std=bi_rgb,
    )
    metrics_local = compute_metrics(mask_crf_local, gt_mask_ds)
    return metrics_local, {
        "prob_softness": prob_soft,
        "pos_w": pos_w,
        "pos_xy_std": pos_xy,
        "bilateral_w": bi_w,
        "bilateral_xy_std": bi_xy,
        "bilateral_rgb_std": bi_rgb,
        **metrics_local,
    }


def crf_grid_search(
    img_rgb: np.ndarray,
    score_map: np.ndarray,
    threshold_center: float,
    sh_mask: np.ndarray,
    gt_mask: np.ndarray,
    prob_softness_vals,
    pos_w_vals,
    pos_xy_std_vals,
    bilateral_w_vals,
    bilateral_xy_std_vals,
    bilateral_rgb_std_vals,
    n_iters: int = 5,
    max_configs: int | None = None,
    downsample_factor: int = 1,
    num_workers: int = 1,
    backend: str = "process",  # "process" or "thread"
):
    """
    Small grid search over CRF hyperparameters for a fixed (k, thr)
    champion configuration.

    We optionally downsample data for a coarse search and then recompute
    the best configuration on the same resolution.
    """
    t0 = time_start()
    best_cfg = None
    best_mask = None
    best_iou = -1.0

    if downsample_factor > 1:
        # Coarse search on downsampled data for speed
        img_rgb_ds = resize(
            img_rgb,
            (img_rgb.shape[0] // downsample_factor, img_rgb.shape[1] // downsample_factor),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(img_rgb.dtype)
        score_map_ds = resize(
            score_map,
            (score_map.shape[0] // downsample_factor, score_map.shape[1] // downsample_factor),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        sh_mask_ds = resize(
            sh_mask.astype(np.float32),
            (sh_mask.shape[0] // downsample_factor, sh_mask.shape[1] // downsample_factor),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
        gt_mask_ds = resize(
            gt_mask.astype(np.float32),
            (gt_mask.shape[0] // downsample_factor, gt_mask.shape[1] // downsample_factor),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
    else:
        img_rgb_ds = img_rgb
        score_map_ds = score_map
        sh_mask_ds = sh_mask
        gt_mask_ds = gt_mask

    cfg_list = []
    for prob_soft in prob_softness_vals:
        for pos_w in pos_w_vals:
            for pos_xy in pos_xy_std_vals:
                for bi_w in bilateral_w_vals:
                    for bi_xy in bilateral_xy_std_vals:
                        for bi_rgb in bilateral_rgb_std_vals:
                            cfg_list.append((prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb))

    if max_configs is not None:
        cfg_list = cfg_list[:max_configs]

    if num_workers > 1 and backend == "process":
        # Process-based parallelism to utilize multiple CPU cores
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            args_iter = [
                (
                    img_rgb_ds,
                    score_map_ds,
                    sh_mask_ds,
                    gt_mask_ds,
                    threshold_center,
                    n_iters,
                    cfg,
                )
                for cfg in cfg_list
            ]
            for metrics, cfg_full in ex.map(_crf_eval_worker, args_iter):
                if metrics["iou"] > best_iou:
                    best_iou = metrics["iou"]
                    best_cfg = cfg_full
    else:
        # Threaded or single-thread fallback
        for cfg in cfg_list:
            prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
            print(
                "[crf] evaluating config: "
                f"soft={prob_soft}, pos_w={pos_w}, pos_xy={pos_xy}, "
                f"bi_w={bi_w}, bi_xy={bi_xy}, bi_rgb={bi_rgb}"
            )
            t_cfg = time_start()
            mask_crf_local = refine_with_densecrf(
                img_rgb=img_rgb_ds,
                score_map=score_map_ds,
                threshold_center=threshold_center,
                sh_mask=sh_mask_ds,
                prob_softness=prob_soft,
                n_iters=n_iters,
                pos_w=pos_w,
                pos_xy_std=pos_xy,
                bilateral_w=bi_w,
                bilateral_xy_std=bi_xy,
                bilateral_rgb_std=bi_rgb,
            )
            time_end("crf_single_config", t_cfg)

            metrics_local = compute_metrics(mask_crf_local, gt_mask_ds)
            print(
                f"[crf-eval] soft={prob_soft}, pos_w={pos_w}, pos_xy={pos_xy}, "
                f"bi_w={bi_w}, bi_xy={bi_xy}, bi_rgb={bi_rgb} -> "
                f"IoU={metrics_local['iou']:.3f}, F1={metrics_local['f1']:.3f}, "
                f"P={metrics_local['precision']:.3f}, R={metrics_local['recall']:.3f}"
            )
            if metrics_local["iou"] > best_iou:
                best_iou = metrics_local["iou"]
                best_cfg = {
                    "prob_softness": prob_soft,
                    "pos_w": pos_w,
                    "pos_xy_std": pos_xy,
                    "bilateral_w": bi_w,
                    "bilateral_xy_std": bi_xy,
                    "bilateral_rgb_std": bi_rgb,
                    **metrics_local,
                }
                best_mask = mask_crf_local

    # Recompute best mask once (to avoid passing large masks between processes)
    if best_cfg is not None and best_mask is None:
        best_mask = refine_with_densecrf(
            img_rgb=img_rgb_ds,
            score_map=score_map_ds,
            threshold_center=threshold_center,
            sh_mask=sh_mask_ds,
            prob_softness=best_cfg["prob_softness"],
            n_iters=n_iters,
            pos_w=best_cfg["pos_w"],
            pos_xy_std=best_cfg["pos_xy_std"],
            bilateral_w=best_cfg["bilateral_w"],
            bilateral_xy_std=best_cfg["bilateral_xy_std"],
            bilateral_rgb_std=best_cfg["bilateral_rgb_std"],
        )

    time_end("crf_grid_search", t0)
    print("\n[crf] best CRF configuration:")
    print(best_cfg)
    return best_cfg, best_mask


# ----------------------------------------------------------------------
# 10. Grid search over k and threshold (raw only)
#     + fine threshold tuning (raw only)
# ----------------------------------------------------------------------

def fine_tune_threshold(
    score_map: np.ndarray,
    base_threshold: float,
    sh_mask: np.ndarray | None,
    gt_mask: np.ndarray,
    step: float = 0.01,
    window: float = 0.08,
):
    """
    Fine-tune a scalar threshold around a coarse optimum using only the raw
    score map (no superpixels).

    - Search in [base_threshold - window, base_threshold + window], clamped
      to [0, 1].
    - Clip predictions by SH_2022 buffer if provided.
    - Keep the threshold that maximizes IoU on gt_mask.
    """
    t0 = time_start()

    # Clamp search interval to the valid score range [0, 1]
    thr_min = max(0.0, base_threshold - window)
    thr_max = min(1.0, base_threshold + window)

    thr_vals = np.arange(thr_min, thr_max + 1e-8, step)

    best_thr = base_threshold
    best_metrics = None
    best_mask = None
    best_iou = -1.0

    for thr in thr_vals:
        # Simple per-pixel thresholding
        mask = score_map >= thr

        # Enforce SH_2022 buffer if given
        if sh_mask is not None:
            mask = np.logical_and(mask, sh_mask)

        # Evaluate IoU/F1 etc.
        metrics = compute_metrics(mask, gt_mask)

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thr = thr
            best_metrics = metrics
            best_mask = mask

    print(
        f"[tune-thr] base={base_threshold:.3f} -> "
        f"best={best_thr:.3f} IoU={best_metrics['iou']:.3f}, "
        f"F1={best_metrics['f1']:.3f}"
    )

    time_end("fine_tune_threshold", t0)
    return best_thr, best_metrics, best_mask


def grid_search_k_threshold(
    img_b: np.ndarray,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int | None,
    k_values: list[int],
    thresholds: list[float],
    feature_dir: str,
    image_id_b: str,
    sh_buffer_mask_b: np.ndarray,
    gt_mask_b: np.ndarray,
    prefetched_tiles_b: dict | None = None,
    use_fp16_matmul: bool = False,
):
    """
    Grid search over (k, threshold) using only raw per-pixel kNN scores.

    - For each k, compute a score map and evaluate many thresholds.
    - SH_2022 buffer is used to clip predictions for each threshold.
    - The best configuration is chosen by IoU on Image B.
    """
    t0 = time_start()

    best_raw_config = None
    best_raw_score_full = None
    best_raw_saliency_full = None
    best_raw_iou = -1.0

    for k in k_values:
        t0_k_total = time_start()

        # 1) Compute raw kNN scores for this k
        t0_k_score = time_start()
        score_full, saliency_full = zero_shot_knn_single_scale_B_with_saliency(
            img_b=img_b,
            pos_bank=pos_bank,
            neg_bank=neg_bank,
            model=model,
            processor=processor,
            device=device,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            k=k,
            aggregate_layers=None,
            feature_dir=feature_dir,
            image_id=image_id_b,
            neg_alpha=NEG_ALPHA,
            prefetched_tiles=prefetched_tiles_b,
            use_fp16_matmul=use_fp16_matmul,
        )
        time_end(f"grid_search_score_full(k={k})", t0_k_score)

        # 2) Evaluate multiple thresholds in batch
        metrics_raw_list = None

        if USE_GPU_THRESHOLD_METRICS and device.type == "cuda":
            try:
                metrics_raw_list = compute_metrics_batch_gpu(
                    score_map=score_full,
                    thresholds=thresholds,
                    sh_mask=sh_buffer_mask_b,
                    gt_mask=gt_mask_b,
                    device=device,
                    batch_size=THRESHOLD_BATCH_SIZE,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                metrics_raw_list = None
                print("[warn] OOM during GPU threshold metrics; falling back to CPU")

        if metrics_raw_list is None:
            metrics_raw_list = compute_metrics_batch_cpu(
                score_map=score_full,
                thresholds=thresholds,
                sh_mask=sh_buffer_mask_b,
                gt_mask=gt_mask_b,
                batch_size=THRESHOLD_CPU_BATCH_SIZE,
            )

        # 3) Track best (k, thr) based on IoU
        for metrics_raw in metrics_raw_list:
            thr = metrics_raw["threshold"]
            iou_raw = metrics_raw["iou"]
            f1_raw = metrics_raw["f1"]

            print(
                f"[eval-raw] k={k}, thr={thr:.3f} -> "
                f"IoU={iou_raw:.3f}, F1={f1_raw:.3f}, "
                f"P={metrics_raw['precision']:.3f}, "
                f"R={metrics_raw['recall']:.3f}"
            )

            if iou_raw > best_raw_iou:
                best_raw_iou = iou_raw
                best_raw_config = {
                    "k": k,
                    "threshold": thr,
                    "source": "raw",
                    **metrics_raw,
                }
                best_raw_score_full = score_full.copy()
                best_raw_saliency_full = saliency_full.copy()

        time_end(f"k_loop_total(k={k})", t0_k_total)

    print("\n[best-raw] configuration:")
    print(best_raw_config)

    time_end("grid_search_k_threshold", t0)

    return best_raw_config, best_raw_score_full, best_raw_saliency_full


# ----------------------------------------------------------------------
# 11. shapefile export
# ----------------------------------------------------------------------

def export_mask_to_shapefile(mask: np.ndarray,
                             ref_raster_path: str,
                             out_path: str):
    """
    Export a binary mask as a polygon shapefile in the CRS of the
    reference raster.

    - We use rasterio.features.shapes to vectorize the mask.
    - Only pixels with value 1 are exported.
    """
    t0 = time_start()

    mask_uint8 = mask.astype("uint8")

    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        crs = src.crs

    shape_generator = rfeatures.shapes(mask_uint8,
                                       mask=mask_uint8 == 1,
                                       transform=transform)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with fiona.open(
        out_path,
        mode="w",
        driver="ESRI Shapefile",
        crs=crs.to_dict() if crs is not None else None,
        schema=schema,
    ) as shp:
        idx = 0
        for geom, value in shape_generator:
            if value != 1:
                continue
            shp.write({
                "geometry": geom,
                "properties": {
                    "id": int(idx),
                },
            })
            idx += 1

    time_end("export_mask_to_shapefile", t0)
    print(f"[info] shapefile written to: {out_path}")


# ----------------------------------------------------------------------
# 12. script entry point
# ----------------------------------------------------------------------

def main():
    """
    End-to-end orchestration for single-scale DINOv3 zero-shot transfer:

    - Load images and labels.
    - Build feature banks on Image A.
    - Run kNN on Image B with a grid over k and thresholds.
    - Fine-tune threshold around the best coarse optimum (raw only).
    - Run CRF grid search around the champion config.
    - Visualize results and export shapefiles for raw-best and CRF-best.
    """
    t0_main = time_start()

    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    model, processor, device = init_model(model_name)

    # ------------ paths ------------
    img_path = (
        "data/dop20_593000_5979000_1km_20cm.tif"
    )  # Image A
    img2_path = (
        "data/dop20_592000_5982000_1km_20cm.tif"
    )  # Image B
    lab_path = (
        "data/planet_labels_2022.tif"
    )  # SH_2022 raster
    gt_vector_path = (
        "data/labels_final.shp"
    )  # ground truth for 592000 tile

    # ------------ data loading A/B ------------
    t0_data = time_start()

    img = load_dop20_image(img_path)
    labels_A = reproject_labels_to_image(img_path, lab_path)

    img_b = load_dop20_image(img2_path)
    labels_SH_B = reproject_labels_to_image(img2_path, lab_path)
    gt_mask_B = rasterize_vector_labels(gt_vector_path, img2_path)

    time_end("data_loading_and_reprojection", t0_data)

    # quick sanity check
    print(f"[debug] GT positives on B: {gt_mask_B.sum()}")
    print(f"[debug] SH_2022 positives on B: {(labels_SH_B > 0).sum()}")

    # ------------ pixel size for buffer ------------
    with rasterio.open(img2_path) as src:
        pixel_size_m = abs(src.transform.a)

    # Buffer around SH_2022 (in meters)
    buffer_m = 8.0
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    print(
        f"[info] pixel_size={pixel_size_m:.3f} m, "
        f"buffer_m={buffer_m}, buffer_pixels={buffer_pixels}"
    )

    # ------------ feature cache folder ------------
    feature_dir = os.path.join(
        os.path.dirname(img_path),
        "dino_features",
    )

    image_id_a = os.path.splitext(os.path.basename(img_path))[0]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    # ------------ build bank on A ------------
    pos_bank, neg_bank = build_banks_single_scale(
        img_a=img,
        labels_a=labels_A,
        model=model,
        processor=processor,
        device=device,
        ps=model.config.patch_size,
        tile_size=1024,
        stride=512,
        pos_frac_thresh=0.1,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=image_id_a,
        bank_cache_dir=os.path.join(feature_dir, "banks"),
    )

    # ------------ SH_2022 buffer mask for B ------------
    sh_buffer_mask_B = build_sh_buffer_mask(labels_SH_B, buffer_pixels)

    # ------------ oracle upper bound (only inside SH buffer allowed) ------------
    _ = compute_oracle_upper_bound(gt_mask_B, sh_buffer_mask_B)

    # ------------ prefetch Image B features once for all k ------------
    prefetched_b = prefetch_features_single_scale_image(
        img_hw3=img_b,
        model=model,
        processor=processor,
        device=device,
        ps=model.config.patch_size,
        tile_size=1024,
        stride=512,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=image_id_b,
    )

    # ------------ grid search on B (raw only) ------------
    K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100, 150, 200, 300, 500]
    THRESHOLDS = np.linspace(0.01, 0.9, 50).tolist()

    best_raw_config, best_raw_score_full, best_raw_saliency_full = grid_search_k_threshold(
        img_b=img_b,
        pos_bank=pos_bank,
        neg_bank=neg_bank,
        model=model,
        processor=processor,
        device=device,
        ps=model.config.patch_size,
        tile_size=1024,
        stride=512,
        k_values=K_VALUES,
        thresholds=THRESHOLDS,
        feature_dir=feature_dir,
        image_id_b=image_id_b,
        sh_buffer_mask_b=sh_buffer_mask_B,
        gt_mask_b=gt_mask_B,
        prefetched_tiles_b=prefetched_b,
        use_fp16_matmul=USE_FP16_KNN,
    )

    # ------------ fine-tune threshold around coarse optimum (raw only) ------------
    thr_best_raw_refined, metrics_raw_refined, mask_raw_best = fine_tune_threshold(
        score_map=best_raw_score_full,
        base_threshold=best_raw_config["threshold"],
        sh_mask=sh_buffer_mask_B,
        gt_mask=gt_mask_B,
    )

    # Update config if refinement improved IoU
    if metrics_raw_refined["iou"] >= best_raw_config["iou"]:
        best_raw_config = {
            **best_raw_config,
            "threshold": thr_best_raw_refined,
            **metrics_raw_refined,
        }
    else:
        # Use the coarse threshold if refinement did not help
        mask_raw_best = best_raw_score_full >= best_raw_config["threshold"]
        mask_raw_best = np.logical_and(mask_raw_best, sh_buffer_mask_B)

    # ------------ choose champion (raw) for CRF calibration ------------
    champion_config = best_raw_config
    champion_score_full = best_raw_score_full

    thr_center_for_crf = champion_config["threshold"]
    k_center_for_crf = champion_config["k"]

    print("\n[crf] using champion config for unary centre:")
    print(f"      source={champion_config['source']}, "
          f"k={k_center_for_crf}, thr_center={thr_center_for_crf:.3f}")

    # ------------ CRF hyperparameter search for champion (single k) ------------
    print("\n[crf] starting CRF hyperparameter search (single k)")

    PROB_SOFTNESS_VALUES = [0.03, 0.05, 0.08]
    POS_W_VALUES = [3.0, 4.0]
    POS_XY_STD_VALUES = [3.0]
    BILATERAL_W_VALUES = [5.0, 7.0]
    BILATERAL_XY_STD_VALUES = [25.0, 50.0]
    BILATERAL_RGB_STD_VALUES = [3.0, 5.0]

    best_crf_cfg_inner, best_crf_mask = crf_grid_search(
        img_rgb=img_b,
        score_map=champion_score_full,
        threshold_center=thr_center_for_crf,
        sh_mask=sh_buffer_mask_B,
        gt_mask=gt_mask_B,
        prob_softness_vals=PROB_SOFTNESS_VALUES,
        pos_w_vals=POS_W_VALUES,
        pos_xy_std_vals=POS_XY_STD_VALUES,
        bilateral_w_vals=BILATERAL_W_VALUES,
        bilateral_xy_std_vals=BILATERAL_XY_STD_VALUES,
        bilateral_rgb_std_vals=BILATERAL_RGB_STD_VALUES,
        n_iters=5,
        max_configs=CRF_MAX_CONFIGS,
        downsample_factor=2,
        num_workers=16,
        backend="process",
    )

    best_crf_config = {"k": k_center_for_crf, **best_crf_cfg_inner}
    print("\n[crf] best CRF configuration with k:")
    print(best_crf_config)

    # If CRF was run on downsampled data, upsample mask back to full res
    if best_crf_mask.shape != img_b.shape[:2]:
        best_crf_mask_full = resize(
            best_crf_mask.astype(np.float32),
            (img_b.shape[0], img_b.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
        print(
            f"[crf] upsampled best CRF mask from {best_crf_mask.shape} "
            f"to {best_crf_mask_full.shape}"
        )
        best_crf_mask = best_crf_mask_full
        metrics_crf_full = compute_metrics(best_crf_mask, gt_mask_B)
        print(
            f"[crf-upsampled] IoU={metrics_crf_full['iou']:.3f}, "
            f"F1={metrics_crf_full['f1']:.3f}, "
            f"P={metrics_crf_full['precision']:.3f}, "
            f"R={metrics_crf_full['recall']:.3f}"
        )

    # ------------ visualization: raw vs CRF ------------
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Row 0: inputs
    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("Image B (RGB)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt_mask_B > 0, cmap="gray")
    axs[0, 1].set_title("Ground truth (labels_final)")
    axs[0, 1].axis("off")

    # Row 1: raw best vs CRF best
    overlay_raw = img_b.copy()
    overlay_mask_raw = mask_raw_best
    overlay_raw[overlay_mask_raw] = (
        0.5 * overlay_raw[overlay_mask_raw] + 0.5 * np.array([0, 255, 0])
    ).astype(overlay_raw.dtype)
    axs[1, 0].imshow(overlay_raw)
    axs[1, 0].set_title(
        f"Raw kNN (k={best_raw_config['k']}, thr={best_raw_config['threshold']:.3f})\n"
        f"IoU={best_raw_config['iou']:.3f}, F1={best_raw_config['f1']:.3f}"
    )
    axs[1, 0].axis("off")

    overlay_crf = img_b.copy()
    overlay_mask_crf = best_crf_mask
    overlay_crf[overlay_mask_crf] = (
        0.5 * overlay_crf[overlay_mask_crf] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay_crf.dtype)
    axs[1, 1].imshow(overlay_crf)
    axs[1, 1].set_title(
        f"CRF (k={best_crf_config['k']}, center_thr={thr_center_for_crf:.3f})\n"
        f"IoU={best_crf_config['iou']:.3f}, F1={best_crf_config['f1']:.3f}"
    )
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # ------------ shapefile export for best masks ------------
    base_name_b = os.path.splitext(os.path.basename(img2_path))[0]
    out_dir_b = os.path.dirname(img2_path)

    # Raw best
    shp_path_raw = os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_raw.shp")
    export_mask_to_shapefile(
        mask=mask_raw_best,
        ref_raster_path=img2_path,
        out_path=shp_path_raw,
    )

    # CRF best
    shp_path_crf = os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_crf.shp")
    export_mask_to_shapefile(
        mask=best_crf_mask,
        ref_raster_path=img2_path,
        out_path=shp_path_crf,
    )

    # ------------ consolidate features per .tif (A and B) ------------
    consolidate_features_for_image(feature_dir, image_id_a)
    consolidate_features_for_image(feature_dir, image_id_b)

    time_end("main (total)", t0_main)

    return {
        "best_raw_config": best_raw_config,
        "best_raw_score_full": best_raw_score_full,
        "best_raw_saliency_full": best_raw_saliency_full,
        "best_crf_config": best_crf_config,
        "best_crf_score_full": champion_score_full,
    }


if __name__ == "__main__":
    main()

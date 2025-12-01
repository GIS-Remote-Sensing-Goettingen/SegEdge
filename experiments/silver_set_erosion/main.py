# SPDX-License-Identifier: GPL-2.0
#
# Single-scale zero-shot transfer from Image A -> Image B using DINOv3.
# - GPU kNN (torch matmul + top-k)
# - Use SH_2022 labels to clip predictions on B (+buffer in meters)
# - Validate vs ground truth shapefile labels_final.shp
# - Grid search over k and threshold to pick best parameters.
# - Superpixel refinement (SLIC + mean-score or majority vote) with its own
#   validation threshold search.
# - DenseCRF refinement with unary calibrated around the best validation
#   threshold instead of min-max.

import time
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from skimage.morphology import erosion, disk, dilation
from skimage.segmentation import slic

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject, Resampling
import rasterio.features as rfeatures
from rasterio.crs import CRS

import fiona
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from transformers import AutoImageProcessor, AutoModel
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


# ----------------------------------------------------------------------
# 0. timing utilities
# ----------------------------------------------------------------------

DEBUG_TIMING = True


def time_start():
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt:.3f} s")


# ----------------------------------------------------------------------
# 1. model setup
# ----------------------------------------------------------------------


def init_model(model_name: str):
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
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    img = reshape_as_image(arr)  # (H, W, C)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def reproject_labels_to_image(ref_img_path: str, labels_path: str) -> np.ndarray:
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
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    return feats / norms


def tile_iterator(image_hw3: np.ndarray,
                  labels_hw: np.ndarray | None = None,
                  tile_size: int = 1024,
                  stride: int | None = None):
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
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps

    img_c = img_tile_hw3[:h_eff, :w_eff]
    if labels_tile_hw is not None:
        lab_c = labels_tile_hw[:h_eff, :w_eff]
    else:
        lab_c = None

    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(labels_tile: np.ndarray,
                          hp: int,
                          wp: int,
                          pos_frac_thresh: float = 0.1):
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp

    labels_c = labels_tile[:hp * patch_h, :wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)

    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))  # (Hp, Wp)

    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0

    return pos_mask, neg_mask


def tile_feature_path(feature_dir: str,
                      image_id: str,
                      y: int,
                      x: int) -> str:
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def save_tile_features(feats_tile: np.ndarray,
                       feature_dir: str,
                       image_id: str,
                       y: int,
                       x: int):
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))


def consolidate_features_for_image(feature_dir: str,
                                   image_id: str,
                                   output_suffix: str = "_features_full.npy"):
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
                             image_id: str | None = None):
    t0 = time_start()

    pos_list = []
    neg_list = []

    cached_tiles = 0
    computed_tiles = 0

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

    time_end("build_banks_single_scale", t0)
    print(
        f"[cache] A: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )

    return pos_bank, neg_bank


# ----------------------------------------------------------------------
# 6. GPU kNN scoring on Image B (single-scale, cached features)
# ----------------------------------------------------------------------


def zero_shot_knn_single_scale_B_with_saliency(
    img_b: np.ndarray,
    pos_bank: np.ndarray,
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
):
    t0 = time_start()

    h_full, w_full = img_b.shape[:2]

    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    saliency_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    cached_tiles = 0
    computed_tiles = 0

    pos_bank_t = torch.from_numpy(pos_bank.astype(np.float32)).to(device)
    k_eff = min(k, pos_bank_t.shape[0])

    for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride):
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(
            img_tile, None, ps
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

        x_feats = feats_tile.reshape(-1, feats_tile.shape[-1]).astype(np.float32)

        with torch.no_grad():
            x_feats_t = torch.from_numpy(x_feats).to(device)      # (Nb, C)
            sims_full = x_feats_t @ pos_bank_t.t()                # (Nb, Npos)
            sims_topk, _ = torch.topk(sims_full, k=k_eff, dim=1)
            sims = sims_topk.cpu().numpy()                        # (Nb, k_eff)

        score_patch = sims.mean(axis=1).reshape(hp, wp)

        weights = sims / (sims.sum(axis=1, keepdims=True) + 1e-8)
        saliency_vals = (weights * sims).sum(axis=1)
        saliency_patch = saliency_vals.reshape(hp, wp)

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

    mask_nonzero = weight_full > 0.0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    saliency_full[mask_nonzero] /= weight_full[mask_nonzero]

    time_end(f"zero_shot_knn_single_scale_B_with_saliency (GPU, k={k})", t0)
    print(
        f"[cache] B: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )

    return score_full, saliency_full


# ----------------------------------------------------------------------
# 7. SH_2022 clipping + metrics
# ----------------------------------------------------------------------


def build_sh_buffer_mask(labels_sh: np.ndarray,
                         buffer_pixels: int) -> np.ndarray:
    base = labels_sh > 0
    if buffer_pixels <= 0:
        return base
    buf = dilation(base.astype(bool), disk(buffer_pixels))
    return buf


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
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

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
    }


# ----------------------------------------------------------------------
# 8. Superpixel refinement utilities
# ----------------------------------------------------------------------


def compute_slic_superpixels(
    img_rgb: np.ndarray,
    n_segments: int = 12000,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute SLIC superpixels once and reuse them.

    Returns:
        segments: HxW int array with labels in [0, n_sp-1].
    """
    h, w, _ = img_rgb.shape
    img_float = img_rgb.astype(np.float32) / 255.0

    t0 = time_start()
    segments = slic(
        img_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    time_end("compute_slic_superpixels", t0)

    assert segments.shape == (h, w)
    return segments


def refine_with_superpixels(
    img_rgb: np.ndarray,
    score_map: np.ndarray,
    threshold: float,
    sh_mask: np.ndarray | None = None,
    segments: np.ndarray | None = None,
    n_segments: int = 12000,
    compactness: float = 10.0,
    sigma: float = 1.0,
    mode: str = "mean_score",   # "mean_score" or "majority"
    sp_fg_frac_thresh: float = 0.5,
) -> np.ndarray:
    """
    Superpixel-based refinement for binary segmentation.

    Args
    ----
    img_rgb      : HxWx3 uint8/float image (DOP20 tile B).
    score_map    : HxW float map (kNN score).
    threshold    : scalar threshold used to binarize scores.
    sh_mask      : optional HxW bool mask (e.g. SH_2022 buffer).
                   If given, we clip the final mask with it.
    segments     : precomputed SLIC labels (HxW). If None, compute here.
    n_segments   : SLIC number of superpixels (if segments is None).
    compactness  : SLIC compactness parameter (if segments is None).
    sigma        : SLIC smoothing parameter (if segments is None).
    mode         : "mean_score" -> average score per superpixel
                   "majority"   -> majority vote of (score >= thr)
    sp_fg_frac_thresh : for "majority" mode, fraction of foreground
                        pixels in a superpixel needed to mark it FG.

    Returns
    -------
    sp_mask      : HxW boolean array (True = foreground).
    """
    h, w = score_map.shape
    assert img_rgb.shape[:2] == (h, w), "img_rgb and score_map must match in HxW"

    if segments is None:
        segments = compute_slic_superpixels(
            img_rgb,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
        )
    else:
        assert segments.shape == (h, w)

    flat_scores = score_map.reshape(-1)
    flat_segs = segments.reshape(-1)
    n_sp = segments.max() + 1

    if mode == "mean_score":
        # Mean kNN score per superpixel
        sums = np.bincount(flat_segs, weights=flat_scores, minlength=n_sp)
        counts = np.bincount(flat_segs, minlength=n_sp)
        mean_scores = sums / (counts + 1e-8)

        # Broadcast back to pixel grid
        sp_scores = mean_scores[flat_segs].reshape(h, w)
        sp_mask = sp_scores >= threshold

    elif mode == "majority":
        # Majority vote over hard mask (score >= thr)
        hard = (flat_scores >= threshold).astype(np.float32)
        sums = np.bincount(flat_segs, weights=hard, minlength=n_sp)
        counts = np.bincount(flat_segs, minlength=n_sp)
        frac_fg = sums / (counts + 1e-8)

        sp_fg = frac_fg >= sp_fg_frac_thresh
        sp_mask = sp_fg[flat_segs].reshape(h, w)
    else:
        raise ValueError(f"Unknown mode for superpixel refinement: {mode}")

    if sh_mask is not None:
        sp_mask = np.logical_and(sp_mask, sh_mask.astype(bool))

    return sp_mask


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

    Args
    ----
    img_rgb         : HxWx3 uint8 image (DOP20 tile B).
    score_map       : HxW float map, higher => more "LWF-like" (from kNN).
    threshold_center: scalar, where logit=0 in logistic(probability) space.
                      Typically the validation-optimal threshold (thr_best).
    sh_mask         : optional HxW boolean mask. If given, we strongly
                      discourage foreground outside this mask.
    prob_softness   : controls steepness of logistic. Smaller => sharper.
    n_iters         : mean-field iterations in DenseCRF.
    *_w, *_std      : pairwise term hyperparams.

    Returns
    -------
    refined_mask : HxW boolean array (True = foreground).
    """
    h, w, _ = img_rgb.shape
    assert score_map.shape == (h, w), "score_map must have shape (H, W)"

    # --- 1) turn score_map into foreground probability -----------------
    s = score_map.astype(np.float32)

    # Logistic centred at threshold_center
    # logit_fg = (s - threshold_center) / prob_softness
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

    return refined_mask


# ----------------------------------------------------------------------
# 10. Grid search over k and threshold
#     (raw + optional superpixel refinement)
# ----------------------------------------------------------------------


def grid_search_k_threshold(
    img_b: np.ndarray,
    pos_bank: np.ndarray,
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
    use_superpixels: bool = True,
    sp_params: dict | None = None,
):
    """
    Grid search over (k, threshold).

    Returns:
        best_raw_config, best_raw_score_full, best_raw_saliency_full,
        best_sp_config,  best_sp_score_full
    """
    if sp_params is None:
        sp_params = {}

    best_raw_config = None
    best_raw_score_full = None
    best_raw_saliency_full = None
    best_raw_iou = -1.0

    best_sp_config = None
    best_sp_score_full = None
    best_sp_iou = -1.0

    for k in k_values:
        score_full, saliency_full = zero_shot_knn_single_scale_B_with_saliency(
            img_b=img_b,
            pos_bank=pos_bank,
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
        )

        for thr in thresholds:
            # ----- raw per-pixel mask -----
            mask_raw = score_full >= thr
            mask_raw = np.logical_and(mask_raw, sh_buffer_mask_b)

            metrics_raw = compute_metrics(mask_raw, gt_mask_b)
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

            # ----- superpixel-refined mask -----
            if use_superpixels:
                mask_sp = refine_with_superpixels(
                    img_rgb=img_b,
                    score_map=score_full,
                    threshold=thr,
                    sh_mask=sh_buffer_mask_b,
                    **sp_params,
                )

                metrics_sp = compute_metrics(mask_sp, gt_mask_b)
                iou_sp = metrics_sp["iou"]
                f1_sp = metrics_sp["f1"]

                print(
                    f"[eval-SP]  k={k}, thr={thr:.3f} -> "
                    f"IoU={iou_sp:.3f}, F1={f1_sp:.3f}, "
                    f"P={metrics_sp['precision']:.3f}, "
                    f"R={metrics_sp['recall']:.3f}"
                )

                if iou_sp > best_sp_iou:
                    best_sp_iou = iou_sp
                    best_sp_config = {
                        "k": k,
                        "threshold": thr,
                        "source": "superpixels",
                        **metrics_sp,
                    }
                    best_sp_score_full = score_full.copy()

    print("\n[best-raw] configuration:")
    print(best_raw_config)

    if best_sp_config is not None:
        print("\n[best-SP] configuration:")
        print(best_sp_config)

    return (
        best_raw_config,
        best_raw_score_full,
        best_raw_saliency_full,
        best_sp_config,
        best_sp_score_full,
    )


# ----------------------------------------------------------------------
# 11. shapefile export
# ----------------------------------------------------------------------


def export_mask_to_shapefile(mask: np.ndarray,
                             ref_raster_path: str,
                             out_path: str):
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
    t0_main = time_start()

    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    model, processor, device = init_model(model_name)

    # ------------ paths ------------
    img_path = (
        "/home/mak/PycharmProjects/SegEdge/experiments/"
        "get_data_from_api/patches_mt/"
        "dop20_593000_5979000_1km_20cm.tif"
    )  # Image A
    img2_path = (
        "/home/mak/PycharmProjects/SegEdge/experiments/"
        "get_data_from_api/patches_mt/"
        "dop20_592000_5982000_1km_20cm.tif"
    )  # Image B
    lab_path = (
        "/run/media/mak/Partition of 1TB disk/SH_dataset/"
        "planet_labels_2022.tif"
    )  # SH_2022 raster
    gt_vector_path = (
        "/home/mak/PycharmProjects/SegEdge/experiments/"
        "get_data_from_api/patches_mt/"
        "labels_final.shp"
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
    )

    # ------------ SH_2022 buffer mask for B ------------
    sh_buffer_mask_B = build_sh_buffer_mask(labels_SH_B, buffer_pixels)

    # ------------ precompute SLIC superpixels for B ------------
    sp_segments = compute_slic_superpixels(
        img_rgb=img_b,
        n_segments=12000,
        compactness=10.0,
        sigma=1.0,
    )
    sp_params = {
        "segments": sp_segments,
        "mode": "mean_score",
        "sp_fg_frac_thresh": 0.5,  # used only for "majority" mode
        # n_segments/compactness/sigma are ignored when segments provided
    }

    # ------------ grid search on B (raw + superpixels; no CRF) ------------
    K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100]
    THRESHOLDS = np.linspace(0.1, 0.9, 20).tolist()

    (
        best_raw_config,
        best_raw_score_full,
        best_raw_saliency_full,
        best_sp_config,
        best_sp_score_full,
    ) = grid_search_k_threshold(
        img_b=img_b,
        pos_bank=pos_bank,
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
        use_superpixels=True,
        sp_params=sp_params,
    )

    # ------------ reconstruct best raw mask ------------
    k_best_raw = best_raw_config["k"]
    thr_best_raw = best_raw_config["threshold"]

    mask_raw_best = best_raw_score_full >= thr_best_raw
    mask_raw_best = np.logical_and(mask_raw_best, sh_buffer_mask_B)

    # ------------ reconstruct best superpixel mask (if any) ------------
    if best_sp_config is not None:
        k_best_sp = best_sp_config["k"]
        thr_best_sp = best_sp_config["threshold"]
        # score map for that k is best_sp_score_full
        mask_sp_best = refine_with_superpixels(
            img_rgb=img_b,
            score_map=best_sp_score_full,
            threshold=thr_best_sp,
            sh_mask=sh_buffer_mask_B,
            **sp_params,
        )
    else:
        mask_sp_best = None

    # ------------ choose champion (raw vs SP) for CRF calibration ------------
    if best_sp_config is not None and best_sp_config["iou"] > best_raw_config["iou"]:
        champion_config = best_sp_config
        champion_score_full = best_sp_score_full
    else:
        champion_config = best_raw_config
        champion_score_full = best_raw_score_full

    thr_center_for_crf = champion_config["threshold"]
    k_center_for_crf = champion_config["k"]

    print("\n[crf] using champion config for unary centre:")
    print(f"      source={champion_config['source']}, "
          f"k={k_center_for_crf}, thr_center={thr_center_for_crf:.3f}")

    # ------------ CRF search over candidate ks around champion k ------------
    k_min = min(K_VALUES)
    k_max = max(K_VALUES)
    candidate_ks = sorted({
        k for k in [
            k_center_for_crf - 2,
            k_center_for_crf - 1,
            k_center_for_crf,
            k_center_for_crf + 1,
            k_center_for_crf + 2,
        ]
        if k_min <= k <= k_max
    })

    print(f"[crf] candidate ks: {candidate_ks}")

    best_crf_iou = -1.0
    best_crf_metrics = None
    best_crf_k = None
    best_crf_mask = None
    best_crf_score = None

    for k_crf in candidate_ks:
        print(f"[crf] evaluating CRF for k={k_crf}")

        # recompute score map for this k (will reuse cached tile features)
        score_full_k, _ = zero_shot_knn_single_scale_B_with_saliency(
            img_b=img_b,
            pos_bank=pos_bank,
            model=model,
            processor=processor,
            device=device,
            ps=model.config.patch_size,
            tile_size=1024,
            stride=512,
            k=k_crf,
            aggregate_layers=None,
            feature_dir=feature_dir,
            image_id=image_id_b,
        )

        # CRF refinement (uses continuous score map, not threshold)
        mask_crf_k = refine_with_densecrf(
            img_rgb=img_b,
            score_map=score_full_k,
            threshold_center=thr_center_for_crf,
            sh_mask=sh_buffer_mask_B,
            prob_softness=0.05,
            n_iters=5,
        )

        metrics_crf_k = compute_metrics(mask_crf_k, gt_mask_B)
        print(
            f"[crf] k={k_crf} -> "
            f"IoU={metrics_crf_k['iou']:.3f}, "
            f"F1={metrics_crf_k['f1']:.3f}, "
            f"P={metrics_crf_k['precision']:.3f}, "
            f"R={metrics_crf_k['recall']:.3f}"
        )

        if metrics_crf_k["iou"] > best_crf_iou:
            best_crf_iou = metrics_crf_k["iou"]
            best_crf_metrics = metrics_crf_k
            best_crf_k = k_crf
            best_crf_mask = mask_crf_k
            best_crf_score = score_full_k

    print("\n[crf] best CRF configuration:")
    best_crf_config = {"k": best_crf_k, **best_crf_metrics}
    print(best_crf_config)

    # ------------ visualization: raw vs SP vs CRF ------------
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))

    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("Image B (RGB)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(labels_SH_B > 0, cmap="gray")
    axs[0, 1].set_title("SH_2022 raster (B)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(gt_mask_B > 0, cmap="gray")
    axs[0, 2].set_title("Ground truth (labels_final)")
    axs[0, 2].axis("off")

    # Best raw kNN + threshold + buffer
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

    # Best superpixel-refined mask
    if mask_sp_best is not None:
        overlay_sp = img_b.copy()
        overlay_mask_sp = mask_sp_best
        overlay_sp[overlay_mask_sp] = (
            0.5 * overlay_sp[overlay_mask_sp] + 0.5 * np.array([0, 0, 255])
        ).astype(overlay_sp.dtype)

        axs[1, 1].imshow(overlay_sp)
        axs[1, 1].set_title(
            f"Superpixels (k={best_sp_config['k']}, thr={best_sp_config['threshold']:.3f})\n"
            f"IoU={best_sp_config['iou']:.3f}, F1={best_sp_config['f1']:.3f}"
        )
    else:
        axs[1, 1].imshow(img_b)
        axs[1, 1].set_title("Superpixels: disabled / no best config")
    axs[1, 1].axis("off")

    # Best CRF-refined mask
    overlay_crf = img_b.copy()
    overlay_mask_crf = best_crf_mask
    overlay_crf[overlay_mask_crf] = (
        0.5 * overlay_crf[overlay_mask_crf] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay_crf.dtype)
    axs[1, 2].imshow(overlay_crf)
    axs[1, 2].set_title(
        f"CRF (k={best_crf_k}, center_thr={thr_center_for_crf:.3f})\n"
        f"IoU={best_crf_metrics['iou']:.3f}, F1={best_crf_metrics['f1']:.3f}"
    )
    axs[1, 2].axis("off")

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

    # Superpixel best
    if mask_sp_best is not None:
        shp_path_sp = os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_sp.shp")
        export_mask_to_shapefile(
            mask=mask_sp_best,
            ref_raster_path=img2_path,
            out_path=shp_path_sp,
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
        "best_sp_config": best_sp_config,
        "best_sp_score_full": best_sp_score_full,
        "best_crf_config": best_crf_config,
        "best_crf_score_full": best_crf_score,
    }


if __name__ == "__main__":
    main()

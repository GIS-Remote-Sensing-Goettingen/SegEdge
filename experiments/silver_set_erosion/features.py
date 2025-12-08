import os
import time

import numpy as np
import torch
from skimage.transform import resize

from timing_utils import time_start, time_end, DEBUG_TIMING, DEBUG_TIMING_VERBOSE


def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize feature vectors along the last dimension."""
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
    """Yield (y,x,img_tile,label_tile) over an image, respecting stride and tile size."""
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
            lab_tile = labels_hw[y:y_end, x:x_end] if labels_hw is not None else None
            yield y, x, img_tile, lab_tile
            x += stride
        y += stride


def crop_to_multiple_of_ps(img_tile_hw3: np.ndarray,
                           labels_tile_hw: np.ndarray | None,
                           ps: int):
    """Crop a tile so height/width are multiples of patch size ps."""
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps
    img_c = img_tile_hw3[:h_eff, :w_eff]
    lab_c = labels_tile_hw[:h_eff, :w_eff] if labels_tile_hw is not None else None
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("crop_to_multiple_of_ps", t0)
    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(labels_tile: np.ndarray,
                          hp: int,
                          wp: int,
                          pos_frac_thresh: float = 0.1):
    """Convert pixel labels to patch-level pos/neg masks using a fraction threshold."""
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp
    labels_c = labels_tile[:hp * patch_h, :wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)
    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))
    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("labels_to_patch_masks", t0)
    return pos_mask, neg_mask


def tile_feature_path(feature_dir: str,
                      image_id: str,
                      y: int,
                      x: int) -> str:
    """Canonical path for storing a tile's feature .npy."""
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def save_tile_features(feats_tile: np.ndarray,
                       feature_dir: str,
                       image_id: str,
                       y: int,
                       x: int):
    """Persist a tile's features to disk."""
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))


def extract_patch_features_single_scale(image_hw3: np.ndarray,
                                        model,
                                        processor,
                                        device,
                                        ps: int = 16,
                                        aggregate_layers=None):
    """Extract single-scale DINO patch features (Hp×Wp×C) from an RGB image."""
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
    patch_tokens = tokens[:, 1 + reg_tokens:, :]
    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps
    assert hp * wp == num_tokens, f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    feats = patch_tokens[0].cpu().numpy().reshape(hp, wp, dim)
    feats = l2_normalize(feats)
    time_end("extract_patch_features_single_scale", t0)
    return feats, hp, wp


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
    """Precompute and cache all tile features for an image; return in-memory dict."""
    t0 = time_start()
    cache = {}
    cached_tiles = computed_tiles = skipped_tiles = 0
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
                img_c, model, processor, device, ps=ps, aggregate_layers=aggregate_layers
            )
            computed_tiles += 1
            if feature_dir is not None and image_id is not None:
                save_tile_features(feats_tile, feature_dir, image_id, y, x)
        cache[(y, x)] = {"feats": feats_tile, "h_eff": h_eff, "w_eff": w_eff, "hp": hp, "wp": wp}
    time_end("prefetch_features_single_scale_image", t0)
    print(f"[prefetch] tiles={len(cache)} (cached={cached_tiles}, computed={computed_tiles}, skipped={skipped_tiles})")
    return cache

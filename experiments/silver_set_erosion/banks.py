import os
import logging
import numpy as np
from skimage.morphology import erosion, disk

from features import (
    tile_iterator,
    crop_to_multiple_of_ps,
    extract_patch_features_single_scale,
    labels_to_patch_masks,
    save_tile_features,
    tile_feature_path,
    add_local_context_mean,
)
from timing_utils import time_start, time_end
import config as cfg

logger = logging.getLogger(__name__)

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
                             bank_cache_dir: str | None = None,
                             context_radius: int = 0):
    """
    Build positive/negative patch banks from Image A using SH_2022 labels.

    - Extract/caches DINO features per tile.
    - Aggregates pixel labels to patch labels via pos_frac_thresh.
    - Optionally loads/saves banks to disk to avoid recomputation.
    """
    t0 = time_start()

    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        cache_tag = f"{image_id}_ps{ps}_ctx{int(context_radius)}"
        pos_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_neg_bank.npy")
        if os.path.exists(pos_cache_path):
            pos_bank = np.load(pos_cache_path)
            neg_bank = np.load(neg_cache_path) if os.path.exists(neg_cache_path) else None
            time_end("build_banks_single_scale(load_cache)", t0)
            logger.info("loaded banks from %s", bank_cache_dir)
            return pos_bank, neg_bank

    pos_list, neg_list = [], []
    cached_tiles = computed_tiles = 0

    labels_eroded = erosion((labels_a > 0).astype(bool), disk(2))

    for y, x, img_tile, lab_tile in tile_iterator(img_a, labels_eroded, tile_size, stride):
        img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, lab_tile, ps)
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
                img_c, model, processor, device, ps=ps, aggregate_layers=aggregate_layers
            )
            computed_tiles += 1
            if feature_dir is not None and image_id is not None:
                save_tile_features(feats_tile, feature_dir, image_id, y, x)
        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, context_radius)

        pos_mask, neg_mask = labels_to_patch_masks(lab_c, hp, wp, pos_frac_thresh=pos_frac_thresh)
        pos_feats_tile = feats_tile[pos_mask]
        neg_feats_tile = feats_tile[neg_mask]
        if pos_feats_tile.size > 0:
            pos_list.append(pos_feats_tile)
        if neg_feats_tile.size > 0:
            neg_list.append(neg_feats_tile)

    if not pos_list:
        raise ValueError("no positive patches found in Image A; check labels or pos_frac_thresh")

    pos_bank = np.concatenate(pos_list, axis=0)
    neg_bank = np.concatenate(neg_list, axis=0) if neg_list else None

    logger.info("Positive bank size: %s patches", len(pos_bank))
    if neg_bank is not None:
        max_neg = getattr(cfg, "MAX_NEG_BANK", 8000)
        logger.info("Negative bank size: %s patches", len(neg_bank))
        if len(neg_bank) > max_neg:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(neg_bank), size=max_neg, replace=False)
            neg_bank = neg_bank[idx]
            logger.info("subsampled negative bank to %s (MAX_NEG_BANK=%s)", len(neg_bank), max_neg)

    time_end("build_banks_single_scale", t0)
    logger.info("A: cached tiles=%s, computed tiles=%s", cached_tiles, computed_tiles)

    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        cache_tag = f"{image_id}_ps{ps}_ctx{int(context_radius)}"
        pos_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_neg_bank.npy")
        np.save(pos_cache_path, pos_bank.astype(np.float32))
        if neg_bank is not None:
            np.save(neg_cache_path, neg_bank.astype(np.float32))
        logger.info("saved banks to %s", bank_cache_dir)

    return pos_bank, neg_bank

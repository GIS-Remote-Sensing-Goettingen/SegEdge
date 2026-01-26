import os
import time
import logging

import numpy as np
import torch
from skimage.transform import resize

from features import (
    tile_iterator,
    crop_to_multiple_of_ps,
    extract_patch_features_single_scale,
    tile_feature_path,
    save_tile_features,
    add_local_context_mean,
    load_tile_features_if_valid,
)
from metrics_utils import compute_metrics_batch_gpu, compute_metrics_batch_cpu
from timing_utils import time_start, time_end, DEBUG_TIMING
import config as cfg
from metrics_utils import compute_metrics

logger = logging.getLogger(__name__)


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
    context_radius: int = 0,
):
    """
    Compute kNN transfer scores on Image B using GPU matmul.

    Args:
        img_b: full RGB image.
        pos_bank/neg_bank: patch banks from Image A.
        ps: DINO patch size.
        tile_size/stride: tiling parameters.
        k: neighbors to average.
        prefetched_tiles: optional in-memory feature cache.
        neg_alpha: weight for negative bank subtraction.
        use_fp16_matmul: enable half precision on CUDA for speed.
    Returns:
        score_full: per-pixel similarity map.
        saliency_full: simple saliency from top-k sims.
    """
    t0 = time_start()
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    saliency_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)
    cached_tiles = computed_tiles = 0
    resample_factor = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)

    pos_bank_t = torch.from_numpy(pos_bank.astype(np.float32)).to(device)
    pos_bank_t_half = pos_bank_t.half() if use_fp16_matmul and device.type == "cuda" else None
    k_pos_eff = min(k, pos_bank_t.shape[0])

    if neg_bank is not None:
        neg_bank_t = torch.from_numpy(neg_bank.astype(np.float32)).to(device)
        neg_bank_t_half = neg_bank_t.half() if use_fp16_matmul and device.type == "cuda" else None
        k_neg_eff = min(k, neg_bank_t.shape[0])
        use_neg = True
        logger.info(
            "zero_shot: using negative bank size=%s, k_neg_eff=%s, alpha=%s",
            neg_bank_t.shape[0],
            k_neg_eff,
            neg_alpha,
        )
    else:
        neg_bank_t = None
        neg_bank_t_half = None
        k_neg_eff = 0
        use_neg = False
        logger.info("zero_shot: negative bank disabled (neg_bank is None)")

    matmul_time = resize_time = 0.0
    if prefetched_tiles is not None:
        tile_iter = sorted(prefetched_tiles.items())
        logger.debug("zero_shot: using prefetched features for %s tiles", len(tile_iter))
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
            cached_tiles += 1
        else:
            y, x, img_tile, _ = tile_entry
            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
            if h_eff < ps or w_eff < ps:
                time_end(f"zero_shot_tile_skip(y={y},x={x})", t0_tile)
                continue
            feats_tile = None
            hp = wp = None
            if feature_dir is not None and image_id is not None:
                hp = h_eff // ps
                wp = w_eff // ps
                feats_tile = load_tile_features_if_valid(
                    feature_dir,
                    image_id,
                    y,
                    x,
                    expected_hp=hp,
                    expected_wp=wp,
                    ps=ps,
                    resample_factor=resample_factor,
                )
                if feats_tile is not None:
                    cached_tiles += 1
            if feats_tile is None:
                feats_tile, hp, wp = extract_patch_features_single_scale(
                    img_c, model, processor, device, ps=ps, aggregate_layers=aggregate_layers
                )
                computed_tiles += 1
                if feature_dir is not None and image_id is not None:
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_eff,
                        "w_eff": w_eff,
                    }
                    save_tile_features(feats_tile, feature_dir, image_id, y, x, meta=meta)

        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, context_radius)

        x_feats = feats_tile.reshape(-1, feats_tile.shape[-1]).astype(np.float32)
        with torch.no_grad():
            x_feats_t = torch.from_numpy(x_feats).to(device)
            if use_fp16_matmul and device.type == "cuda":
                x_feats_t = x_feats_t.half()
                pos_bank_local = pos_bank_t_half
                neg_bank_local = neg_bank_t_half
            else:
                pos_bank_local = pos_bank_t
                neg_bank_local = neg_bank_t
            t_matmul0 = time.perf_counter() if DEBUG_TIMING else None
            sims_pos_full = x_feats_t @ pos_bank_local.t()
            sims_pos_topk, _ = torch.topk(sims_pos_full, k=k_pos_eff, dim=1)
            score_pos = sims_pos_topk.mean(dim=1)
            if use_neg:
                sims_neg_full = x_feats_t @ neg_bank_local.t()
                sims_neg_topk, _ = torch.topk(sims_neg_full, k=k_neg_eff, dim=1)
                score_neg = sims_neg_topk.mean(dim=1)
                score_batch = score_pos - neg_alpha * score_neg
            else:
                score_batch = score_pos
            if DEBUG_TIMING and t_matmul0 is not None:
                matmul_time += time.perf_counter() - t_matmul0
        score_patch = score_batch.cpu().numpy().reshape(hp, wp)
        sims_pos = sims_pos_topk.float().cpu().numpy()
        weights = sims_pos / (sims_pos.sum(axis=1, keepdims=True) + 1e-8)
        saliency_vals = (weights * sims_pos).sum(axis=1)
        saliency_patch = saliency_vals.reshape(hp, wp)
        t_resize0 = time.perf_counter() if DEBUG_TIMING else None
        score_tile = resize(score_patch, (h_eff, w_eff), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        saliency_tile = resize(saliency_patch, (h_eff, w_eff), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
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
    logger.info("B: cached tiles=%s, computed tiles=%s", cached_tiles, computed_tiles)
    if DEBUG_TIMING:
        logger.debug("k=%s matmul_time=%.2fs, resize_time=%.2fs", k, matmul_time, resize_time)
    return score_full, saliency_full


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
    context_radius: int = 0,
):
    """
    Sweep over k values and global thresholds on Image B; return best raw config + score map.
    """
    t0 = time_start()
    best_raw_config = None
    best_raw_score_full = None
    best_raw_saliency_full = None
    best_raw_iou = -1.0

    for k in k_values:
        t0_k_total = time_start()
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
            neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
            prefetched_tiles=prefetched_tiles_b,
            use_fp16_matmul=use_fp16_matmul,
            context_radius=context_radius,
        )
        time_end(f"grid_search_score_full(k={k})", t0_k_score)

        metrics_raw_list = None
        if getattr(cfg, "USE_GPU_THRESHOLD_METRICS", True) and device.type == "cuda":
            try:
                metrics_raw_list = compute_metrics_batch_gpu(
                    score_map=score_full,
                    thresholds=thresholds,
                    sh_mask=sh_buffer_mask_b,
                    gt_mask=gt_mask_b,
                    device=device,
                    batch_size=getattr(cfg, "THRESHOLD_BATCH_SIZE", 8),
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                metrics_raw_list = None
                logger.warning("OOM during GPU threshold metrics; falling back to CPU")
        if metrics_raw_list is None:
            metrics_raw_list = compute_metrics_batch_cpu(
                score_map=score_full,
                thresholds=thresholds,
                sh_mask=sh_buffer_mask_b,
                gt_mask=gt_mask_b,
                batch_size=getattr(cfg, "THRESHOLD_CPU_BATCH_SIZE", 16),
            )

        for metrics_raw in metrics_raw_list:
            thr = metrics_raw["threshold"]
            iou_raw = metrics_raw["iou"]
            f1_raw = metrics_raw["f1"]
            logger.info(
                "eval-raw k=%s, thr=%.3f -> IoU=%.3f, F1=%.3f, P=%.3f, R=%.3f",
                k,
                thr,
                iou_raw,
                f1_raw,
                metrics_raw["precision"],
                metrics_raw["recall"],
            )
            if iou_raw > best_raw_iou:
                best_raw_iou = iou_raw
                best_raw_config = {"k": k, "threshold": thr, "source": "raw", **metrics_raw}
                best_raw_score_full = score_full.copy()
                best_raw_saliency_full = saliency_full.copy()

        time_end(f"k_loop_total(k={k})", t0_k_total)

    logger.info("best-raw configuration: %s", best_raw_config)
    time_end("grid_search_k_threshold", t0)
    return best_raw_config, best_raw_score_full, best_raw_saliency_full


def fine_tune_threshold(
    score_map: np.ndarray,
    base_threshold: float,
    sh_mask: np.ndarray | None,
    gt_mask: np.ndarray,
    step: float = 0.01,
    window: float = 0.08,
):
    """Refine a scalar threshold around a base value; keeps best IoU mask."""
    t0 = time_start()
    thr_min = max(0.0, base_threshold - window)
    thr_max = min(1.0, base_threshold + window)
    thr_vals = np.arange(thr_min, thr_max + 1e-8, step)
    best_thr = base_threshold
    best_metrics = None
    best_mask = None
    best_iou = -1.0
    for thr in thr_vals:
        mask = score_map >= thr
        if sh_mask is not None:
            mask = np.logical_and(mask, sh_mask)
        metrics = compute_metrics(mask, gt_mask)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thr = thr
            best_metrics = metrics
            best_mask = mask
    logger.info(
        "tune-thr base=%.3f -> best=%.3f IoU=%.3f, F1=%.3f",
        base_threshold,
        best_thr,
        best_metrics["iou"],
        best_metrics["f1"],
    )
    time_end("fine_tune_threshold", t0)
    return best_thr, best_metrics, best_mask

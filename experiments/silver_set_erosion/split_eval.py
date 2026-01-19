import logging
import os

import numpy as np
import torch
from scipy.ndimage import median_filter

import config as cfg
from logging_utils import setup_logging
from banks import build_banks_single_scale
from crf_utils import crf_grid_search, refine_with_densecrf
from features import prefetch_features_single_scale_image
from io_utils import (
    load_dop20_image,
    reproject_labels_to_image,
    rasterize_vector_labels,
    build_sh_buffer_mask,
)
from knn import grid_search_k_threshold, fine_tune_threshold, zero_shot_knn_single_scale_B_with_saliency
from metrics_utils import compute_metrics, compute_metrics_batch_cpu, compute_metrics_batch_gpu
from shadow_filter import shadow_filter_grid
from transformers import AutoImageProcessor, AutoModel
from xdboost import build_xgb_dataset, hyperparam_search_xgb_iou, train_xgb_classifier, xgb_score_image_b

logger = logging.getLogger(__name__)


def init_model(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor, device


def build_banks_for_sources(model, processor, device, ps, tile_size, stride, feature_dir):
    img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    lab_a_paths = [cfg.SOURCE_LABEL_RASTER] * len(img_a_paths)

    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    pos_banks = []
    neg_banks = []
    for img_a_path, lab_a_path in zip(img_a_paths, lab_a_paths, strict=True):
        image_id_a = os.path.splitext(os.path.basename(img_a_path))[0]
        logger.info("bank source A: %s (labels: %s)", img_a_path, lab_a_path)
        img_a = load_dop20_image(img_a_path)
        labels_a = reproject_labels_to_image(img_a_path, lab_a_path)
        pos_i, neg_i = build_banks_single_scale(
            img_a,
            labels_a,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            getattr(cfg, "POS_FRAC_THRESH", 0.1),
            None,
            feature_dir,
            image_id_a,
            cfg.BANK_CACHE_DIR,
            context_radius=context_radius,
        )
        pos_banks.append(pos_i)
        if neg_i is not None and len(neg_i) > 0:
            neg_banks.append(neg_i)

    pos_bank = np.concatenate(pos_banks, axis=0)
    neg_bank = np.concatenate(neg_banks, axis=0) if neg_banks else None
    logger.info("combined banks: pos=%s, neg=%s", len(pos_bank), 0 if neg_bank is None else len(neg_bank))
    return pos_bank, neg_bank


def build_xgb_training_data(ps, tile_size, stride, feature_dir):
    img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    lab_a_paths = [cfg.SOURCE_LABEL_RASTER] * len(img_a_paths)

    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    X_list = []
    y_list = []
    for img_a_path, lab_a_path in zip(img_a_paths, lab_a_paths, strict=True):
        image_id_a = os.path.splitext(os.path.basename(img_a_path))[0]
        img_a = load_dop20_image(img_a_path)
        labels_a = reproject_labels_to_image(img_a_path, lab_a_path)
        X_i, y_i = build_xgb_dataset(
            img_a,
            labels_a,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_a,
            pos_frac=cfg.POS_FRAC_THRESH,
            max_neg=getattr(cfg, "MAX_NEG_BANK", 8000),
            context_radius=context_radius,
        )
        X_list.append(X_i)
        y_list.append(y_i)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def prep_b_tile(img_path, gt_paths):
    img_b = load_dop20_image(img_path)
    labels_sh = reproject_labels_to_image(img_path, cfg.SOURCE_LABEL_RASTER)
    gt_mask_b = rasterize_vector_labels(gt_paths, img_path)

    with __import__("rasterio").open(img_path) as src:
        pixel_size_m = abs(src.transform.a)
    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)
    if getattr(cfg, "CLIP_GT_TO_BUFFER", False):
        gt_mask_eval = np.logical_and(gt_mask_b, sh_buffer_mask)
    else:
        gt_mask_eval = gt_mask_b

    return img_b, labels_sh, gt_mask_eval, sh_buffer_mask


def log_metrics(tag, metrics):
    logger.info(
        "%s IoU=%.3f, F1=%.3f, P=%.3f, R=%.3f",
        tag,
        metrics["iou"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )


def tune_on_validation(
    model,
    processor,
    device,
    pos_bank,
    neg_bank,
    ps,
    tile_size,
    stride,
    feature_dir,
    img_b_path,
    gt_paths,
):
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    img_b, labels_sh_b, gt_mask_eval, sh_buffer_mask = prep_b_tile(img_b_path, gt_paths)
    image_id_b = os.path.splitext(os.path.basename(img_b_path))[0]

    prefetched_b = prefetch_features_single_scale_image(
        img_b,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        None,
        feature_dir,
        image_id_b,
    )

    best_raw_config, best_raw_score_full, _ = grid_search_k_threshold(
        img_b,
        pos_bank,
        neg_bank,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        cfg.K_VALUES,
        cfg.THRESHOLDS,
        feature_dir,
        image_id_b,
        sh_buffer_mask,
        gt_mask_eval,
        prefetched_b,
        getattr(cfg, "USE_FP16_KNN", True),
        context_radius=context_radius,
    )
    thr_refined, metrics_refined, mask_raw_best = fine_tune_threshold(
        best_raw_score_full,
        best_raw_config["threshold"],
        sh_buffer_mask,
        gt_mask_eval,
    )
    if metrics_refined["iou"] >= best_raw_config["iou"]:
        best_raw_config = {**best_raw_config, "threshold": thr_refined, **metrics_refined}
    else:
        mask_raw_best = (best_raw_score_full >= best_raw_config["threshold"]) & sh_buffer_mask
    mask_raw_best = median_filter(mask_raw_best.astype(np.uint8), size=3) > 0
    metrics_raw_filtered = compute_metrics(mask_raw_best, gt_mask_eval)
    best_raw_config = {**best_raw_config, **metrics_raw_filtered}
    log_metrics("val kNN", best_raw_config)

    X, y = build_xgb_training_data(ps, tile_size, stride, feature_dir)
    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)

    if param_grid:
        bst, best_params_xgb, _, best_thr_xgb, best_metrics_xgb = hyperparam_search_xgb_iou(
            X,
            y,
            cfg.THRESHOLDS,
            sh_buffer_mask,
            gt_mask_eval,
            img_b,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_b,
            prefetched_tiles=prefetched_b,
            device=device,
            use_gpu=use_gpu_xgb,
            param_grid=param_grid,
            num_boost_round=num_boost_round,
            val_fraction=val_fraction,
            early_stopping_rounds=early_stop,
            verbose_eval=verbose_eval,
            seed=42,
            context_radius=context_radius,
        )
        best_xgb_config = {"k": -1, "threshold": best_thr_xgb, "source": "xgb", **best_metrics_xgb, "params": best_params_xgb}
    else:
        bst = train_xgb_classifier(X, y, use_gpu=use_gpu_xgb, num_boost_round=num_boost_round, verbose_eval=verbose_eval)
        score_full_xgb = xgb_score_image_b(img_b, bst, ps, tile_size, stride, feature_dir, image_id_b, prefetched_tiles=prefetched_b, context_radius=context_radius)
        try:
            metrics_list = compute_metrics_batch_gpu(score_full_xgb, cfg.THRESHOLDS, sh_buffer_mask, gt_mask_eval, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            metrics_list = compute_metrics_batch_cpu(score_full_xgb, cfg.THRESHOLDS, sh_buffer_mask, gt_mask_eval)
        best_xgb = max(metrics_list, key=lambda m: m["iou"])
        mask_xgb = (score_full_xgb >= best_xgb["threshold"]) & sh_buffer_mask
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
        metrics_xgb_filtered = compute_metrics(mask_xgb, gt_mask_eval)
        best_xgb_config = {"k": -1, "threshold": best_xgb["threshold"], "source": "xgb", **metrics_xgb_filtered, "params": None}

    log_metrics("val XGB", best_xgb_config)

    champion_config = best_raw_config if best_raw_config["iou"] >= best_xgb_config["iou"] else best_xgb_config
    champion_source = champion_config["source"]
    champion_score = best_raw_score_full if champion_source == "raw" else xgb_score_image_b(
        img_b,
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_b,
        context_radius=context_radius,
    )
    thr_center = champion_config["threshold"]

    best_crf_cfg, best_crf_mask = crf_grid_search(
        img_b,
        champion_score,
        thr_center,
        sh_buffer_mask,
        gt_mask_eval,
        cfg.PROB_SOFTNESS_VALUES,
        cfg.POS_W_VALUES,
        cfg.POS_XY_STD_VALUES,
        cfg.BILATERAL_W_VALUES,
        cfg.BILATERAL_XY_STD_VALUES,
        cfg.BILATERAL_RGB_STD_VALUES,
        5,
        getattr(cfg, "CRF_MAX_CONFIGS", 64),
        2,
        cfg.CRF_NUM_WORKERS,
        "process",
    )
    log_metrics("val CRF", best_crf_cfg)

    best_shadow_cfg, _ = shadow_filter_grid(
        img_b,
        best_crf_mask,
        gt_mask_eval,
        cfg.SHADOW_WEIGHT_SETS,
        cfg.SHADOW_THRESHOLDS,
    )
    logger.info("val shadow best: %s", best_shadow_cfg)

    return {
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": best_crf_cfg,
        "best_shadow_cfg": best_shadow_cfg,
        "bst": bst,
    }


def eval_holdout_tile(
    model,
    processor,
    device,
    pos_bank,
    neg_bank,
    ps,
    tile_size,
    stride,
    feature_dir,
    img_b_path,
    gt_paths,
    tuned,
):
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    img_b, labels_sh_b, gt_mask_eval, sh_buffer_mask = prep_b_tile(img_b_path, gt_paths)
    image_id_b = os.path.splitext(os.path.basename(img_b_path))[0]
    prefetched_b = prefetch_features_single_scale_image(
        img_b, model, processor, device, ps, tile_size, stride, None, feature_dir, image_id_b
    )

    k = tuned["best_raw_config"]["k"]
    score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
        img_b,
        pos_bank,
        neg_bank,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        k=k,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=image_id_b,
        neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
        prefetched_tiles=prefetched_b,
        use_fp16_matmul=getattr(cfg, "USE_FP16_KNN", True),
        context_radius=context_radius,
    )
    knn_thr = tuned["best_raw_config"]["threshold"]
    mask_knn = (score_knn >= knn_thr) & sh_buffer_mask
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    metrics_knn = compute_metrics(mask_knn, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} kNN", metrics_knn)

    bst = tuned["bst"]
    score_xgb = xgb_score_image_b(
        img_b,
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_b,
        context_radius=context_radius,
    )
    xgb_thr = tuned["best_xgb_config"]["threshold"]
    mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} XGB", metrics_xgb)

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
        thr_center = knn_thr
    else:
        champion_score = score_xgb
        thr_center = xgb_thr

    crf_cfg = tuned["best_crf_config"]
    mask_crf = refine_with_densecrf(
        img_b,
        champion_score,
        thr_center,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
    )
    metrics_crf = compute_metrics(mask_crf, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} CRF", metrics_crf)

    shadow_cfg = tuned["best_shadow_cfg"]
    _, mask_shadow = shadow_filter_grid(
        img_b,
        mask_crf,
        gt_mask_eval,
        [shadow_cfg["weights"]],
        [shadow_cfg["threshold"]],
    )
    metrics_shadow = compute_metrics(mask_shadow, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} shadow", metrics_shadow)


def main():
    setup_logging(getattr(cfg, "LOG_PATH", None))
    model, processor, device = init_model(cfg.MODEL_NAME)
    ps = getattr(cfg, "PATCH_SIZE", model.config.patch_size)
    tile_size = getattr(cfg, "TILE_SIZE", 1024)
    stride = getattr(cfg, "STRIDE", tile_size)
    feature_dir = cfg.FEATURE_DIR
    os.makedirs(feature_dir, exist_ok=True)

    val_b_path = getattr(cfg, "VAL_TILE", None) or cfg.TARGET_TILE
    gt_paths = cfg.EVAL_GT_VECTORS

    holdout_b_paths = getattr(cfg, "HOLDOUT_TILES", None)
    if not holdout_b_paths:
        raise ValueError("HOLDOUT_TILES must be set for split evaluation.")

    pos_bank, neg_bank = build_banks_for_sources(model, processor, device, ps, tile_size, stride, feature_dir)
    tuned = tune_on_validation(
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        ps,
        tile_size,
        stride,
        feature_dir,
        val_b_path,
        gt_paths,
    )

    for i, b_path in enumerate(holdout_b_paths):
        logger.info("holdout eval on %s (gt=%s)", b_path, gt_paths)
        eval_holdout_tile(
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            ps,
            tile_size,
            stride,
            feature_dir,
            b_path,
            gt_paths,
            tuned,
        )


if __name__ == "__main__":
    main()

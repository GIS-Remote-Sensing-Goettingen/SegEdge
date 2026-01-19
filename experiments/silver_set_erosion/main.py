# SPDX-License-Identifier: GPL-2.0
# Orchestrator for the DINOv3 zero-shot pipeline (kNN + CRF).

import os
import logging
import numpy as np
import torch
from skimage.transform import resize

import config as cfg
from experiments.silver_set_erosion.xdboost import (
    build_xgb_dataset,
    train_xgb_classifier,
    hyperparam_search_xgb,
    hyperparam_search_xgb_iou,
    xgb_score_image_b,
)
from timing_utils import time_start, time_end, DEBUG_TIMING
from scipy.ndimage import median_filter
from io_utils import (
    load_dop20_image,
    reproject_labels_to_image,
    rasterize_vector_labels,
    build_sh_buffer_mask,
    export_mask_to_shapefile,
    consolidate_features_for_image,
    export_best_settings,
)
from features import prefetch_features_single_scale_image
from banks import build_banks_single_scale
from knn import grid_search_k_threshold, fine_tune_threshold
from metrics_utils import compute_oracle_upper_bound, compute_metrics, compute_metrics_batch_cpu, \
    compute_metrics_batch_gpu
from crf_utils import crf_grid_search
from plotting import save_plot, save_best_model_plot, save_knn_xgb_gt_plot
from transformers import AutoImageProcessor, AutoModel
from shadow_filter import shadow_filter_grid
from logging_utils import setup_logging


# Config-driven flags
USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)
CRF_MAX_CONFIGS = getattr(cfg, "CRF_MAX_CONFIGS", 64)

logger = logging.getLogger(__name__)


def init_model(model_name: str):
    """Load DINOv3 backbone + processor on CPU/GPU with timing."""
    t0 = time_start()
    processor = AutoImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    time_end("init_model", t0)
    return model, processor, device


def main():
    """
    Full segmentation pipeline:
         • Build banks from Image A
         • kNN transfer to Image B
         • Threshold selection
         • CRF refinement
         • Shadow filtering
         • Export diagnostics + shapefiles
    """

    setup_logging(getattr(cfg, "LOG_PATH", None))
    t0_main = time_start()
    model_name = cfg.MODEL_NAME

    # ------------------------------------------------------------
    # Init DINOv3 model & processor
    # ------------------------------------------------------------
    model, processor, device = init_model(model_name)
    ps = getattr(cfg, "PATCH_SIZE", model.config.patch_size)
    tile_size = getattr(cfg, "TILE_SIZE", 1024)
    stride = getattr(cfg, "STRIDE", tile_size)

    # ------------------------------------------------------------
    # Resolve paths to imagery + SH_2022 + GT vector labels
    # ------------------------------------------------------------
    img_path_default = cfg.IMG_PATH
    img2_path = cfg.IMG2_PATH
    lab_path = cfg.LAB_PATH
    gt_vector_paths = getattr(cfg, "GT_VECTOR_PATHS", None) or cfg.GT_VECTOR_PATH

    # ------------------------------------------------------------
    # Resolve one or more labeled source images (Image A list)
    # ------------------------------------------------------------
    img_a_paths = getattr(cfg, "IMG_A_PATHS", None) or [img_path_default]
    lab_a_paths = getattr(cfg, "LAB_A_PATHS", None)
    if lab_a_paths is None:
        lab_a_paths = [lab_path] * len(img_a_paths)
    if len(lab_a_paths) != len(img_a_paths):
        raise ValueError("LAB_A_PATHS must be None or have the same length as IMG_A_PATHS")

    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)

    # ------------------------------------------------------------
    # Load imagery and reproject labels (Image B + SH + GT)
    # ------------------------------------------------------------
    t0_data = time_start()
    img_b = load_dop20_image(
        img2_path       # target image B for inference
    )
    labels_SH_B = reproject_labels_to_image(
        img2_path,      # image B path
        lab_path        # SH_2022 raster
    )

    gt_mask_B = rasterize_vector_labels(
        gt_vector_paths, # vector ground truth polygon(s) for B (union-merged if list)
        img2_path        # raster reference to match CRS/resolution
    )
    time_end("data_loading_and_reprojection", t0_data)

    logger.debug("GT positives on B: %s", gt_mask_B.sum())
    logger.debug("SH_2022 positives on B: %s", (labels_SH_B > 0).sum())

    # ------------------------------------------------------------
    # Build SH_2022 buffer (spatial prior)
    # ------------------------------------------------------------
    with __import__('rasterio').open(img2_path) as src:
        pixel_size_m = abs(src.transform.a)

    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    logger.info("pixel_size=%.3f m, buffer_m=%s, buffer_pixels=%s", pixel_size_m, buffer_m, buffer_pixels)

    sh_buffer_mask_B = build_sh_buffer_mask(
        labels_SH_B,    # SH_2022 label raster on B
        buffer_pixels   # radius in pixel units for dilation
    )

    if getattr(cfg, "CLIP_GT_TO_BUFFER", False):
        gt_mask_eval = np.logical_and(gt_mask_B, sh_buffer_mask_B)
        logger.info("CLIP_GT_TO_BUFFER enabled: GT positives -> %s (was %s)", gt_mask_eval.sum(), gt_mask_B.sum())
    else:
        gt_mask_eval = gt_mask_B

    _ = compute_oracle_upper_bound(
        gt_mask_eval,      # GT mask on B (possibly clipped)
        sh_buffer_mask_B  # SH buffer region (max allowed FG region)
    )

    # ------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------
    feature_dir = cfg.FEATURE_DIR
    os.makedirs(feature_dir, exist_ok=True)

    image_id_a_list = [os.path.splitext(os.path.basename(p))[0] for p in img_a_paths]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    # ------------------------------------------------------------
    # Build DINOv3 positive/negative banks from one or more Image A sources
    # ------------------------------------------------------------
    pos_banks = []
    neg_banks = []
    for img_a_path, lab_a_path, image_id_a in zip(img_a_paths, lab_a_paths, image_id_a_list, strict=True):
        logger.info("source A: %s (labels: %s)", img_a_path, lab_a_path)
        img_a = load_dop20_image(img_a_path)
        labels_A = reproject_labels_to_image(img_a_path, lab_a_path)

        pos_bank_i, neg_bank_i = build_banks_single_scale(
            img_a,                    # img_a: RGB array
            labels_A,                 # SH labels on A (reprojected)
            model,                    # DINO model
            processor,                # processor
            device,                   # GPU/CPU device
            ps,                       # patch size
            tile_size,                # tiling
            stride,                   # overlap
            getattr(cfg, "POS_FRAC_THRESH", 0.1),
            None,
            feature_dir,
            image_id_a,
            cfg.BANK_CACHE_DIR,
            context_radius=context_radius,
        )
        pos_banks.append(pos_bank_i)
        if neg_bank_i is not None and len(neg_bank_i) > 0:
            neg_banks.append(neg_bank_i)

    pos_bank = np.concatenate(pos_banks, axis=0)
    neg_bank = np.concatenate(neg_banks, axis=0) if neg_banks else None
    logger.info("combined banks: pos=%s, neg=%s", len(pos_bank), 0 if neg_bank is None else len(neg_bank))

    # ------------------------------------------------------------
    # Prefetch DINO features for Image B (so kNN grid search is fast)
    # ------------------------------------------------------------
    prefetched_b = prefetch_features_single_scale_image(
        img_b,                  # img_hw3: RGB array of image B
        model,                  # DINO model
        processor,              # processor
        device,                 # GPU/CPU
        ps,                     # ps: ViT patch size
        tile_size,              # tile_size
        stride,                 # stride
        None,                   # aggregate_layers
        feature_dir,            # location to load/store B’s features
        image_id_b              # ID for B tiles
    )

    # ------------------------------------------------------------
    # kNN grid search (k-values × thresholds)
    # ------------------------------------------------------------
    best_raw_config, best_raw_score_full, best_raw_saliency_full = grid_search_k_threshold(
        img_b,                 # full RGB image B
        pos_bank,              # positive bank (N_pos × C)
        neg_bank,              # negative bank (N_neg × C)
        model,                 # DINO model
        processor,             # image processor
        device,                # GPU device
        ps,                    # ps
        tile_size,             # tile_size
        stride,                # stride
        cfg.K_VALUES,          # list of k-values to try (e.g. [1,3,5,7])
        cfg.THRESHOLDS,        # global thresholds for FG mask
        feature_dir,           # directory for B/features
        image_id_b,            # B tile cache ID
        sh_buffer_mask_B,      # spatial prior mask
        gt_mask_eval,          # ground-truth for scoring
        prefetched_b,          # cached DINO features for B
        USE_FP16_KNN,          # use half precision matmul for speed
        context_radius=context_radius,
    )

    # ------------------------------------------------------------
    # Local threshold refinement around the best raw solution
    # ------------------------------------------------------------
    thr_best_raw_refined, metrics_raw_refined, mask_raw_best = fine_tune_threshold(
        best_raw_score_full,   # score_map from best k
        best_raw_config["threshold"],  # base threshold to refine
        sh_buffer_mask_B,      # prior mask
        gt_mask_eval           # GT for evaluation
    )

    # Update configuration if refined solution is better
    if metrics_raw_refined["iou"] >= best_raw_config["iou"]:
        best_raw_config = {
            **best_raw_config,
            "threshold": thr_best_raw_refined,
            **metrics_raw_refined,
        }
    else:
        mask_raw_best = best_raw_score_full >= best_raw_config["threshold"]
        mask_raw_best = np.logical_and(mask_raw_best, sh_buffer_mask_B)

    # Median filter to clean speckle after thresholding
    mask_raw_best = median_filter(mask_raw_best.astype(np.uint8), size=3) > 0
    metrics_raw_filtered = compute_metrics(mask_raw_best, gt_mask_eval)
    best_raw_config = {**best_raw_config, **metrics_raw_filtered}

    # Preserve kNN mask/config before potential champion swap
    mask_knn = mask_raw_best.copy()
    knn_config = best_raw_config.copy()

    champion_config = best_raw_config
    champion_score_full = best_raw_score_full
    thr_center_for_crf = champion_config["threshold"]
    k_center_for_crf = champion_config["k"]

    # ------------------------------------------------------------
    # XGBoost branch (patch-level classifier)
    # ------------------------------------------------------------
    X_list = []
    y_list = []
    for img_a_path, lab_a_path, image_id_a in zip(img_a_paths, lab_a_paths, image_id_a_list, strict=True):
        img_a = load_dop20_image(img_a_path)
        labels_A = reproject_labels_to_image(img_a_path, lab_a_path)
        X_i, y_i = build_xgb_dataset(
            img_a,
            labels_A,
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
    X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.float32)

    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)

    if param_grid:
        bst, best_params_xgb, best_iou_xgb, best_thr_xgb, best_metrics_xgb = hyperparam_search_xgb_iou(
            X,
            y,
            cfg.THRESHOLDS,
            sh_buffer_mask_B,
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
        best_xgb = best_metrics_xgb
        best_xgb_config = {
            "k": -1,
            "threshold": best_thr_xgb,
            "source": "xgb",
            **best_xgb,
            "params": best_params_xgb,
        }
        score_full_xgb = xgb_score_image_b(
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
        mask_xgb = (score_full_xgb >= best_thr_xgb) & sh_buffer_mask_B
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
        metrics_xgb_filtered = compute_metrics(mask_xgb, gt_mask_eval)
        best_xgb_config = {**best_xgb_config, **metrics_xgb_filtered}
        logger.info("xgb-best thr=%.3f, IoU=%.3f, F1=%.3f", best_thr_xgb, best_xgb_config["iou"], best_xgb_config["f1"])
    else:
        bst = train_xgb_classifier(
            X,
            y,
            use_gpu=use_gpu_xgb,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
        )
        best_params_xgb = None
        score_full_xgb = xgb_score_image_b(
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
        try:
            metrics_list = compute_metrics_batch_gpu(score_full_xgb, cfg.THRESHOLDS, sh_buffer_mask_B, gt_mask_eval, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            metrics_list = compute_metrics_batch_cpu(score_full_xgb, cfg.THRESHOLDS, sh_buffer_mask_B, gt_mask_eval)
        best_xgb = max(metrics_list, key=lambda m: m["iou"])
        mask_xgb = (score_full_xgb >= best_xgb["threshold"]) & sh_buffer_mask_B
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
        metrics_xgb_filtered = compute_metrics(mask_xgb, gt_mask_eval)
        best_xgb_config = {
            "k": -1,
            "threshold": best_xgb["threshold"],
            "source": "xgb",
            **metrics_xgb_filtered,
            "params": best_params_xgb,
        }
        logger.info("xgb-best thr=%.3f, IoU=%.3f, F1=%.3f", best_xgb["threshold"], best_xgb_config["iou"], best_xgb_config["f1"])

    # Champion selection: choose better of kNN or XGB for CRF
    if best_xgb_config["iou"] > champion_config["iou"]:
        champion_config = best_xgb_config
        champion_score_full = score_full_xgb
        thr_center_for_crf = champion_config["threshold"]
        k_center_for_crf = champion_config["k"]
        mask_raw_best = mask_xgb

    # Save GT vs kNN vs XGB overlays
    save_knn_xgb_gt_plot(
        img_b,
        gt_mask_eval,
        mask_knn,
        mask_xgb,
        cfg.PLOT_DIR,
        image_id_b,
        title_knn=f"kNN IoU={knn_config['iou']:.3f}",
        title_xgb=f"XGB IoU={best_xgb_config['iou']:.3f}",
        filename_suffix="knn_vs_xgb.png",
    )

    # Quick visualization of champion vs GT before CRF
    save_best_model_plot(
        img_b,
        gt_mask_eval,
        mask_raw_best,
        title=f"Champion ({champion_config['source']}) IoU={champion_config['iou']:.3f}",
        plot_dir=cfg.PLOT_DIR,
        image_id_b=image_id_b,
        filename_suffix="champion_pre_crf.png",
    )

    # Free large intermediates before CRF
    for _obj in ["pos_bank", "neg_bank", "prefetched_b", "best_raw_score_full", "best_raw_saliency_full", "score_full_xgb"]:
        if _obj in locals():
            del locals()[_obj]
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # CRF grid search (spatial regularization)
    # ------------------------------------------------------------
    best_crf_cfg_inner, best_crf_mask = crf_grid_search(
        img_b,                       # RGB on B
        champion_score_full,         # score map from best raw stage
        thr_center_for_crf,          # CRF unary logistic center
        sh_buffer_mask_B,            # prior mask
        gt_mask_eval,                # GT mask
        cfg.PROB_SOFTNESS_VALUES,    # CRF unary softness candidates
        cfg.POS_W_VALUES,            # Gaussian pairwise compat weights
        cfg.POS_XY_STD_VALUES,       # Gaussian XY sigma
        cfg.BILATERAL_W_VALUES,      # bilateral compat weights
        cfg.BILATERAL_XY_STD_VALUES, # bilateral XY sigma
        cfg.BILATERAL_RGB_STD_VALUES,# bilateral RGB sigma
        5,                           # n_iters: mean-field iterations
        CRF_MAX_CONFIGS,             # max configs to search
        2,                           # downsample_factor: speed boost
        cfg.CRF_NUM_WORKERS,         # multiprocessing workers
        "process"                    # backend: process-based parallelism
    )

    best_crf_config = {
        "k": k_center_for_crf,       # inherited from raw best
        **best_crf_cfg_inner         # CRF hyperparams
    }

    # If CRF was computed at lower resolution → upsample to original grid
    if best_crf_mask.shape != img_b.shape[:2]:
        best_crf_mask_full = resize(
            best_crf_mask.astype(np.float32),   # low-res CRF output
            (img_b.shape[0], img_b.shape[1]),   # original size
            order=0, preserve_range=True, anti_aliasing=False
        ) > 0.5
        best_crf_mask = best_crf_mask_full

        metrics_crf_full = compute_metrics(
            best_crf_mask,      # upsampled CRF mask
            gt_mask_eval        # GT
        )
        logger.info(
            "crf-upsampled IoU=%.3f, F1=%.3f, P=%.3f, R=%.3f",
            metrics_crf_full["iou"],
            metrics_crf_full["f1"],
            metrics_crf_full["precision"],
            metrics_crf_full["recall"],
        )

    # ------------------------------------------------------------
    # Shadow filtering stage
    # ------------------------------------------------------------
    shadow_cfg, shadow_mask = shadow_filter_grid(
        img_b,                                 # RGB B
        best_crf_mask,                         # mask after CRF
        gt_mask_eval,                          # GT for scoring
        cfg.SHADOW_WEIGHT_SETS,                # e.g. [(1,1,1), (0.7,1,1)]
        cfg.SHADOW_THRESHOLDS                  # thresholds in weighted-sum space
    )
    shadow_best = {"cfg": shadow_cfg, "mask": shadow_mask}

    # ------------------------------------------------------------
    # Diagnostics + visualization
    # ------------------------------------------------------------
    save_plot(
        img_b,                 # RGB B
        gt_mask_B,             # GT mask
        mask_raw_best,         # raw (after refined threshold)
        best_raw_config,       # raw pipeline config
        best_crf_mask,         # CRF output
        best_crf_config,       # CRF config
        thr_center_for_crf,    # raw threshold used for CRF unary center
        cfg.PLOT_DIR,          # where to save plots
        image_id_b,            # identifier for outputs
        best_shadow=shadow_best  # shadow filtering results
    )

    # ------------------------------------------------------------
    # Export shapefiles (raw, CRF, shadow)
    # ------------------------------------------------------------
    base_name_b = os.path.splitext(os.path.basename(img2_path))[0]
    out_dir_b = os.path.dirname(img2_path)

    export_mask_to_shapefile(
        mask_raw_best,                         # raw prediction mask
        img2_path,                             # reference image
        os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_raw.shp")
    )
    export_mask_to_shapefile(
        best_crf_mask,                         # CRF-refined mask
        img2_path,
        os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_crf.shp")
    )
    export_mask_to_shapefile(
        shadow_mask,                           # shadow-filtered final mask
        img2_path,
        os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_shadow.shp")
    )

    # ------------------------------------------------------------
    # Consolidate tile-level feature files (.npy) → one per image
    # ------------------------------------------------------------
    for image_id_a in image_id_a_list:
        consolidate_features_for_image(feature_dir, image_id_a)
    consolidate_features_for_image(
        feature_dir,
        image_id_b          # ID for image B
    )

    # ------------------------------------------------------------
    # Export best settings for reproducibility
    # ------------------------------------------------------------
    export_best_settings(
        best_raw_config,    # raw pipeline config
        best_crf_config,    # CRF config
        model_name,         # DINO model name
        img_a_paths,        # path(s) to A
        img2_path,          # path to B
        buffer_m,           # SH buffer in meters
        pixel_size_m,       # pixel spacing in meters
        shadow_cfg=shadow_cfg,
        extra_settings={
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": getattr(cfg, "NEG_ALPHA", 1.0),
            "pos_frac_thresh": getattr(cfg, "POS_FRAC_THRESH", 0.1),
        },
    )

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()

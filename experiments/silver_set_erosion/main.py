# SPDX-License-Identifier: GPL-2.0
# Orchestrator for the DINOv3 zero-shot pipeline (kNN + CRF).

import os
import numpy as np
import torch
from skimage.transform import resize

import config as cfg
from timing_utils import time_start, time_end, DEBUG_TIMING
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
from metrics_utils import compute_oracle_upper_bound, compute_metrics
from crf_utils import crf_grid_search
from plotting import save_plot
from transformers import AutoImageProcessor, AutoModel
from shadow_filter import shadow_filter_grid


# Config-driven flags
USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)
CRF_MAX_CONFIGS = getattr(cfg, "CRF_MAX_CONFIGS", 64)


def init_model(model_name: str):
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

    t0_main = time_start()
    model_name = cfg.MODEL_NAME

    # ------------------------------------------------------------
    # Init DINOv3 model & processor
    # ------------------------------------------------------------
    model, processor, device = init_model(
        model_name,     # name of pretrained DINOv3 model (e.g. "facebook/dinov3-large")
    )

    # ------------------------------------------------------------
    # Resolve paths to imagery + SH_2022 + GT vector labels
    # ------------------------------------------------------------
    img_path = cfg.IMG_PATH
    img2_path = cfg.IMG2_PATH
    lab_path = cfg.LAB_PATH
    gt_vector_path = cfg.GT_VECTOR_PATH

    # ------------------------------------------------------------
    # Load imagery and reproject labels
    # ------------------------------------------------------------
    t0_data = time_start()
    img = load_dop20_image(
        img_path        # source path for DOP20 orthophoto A (bank image)
    )
    labels_A = reproject_labels_to_image(
        img_path,       # image A path (target CRS/grid)
        lab_path        # SH_2022 raster to reproject
    )

    img_b = load_dop20_image(
        img2_path       # target image B for inference
    )
    labels_SH_B = reproject_labels_to_image(
        img2_path,      # image B path
        lab_path        # SH_2022 raster
    )

    gt_mask_B = rasterize_vector_labels(
        gt_vector_path, # vector ground truth polygons for B
        img2_path       # raster reference to match CRS/resolution
    )
    time_end("data_loading_and_reprojection", t0_data)

    print(f"[debug] GT positives on B: {gt_mask_B.sum()}")
    print(f"[debug] SH_2022 positives on B: {(labels_SH_B > 0).sum()}")

    # ------------------------------------------------------------
    # Build SH_2022 buffer (spatial prior)
    # ------------------------------------------------------------
    with __import__('rasterio').open(img2_path) as src:
        pixel_size_m = abs(src.transform.a)

    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    print(f"[info] pixel_size={pixel_size_m:.3f} m, buffer_m={buffer_m}, buffer_pixels={buffer_pixels}")

    sh_buffer_mask_B = build_sh_buffer_mask(
        labels_SH_B,    # SH_2022 label raster on B
        buffer_pixels   # radius in pixel units for dilation
    )

    if getattr(cfg, "CLIP_GT_TO_BUFFER", False):
        gt_mask_eval = np.logical_and(gt_mask_B, sh_buffer_mask_B)
        print(f"[info] CLIP_GT_TO_BUFFER enabled: GT positives -> {gt_mask_eval.sum()} (was {gt_mask_B.sum()})")
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

    image_id_a = os.path.splitext(os.path.basename(img_path))[0]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    # ------------------------------------------------------------
    # Build DINOv3 positive/negative banks on Image A
    # ------------------------------------------------------------
    pos_bank, neg_bank = build_banks_single_scale(
        img,                    # img_a: RGB array of image A
        labels_A,               # labels_a: SH_2022 labels on A
        model,                  # DINO model for patch embeddings
        processor,              # processor for preprocessing
        device,                 # GPU/CPU device
        model.config.patch_size,  # ps: ViT patch size (16)
        1024,                   # tile_size: patch extraction block
        512,                    # stride: tile overlap
        0.1,                    # pos_frac_thresh: fraction of FG per patch
        None,                   # aggregate_layers: None → default layer
        feature_dir,            # feature_dir: store tile-level embeddings
        image_id_a,             # unique ID for caching A’s tiles
        cfg.BANK_CACHE_DIR      # bank cache directory (reuses banks)
    )

    # ------------------------------------------------------------
    # Prefetch DINO features for Image B (so kNN grid search is fast)
    # ------------------------------------------------------------
    prefetched_b = prefetch_features_single_scale_image(
        img_b,                  # img_hw3: RGB array of image B
        model,                  # DINO model
        processor,              # processor
        device,                 # GPU/CPU
        model.config.patch_size,# ps: ViT patch size
        1024,                   # tile_size
        512,                    # stride
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
        model.config.patch_size, # ps
        1024,                  # tile_size
        512,                   # stride
        cfg.K_VALUES,          # list of k-values to try (e.g. [1,3,5,7])
        cfg.THRESHOLDS,        # global thresholds for FG mask
        feature_dir,           # directory for B/features
        image_id_b,            # B tile cache ID
        sh_buffer_mask_B,      # spatial prior mask
        gt_mask_eval,          # ground-truth for scoring
        prefetched_b,          # cached DINO features for B
        USE_FP16_KNN           # use half precision matmul for speed
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

    champion_config = best_raw_config
    champion_score_full = best_raw_score_full
    thr_center_for_crf = champion_config["threshold"]
    k_center_for_crf = champion_config["k"]

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
        print(
            f"[crf-upsampled] IoU={metrics_crf_full['iou']:.3f}, "
            f"F1={metrics_crf_full['f1']:.3f}, "
            f"P={metrics_crf_full['precision']:.3f}, "
            f"R={metrics_crf_full['recall']:.3f}"
        )

    # ------------------------------------------------------------
    # Shadow filtering stage
    # ------------------------------------------------------------
    shadow_cfg, shadow_mask = shadow_filter_grid(
        img_b,                                 # RGB B
        best_crf_mask,                         # mask after CRF
        gt_mask_B,                             # GT for scoring
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
    consolidate_features_for_image(
        feature_dir,        # directory containing tile features
        image_id_a          # ID for image A
    )
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
        img_path,           # path to A
        img2_path,          # path to B
        buffer_m,           # SH buffer in meters
        pixel_size_m        # pixel spacing in meters
    )

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()

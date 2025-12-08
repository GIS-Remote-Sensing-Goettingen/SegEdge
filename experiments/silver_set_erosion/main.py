# SPDX-License-Identifier: GPL-2.0
# Orchestrator for the DINOv3 zero-shot pipeline (kNN + CRF).

import os
import numpy as np
import torch

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


def fine_tune_threshold(score_map: np.ndarray,
                        base_threshold: float,
                        sh_mask: np.ndarray | None,
                        gt_mask: np.ndarray,
                        step: float = 0.01,
                        window: float = 0.08):
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

    print(
        f"[tune-thr] base={base_threshold:.3f} -> best={best_thr:.3f} "
        f"IoU={best_metrics['iou']:.3f}, F1={best_metrics['f1']:.3f}"
    )
    time_end("fine_tune_threshold", t0)
    return best_thr, best_metrics, best_mask


def save_plot(img_b, gt_mask_B, mask_raw_best, best_raw_config, best_crf_mask, best_crf_config, thr_center_for_crf, plot_dir, image_id_b):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("Image B (RGB)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt_mask_B > 0, cmap="gray")
    axs[0, 1].set_title("Ground truth (labels_final)")
    axs[0, 1].axis("off")

    overlay_raw = img_b.copy()
    overlay_raw[mask_raw_best] = (0.5 * overlay_raw[mask_raw_best] + 0.5 * np.array([0, 255, 0])).astype(overlay_raw.dtype)
    axs[1, 0].imshow(overlay_raw)
    axs[1, 0].set_title(
        f"Raw kNN (k={best_raw_config['k']}, thr={best_raw_config['threshold']:.3f})\n"
        f"IoU={best_raw_config['iou']:.3f}, F1={best_raw_config['f1']:.3f}"
    )
    axs[1, 0].axis("off")

    overlay_crf = img_b.copy()
    overlay_crf[best_crf_mask] = (0.5 * overlay_crf[best_crf_mask] + 0.5 * np.array([255, 0, 0])).astype(overlay_crf.dtype)
    axs[1, 1].imshow(overlay_crf)
    axs[1, 1].set_title(
        f"CRF (k={best_crf_config['k']}, center_thr={thr_center_for_crf:.3f})\n"
        f"IoU={best_crf_config['iou']:.3f}, F1={best_crf_config['f1']:.3f}"
    )
    axs[1, 1].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_raw_crf.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved to {plot_path}")


def export_best_settings(best_raw_config, best_crf_config, model_name, img_path, img2_path, buffer_m, pixel_size_m):
    best_settings = {
        "best_raw_config": best_raw_config,
        "best_crf_config": best_crf_config,
        "model_name": model_name,
        "img_a": img_path,
        "img_b": img2_path,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
    }
    os.makedirs(os.path.dirname(cfg.BEST_SETTINGS_PATH), exist_ok=True)
    with open(cfg.BEST_SETTINGS_PATH, "w", encoding="utf-8") as f:
        def _write_yaml(d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    f.write("  " * indent + f"{k}:\n")
                    _write_yaml(v, indent + 1)
                else:
                    f.write("  " * indent + f"{k}: {v}\n")
        _write_yaml(best_settings)
    print(f"[config] best settings written to {cfg.BEST_SETTINGS_PATH}")


def main():
    t0_main = time_start()
    model_name = cfg.MODEL_NAME
    model, processor, device = init_model(model_name)

    # Paths
    img_path = cfg.IMG_PATH
    img2_path = cfg.IMG2_PATH
    lab_path = cfg.LAB_PATH
    gt_vector_path = cfg.GT_VECTOR_PATH

    # Load data
    t0_data = time_start()
    img = load_dop20_image(img_path)
    labels_A = reproject_labels_to_image(img_path, lab_path)
    img_b = load_dop20_image(img2_path)
    labels_SH_B = reproject_labels_to_image(img2_path, lab_path)
    gt_mask_B = rasterize_vector_labels(gt_vector_path, img2_path)
    time_end("data_loading_and_reprojection", t0_data)
    print(f"[debug] GT positives on B: {gt_mask_B.sum()}")
    print(f"[debug] SH_2022 positives on B: {(labels_SH_B > 0).sum()}")

    # Buffer
    with __import__('rasterio').open(img2_path) as src:
        pixel_size_m = abs(src.transform.a)
    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    print(f"[info] pixel_size={pixel_size_m:.3f} m, buffer_m={buffer_m}, buffer_pixels={buffer_pixels}")
    sh_buffer_mask_B = build_sh_buffer_mask(labels_SH_B, buffer_pixels)
    _ = compute_oracle_upper_bound(gt_mask_B, sh_buffer_mask_B)

    feature_dir = cfg.FEATURE_DIR
    os.makedirs(feature_dir, exist_ok=True)
    image_id_a = os.path.splitext(os.path.basename(img_path))[0]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    # Banks
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
        bank_cache_dir=cfg.BANK_CACHE_DIR,
    )

    # Prefetch B features
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

    # kNN grid search
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
        k_values=cfg.K_VALUES,
        thresholds=cfg.THRESHOLDS,
        feature_dir=feature_dir,
        image_id_b=image_id_b,
        sh_buffer_mask_b=sh_buffer_mask_B,
        gt_mask_b=gt_mask_B,
        prefetched_tiles_b=prefetched_b,
        use_fp16_matmul=USE_FP16_KNN,
    )

    # Fine-tune threshold
    thr_best_raw_refined, metrics_raw_refined, mask_raw_best = fine_tune_threshold(
        score_map=best_raw_score_full,
        base_threshold=best_raw_config["threshold"],
        sh_mask=sh_buffer_mask_B,
        gt_mask=gt_mask_B,
    )
    if metrics_raw_refined["iou"] >= best_raw_config["iou"]:
        best_raw_config = {**best_raw_config, "threshold": thr_best_raw_refined, **metrics_raw_refined}
    else:
        mask_raw_best = best_raw_score_full >= best_raw_config["threshold"]
        mask_raw_best = np.logical_and(mask_raw_best, sh_buffer_mask_B)

    champion_config = best_raw_config
    champion_score_full = best_raw_score_full
    thr_center_for_crf = champion_config["threshold"]
    k_center_for_crf = champion_config["k"]

    # CRF search
    best_crf_cfg_inner, best_crf_mask = crf_grid_search(
        img_rgb=img_b,
        score_map=champion_score_full,
        threshold_center=thr_center_for_crf,
        sh_mask=sh_buffer_mask_B,
        gt_mask=gt_mask_B,
        prob_softness_vals=cfg.PROB_SOFTNESS_VALUES,
        pos_w_vals=cfg.POS_W_VALUES,
        pos_xy_std_vals=cfg.POS_XY_STD_VALUES,
        bilateral_w_vals=cfg.BILATERAL_W_VALUES,
        bilateral_xy_std_vals=cfg.BILATERAL_XY_STD_VALUES,
        bilateral_rgb_std_vals=cfg.BILATERAL_RGB_STD_VALUES,
        n_iters=5,
        max_configs=CRF_MAX_CONFIGS,
        downsample_factor=2,
        num_workers=getattr(cfg, "CRF_NUM_WORKERS", 8),
        backend="process",
    )

    best_crf_config = {"k": k_center_for_crf, **best_crf_cfg_inner}
    if best_crf_mask.shape != img_b.shape[:2]:
        best_crf_mask_full = resize(best_crf_mask.astype(np.float32), (img_b.shape[0], img_b.shape[1]), order=0, preserve_range=True, anti_aliasing=False) > 0.5
        best_crf_mask = best_crf_mask_full
        metrics_crf_full = compute_metrics(best_crf_mask, gt_mask_B)
        print(
            f"[crf-upsampled] IoU={metrics_crf_full['iou']:.3f}, "
            f"F1={metrics_crf_full['f1']:.3f}, "
            f"P={metrics_crf_full['precision']:.3f}, R={metrics_crf_full['recall']:.3f}"
        )

    # Shadow filtering
    shadow_cfg, shadow_mask = shadow_filter_grid(
        img_rgb=img_b,
        base_mask=best_crf_mask,
        gt_mask=gt_mask_B,
        weight_sets=getattr(cfg, "SHADOW_WEIGHT_SETS", [(1.0, 1.0, 1.0)]),
        thresholds=getattr(cfg, "SHADOW_THRESHOLDS", [100]),
    )
    shadow_best = {"cfg": shadow_cfg, "mask": shadow_mask}

    # Plot
    save_plot(img_b, gt_mask_B, mask_raw_best, best_raw_config, best_crf_mask, best_crf_config, thr_center_for_crf, cfg.PLOT_DIR, image_id_b, best_shadow=shadow_best)

    # Shapefiles
    base_name_b = os.path.splitext(os.path.basename(img2_path))[0]
    out_dir_b = os.path.dirname(img2_path)
    export_mask_to_shapefile(mask_raw_best, img2_path, os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_raw.shp"))
    export_mask_to_shapefile(best_crf_mask, img2_path, os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_crf.shp"))
    export_mask_to_shapefile(shadow_mask, img2_path, os.path.join(out_dir_b, f"{base_name_b}_pred_mask_best_shadow.shp"))

    # Consolidate features
    consolidate_features_for_image(feature_dir, image_id_a)
    consolidate_features_for_image(feature_dir, image_id_b)

    export_best_settings(best_raw_config, best_crf_config, model_name, img_path, img2_path, buffer_m, pixel_size_m)
    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def save_plot(img_b, gt_mask_B, mask_raw_best, best_raw_config, best_crf_mask, best_crf_config, thr_center_for_crf, plot_dir, image_id_b, best_shadow=None, labels_sh=None):
    """Save comparison figure (RGB, GT, raw, CRF, optional shadow/labels) to plot_dir."""
    show_labels = labels_sh is not None
    show_shadow = best_shadow is not None
    cols = 3 if (show_labels or show_shadow) else 2
    fig, axs = plt.subplots(2, cols, figsize=(8 * cols, 12))
    if cols == 2 and axs.ndim == 1:
        axs = np.array([[axs[0], axs[1]], [axs[2], axs[3]]])
    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("Image B (RGB)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt_mask_B > 0, cmap="gray")
    axs[0, 1].set_title("Ground truth (union)")
    axs[0, 1].axis("off")

    if show_labels:
        axs[0, 2].imshow(labels_sh > 0, cmap="gray")
        axs[0, 2].set_title("SOURCE_LABEL_RASTER (reprojected)")
        axs[0, 2].axis("off")

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

    if show_shadow:
        shadow_mask = best_shadow["mask"]
        shadow_cfg = best_shadow["cfg"]
        overlay_shadow = img_b.copy()
        overlay_shadow[shadow_mask] = (
            0.5 * overlay_shadow[shadow_mask] + 0.5 * np.array([255, 255, 0])
        ).astype(overlay_shadow.dtype)
        axs[1, 2].imshow(overlay_shadow)
        axs[1, 2].set_title(
            f"Shadow filter w={shadow_cfg['weights']}, thr={shadow_cfg['threshold']}\n"
            f"IoU={shadow_cfg['iou']:.3f}, F1={shadow_cfg['f1']:.3f}"
        )
        axs[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_raw_crf.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved to %s", plot_path)


def save_best_model_plot(img_b, gt_mask, pred_mask, title, plot_dir, image_id_b, filename_suffix="champion.png"):
    """Save a simple overlay plot of the champion mask vs GT."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_b)
    axs[0].set_title("Image B (RGB)")
    axs[0].axis("off")

    axs[1].imshow(gt_mask.astype(bool), cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    overlay = img_b.copy()
    overlay[pred_mask] = (0.5 * overlay[pred_mask] + 0.5 * np.array([255, 0, 0])).astype(overlay.dtype)
    axs[2].imshow(overlay)
    axs[2].set_title(title)
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_{filename_suffix}")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved champion overlay to %s", plot_path)


def save_knn_xgb_gt_plot(img_b,
                         gt_mask,
                         mask_knn,
                         mask_xgb,
                         plot_dir,
                         image_id_b,
                         title_knn="kNN",
                         title_xgb="XGBoost",
                         filename_suffix="knn_xgb_gt.png"):
    """Save three panels: RGB+GT overlay, RGB+kNN overlay, RGB+XGB overlay."""
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # GT overlay
    overlay_gt = img_b.copy()
    overlay_gt[gt_mask.astype(bool)] = (
        0.5 * overlay_gt[gt_mask.astype(bool)] + 0.5 * np.array([0, 0, 255])
    ).astype(overlay_gt.dtype)
    axs[0].imshow(overlay_gt)
    axs[0].set_title("GT overlay")
    axs[0].axis("off")

    # kNN overlay
    overlay_knn = img_b.copy()
    overlay_knn[mask_knn.astype(bool)] = (
        0.5 * overlay_knn[mask_knn.astype(bool)] + 0.5 * np.array([0, 255, 0])
    ).astype(overlay_knn.dtype)
    axs[1].imshow(overlay_knn)
    axs[1].set_title(title_knn)
    axs[1].axis("off")

    # XGB overlay
    overlay_xgb = img_b.copy()
    overlay_xgb[mask_xgb.astype(bool)] = (
        0.5 * overlay_xgb[mask_xgb.astype(bool)] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay_xgb.dtype)
    axs[2].imshow(overlay_xgb)
    axs[2].set_title(title_xgb)
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_{filename_suffix}")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved kNN/XGB/GT overlay to %s", plot_path)

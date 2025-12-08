import os
import numpy as np
import matplotlib.pyplot as plt


def save_plot(img_b, gt_mask_B, mask_raw_best, best_raw_config, best_crf_mask, best_crf_config, thr_center_for_crf, plot_dir, image_id_b, best_shadow=None):
    """Save comparison figure (RGB, GT, raw, CRF, optional shadow) to plot_dir."""
    # Layout: if shadow provided, use 2x3; else 2x2
    if best_shadow is None:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(22, 12))
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

    if best_shadow is not None:
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
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved to {plot_path}")

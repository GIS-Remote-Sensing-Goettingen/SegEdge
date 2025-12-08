import numpy as np
from metrics_utils import compute_metrics
from timing_utils import time_start, time_end


def shadow_filter_grid(img_rgb: np.ndarray,
                       base_mask: np.ndarray,
                       gt_mask: np.ndarray,
                       weight_sets,
                       thresholds):
    """
    Filter out dark pixels under the mask using weighted RGB sums.
    Returns best config and filtered mask.
    """
    t0 = time_start()
    img_float = img_rgb.astype(np.float32)
    best_cfg = None
    best_mask = base_mask
    best_iou = -1.0

    for weights in weight_sets:
        w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
        wsum = (img_float * w).sum(axis=2)
        for thr in thresholds:
            filt_mask = np.logical_and(base_mask, wsum >= thr)
            metrics = compute_metrics(filt_mask, gt_mask)
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_cfg = {"weights": weights, "threshold": float(thr), **metrics}
                best_mask = filt_mask
    time_end("shadow_filter_grid", t0)
    return best_cfg, best_mask

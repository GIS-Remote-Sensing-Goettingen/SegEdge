from concurrent.futures import ProcessPoolExecutor
import numpy as np
import logging

from metrics_utils import compute_metrics
from timing_utils import time_start, time_end

logger = logging.getLogger(__name__)

def _shadow_filter_single_weights(img_float: np.ndarray,
                                  base_mask: np.ndarray,
                                  gt_mask: np.ndarray,
                                  weights,
                                  thresholds):
    """
    Evaluate all thresholds for a single weight triplet in a vectorized way.

    Args
    ----
    img_float : HxWx3 float32 image
        Preconverted image (no re-cast inside).
    base_mask : HxW bool
        Mask from CRF / raw prediction; only pixels under this mask may become FG.
    gt_mask   : HxW bool or {0,1}
        Ground-truth mask.
    weights   : (3,) iterable
        RGB weights for shadow detection.
    thresholds : list[float]
        Weighted-sum thresholds to test.

    Returns
    -------
    best_cfg  : dict or None
        The best config (for this weights only) with keys:
        {"weights", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "iou", "f1"}.
    best_mask : HxW bool
        Filtered mask for the best threshold for this weights.
    """
    # justification:
    # - This function is self-contained, so it's picklable for ProcessPoolExecutor.
    # - It handles all thresholds at once in a vectorized manner.

    # Convert masks to boolean arrays
    base_mask_bool = base_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)

    # Compute weighted sum once for this weight set
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    wsum = (img_float * w).sum(axis=2)  # (H, W)

    # Flatten everything and restrict to pixels under base_mask
    flat_base = base_mask_bool.reshape(-1)              # (N_total,)
    flat_gt = gt_mask_bool.reshape(-1)                  # (N_total,)
    flat_wsum = wsum.reshape(-1)                        # (N_total,)

    valid_idx = flat_base  # only where base_mask is True can we keep FG
    if not np.any(valid_idx):
        # No predicted positives at all; nothing to optimize
        return None, base_mask_bool

    vals = flat_wsum[valid_idx]         # (N_valid,)
    gt_vals = flat_gt[valid_idx]        # (N_valid,)

    thr_arr = np.array(thresholds, dtype=np.float32).reshape(-1, 1)  # (T, 1)

    # mask_thr[t, i] = 1 if vals[i] >= thr[t]
    mask_thr = vals[None, :] >= thr_arr  # (T, N_valid)

    # Booleans: we treat "True" as FG at that threshold
    gt_bool = gt_vals.astype(bool)  # (N_valid,)

    # Vectorized contingency counts
    tp = np.logical_and(mask_thr, gt_bool[None, :]).sum(axis=1).astype(np.float64)
    fp = np.logical_and(mask_thr, ~gt_bool[None, :]).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~mask_thr, gt_bool[None, :]).sum(axis=1).astype(np.float64)
    tn = np.logical_and(~mask_thr, ~gt_bool[None, :]).sum(axis=1).astype(np.float64)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    # Pick best threshold for this weights
    best_idx = int(np.argmax(iou))
    best_thr = float(thr_arr[best_idx, 0])
    best_metrics = {
        "tp": int(tp[best_idx]),
        "fp": int(fp[best_idx]),
        "fn": int(fn[best_idx]),
        "tn": int(tn[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "iou": float(iou[best_idx]),
        "f1": float(f1[best_idx]),
    }

    best_cfg = {
        "weights": tuple(float(x) for x in weights),
        "threshold": best_thr,
        **best_metrics,
    }

    # Rebuild best mask at full resolution
    best_mask_flat = np.zeros_like(flat_base, dtype=bool)
    # Under base_mask, keep those satisfying weighted-sum >= best_thr
    best_mask_flat[valid_idx] = vals >= best_thr
    best_mask = best_mask_flat.reshape(base_mask_bool.shape)

    return best_cfg, best_mask



def shadow_filter_grid(img_rgb: np.ndarray,
                       base_mask: np.ndarray,
                       gt_mask: np.ndarray,
                       weight_sets,
                       thresholds,
                       num_workers: int = 1):
    """
    Filter out dark pixels under the mask using weighted RGB sums.

    Vectorizes all thresholds for each weight-set and optionally parallelizes
    across weight sets. Returns the best (weights, threshold) and mask.
    """
    t0 = time_start()

    # Convert image once to float32
    img_float = img_rgb.astype(np.float32)

    # Ensure masks are boolean
    base_mask_bool = base_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)

    best_cfg_global = None
    best_mask_global = base_mask_bool.copy()
    best_iou_global = -1.0

    weight_sets_list = list(weight_sets)

    # --- single-process path (recommended for few weight sets) ---
    if num_workers <= 1 or len(weight_sets_list) == 1:
        for weights in weight_sets_list:
            cfg, mask = _shadow_filter_single_weights(
                img_float=img_float,
                base_mask=base_mask_bool,
                gt_mask=gt_mask_bool,
                weights=weights,
                thresholds=thresholds,
            )
            if cfg is None:
                continue

            if cfg["iou"] > best_iou_global:
                best_iou_global = cfg["iou"]
                best_cfg_global = cfg
                best_mask_global = mask

    # --- multi-process path (parallelize over weight sets) ---
    else:
        # justification:
        # - Parallelizing over weight_sets keeps each task coarse-grained,
        #   so process overhead is amortized.
        # - On Linux, large arrays are shared via copy-on-write (fork),
        #   which is relatively cheap unless you modify them.
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(
                    _shadow_filter_single_weights,
                    img_float,
                    base_mask_bool,
                    gt_mask_bool,
                    weights,
                    thresholds,
                )
                for weights in weight_sets_list
            ]
            for fut in futures:
                cfg, mask = fut.result()
                if cfg is None:
                    continue
                if cfg["iou"] > best_iou_global:
                    best_iou_global = cfg["iou"]
                    best_cfg_global = cfg
                    best_mask_global = mask

    #print stats
    logger.info("shadow_filter_grid best config: %s", best_cfg_global)

    time_end("shadow_filter_grid", t0)
    return best_cfg_global, best_mask_global

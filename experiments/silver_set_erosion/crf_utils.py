import numpy as np
from concurrent.futures import ProcessPoolExecutor
from skimage.transform import resize

from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from metrics_utils import compute_metrics
from timing_utils import time_start, time_end


def refine_with_densecrf(
    img_rgb: np.ndarray,
    score_map: np.ndarray,
    threshold_center: float,
    sh_mask: np.ndarray | None = None,
    prob_softness: float = 0.05,
    n_iters: int = 5,
    pos_w: float = 3.0,
    pos_xy_std: float = 3.0,
    bilateral_w: float = 5.0,
    bilateral_xy_std: float = 50.0,
    bilateral_rgb_std: float = 5.0,
) -> np.ndarray:
    """Run DenseCRF with a logistic unary centered at threshold_center; returns refined mask."""
    t0 = time_start()
    h, w, _ = img_rgb.shape
    assert score_map.shape == (h, w), "score_map must have shape (H, W)"
    s = score_map.astype(np.float32)
    logits_fg = (s - threshold_center) / prob_softness
    p_fg = 1.0 / (1.0 + np.exp(-logits_fg))
    eps = 1e-6
    p_fg = np.clip(p_fg, eps, 1.0 - eps)
    if sh_mask is not None:
        sh_mask = sh_mask.astype(bool)
        p_fg[~sh_mask] = eps
    p_bg = 1.0 - p_fg
    probs = np.stack([p_bg, p_fg], axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(
        sxy=(pos_xy_std, pos_xy_std),
        compat=pos_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    img_rgb_u8 = img_rgb.astype(np.uint8) if img_rgb.dtype != np.uint8 else img_rgb
    if not img_rgb_u8.flags["C_CONTIGUOUS"]:
        img_rgb_u8 = np.ascontiguousarray(img_rgb_u8)
    d.addPairwiseBilateral(
        sxy=(bilateral_xy_std, bilateral_xy_std),
        srgb=(bilateral_rgb_std, bilateral_rgb_std, bilateral_rgb_std),
        rgbim=img_rgb_u8,
        compat=bilateral_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    Q = d.inference(n_iters)
    Q = np.array(Q).reshape(2, h, w)
    labels = np.argmax(Q, axis=0).astype(np.uint8)
    refined_mask = labels == 1
    if sh_mask is not None:
        refined_mask = np.logical_and(refined_mask, sh_mask)
    time_end("refine_with_densecrf", t0)
    return refined_mask


def _crf_eval_worker(args):
    """Process-pool helper to evaluate one CRF config; returns metrics and config dict."""
    (img_rgb_ds, score_map_ds, sh_mask_ds, gt_mask_ds, threshold_center, n_iters, cfg) = args
    prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
    mask_crf_local = refine_with_densecrf(
        img_rgb_ds,
        score_map_ds,
        threshold_center,
        sh_mask_ds,
        prob_softness=prob_soft,
        n_iters=n_iters,
        pos_w=pos_w,
        pos_xy_std=pos_xy,
        bilateral_w=bi_w,
        bilateral_xy_std=bi_xy,
        bilateral_rgb_std=bi_rgb,
    )
    metrics_local = compute_metrics(mask_crf_local, gt_mask_ds)
    return metrics_local, {
        "prob_softness": prob_soft,
        "pos_w": pos_w,
        "pos_xy_std": pos_xy,
        "bilateral_w": bi_w,
        "bilateral_xy_std": bi_xy,
        "bilateral_rgb_std": bi_rgb,
        **metrics_local,
    }


def crf_grid_search(
    img_rgb: np.ndarray,
    score_map: np.ndarray,
    threshold_center: float,
    sh_mask: np.ndarray,
    gt_mask: np.ndarray,
    prob_softness_vals,
    pos_w_vals,
    pos_xy_std_vals,
    bilateral_w_vals,
    bilateral_xy_std_vals,
    bilateral_rgb_std_vals,
    n_iters: int = 5,
    max_configs: int | None = None,
    downsample_factor: int = 1,
    num_workers: int = 1,
    backend: str = "process",
):
    """Small grid search over CRF hyperparameters; optional downsampling + multiprocessing."""
    t0 = time_start()
    best_cfg = None
    best_mask = None
    best_iou = -1.0

    if downsample_factor > 1:
        img_rgb_ds = resize(img_rgb, (img_rgb.shape[0] // downsample_factor, img_rgb.shape[1] // downsample_factor),
                            order=1, preserve_range=True, anti_aliasing=True).astype(img_rgb.dtype)
        score_map_ds = resize(score_map, (score_map.shape[0] // downsample_factor, score_map.shape[1] // downsample_factor),
                              order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        sh_mask_ds = resize(sh_mask.astype(np.float32),
                            (sh_mask.shape[0] // downsample_factor, sh_mask.shape[1] // downsample_factor),
                            order=0, preserve_range=True, anti_aliasing=False) > 0.5
        gt_mask_ds = resize(gt_mask.astype(np.float32),
                            (gt_mask.shape[0] // downsample_factor, gt_mask.shape[1] // downsample_factor),
                            order=0, preserve_range=True, anti_aliasing=False) > 0.5
    else:
        img_rgb_ds = img_rgb
        score_map_ds = score_map
        sh_mask_ds = sh_mask
        gt_mask_ds = gt_mask

    cfg_list = []
    for prob_soft in prob_softness_vals:
        for pos_w in pos_w_vals:
            for pos_xy in pos_xy_std_vals:
                for bi_w in bilateral_w_vals:
                    for bi_xy in bilateral_xy_std_vals:
                        for bi_rgb in bilateral_rgb_std_vals:
                            cfg_list.append((prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb))
    if max_configs is not None:
        cfg_list = cfg_list[:max_configs]

    if num_workers > 1 and backend == "process":
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            args_iter = [(img_rgb_ds, score_map_ds, sh_mask_ds, gt_mask_ds, threshold_center, n_iters, cfg)
                         for cfg in cfg_list]
            for metrics, cfg_full in ex.map(_crf_eval_worker, args_iter):
                if metrics["iou"] > best_iou:
                    best_iou = metrics["iou"]
                    best_cfg = cfg_full
    else:
        for cfg in cfg_list:
            prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
            t_cfg = time_start()
            mask_crf_local = refine_with_densecrf(
                img_rgb=img_rgb_ds,
                score_map=score_map_ds,
                threshold_center=threshold_center,
                sh_mask=sh_mask_ds,
                prob_softness=prob_soft,
                n_iters=n_iters,
                pos_w=pos_w,
                pos_xy_std=pos_xy,
                bilateral_w=bi_w,
                bilateral_xy_std=bi_xy,
                bilateral_rgb_std=bi_rgb,
            )
            time_end("crf_single_config", t_cfg)
            metrics_local = compute_metrics(mask_crf_local, gt_mask_ds)
            if metrics_local["iou"] > best_iou:
                best_iou = metrics_local["iou"]
                best_cfg = {
                    "prob_softness": prob_soft,
                    "pos_w": pos_w,
                    "pos_xy_std": pos_xy,
                    "bilateral_w": bi_w,
                    "bilateral_xy_std": bi_xy,
                    "bilateral_rgb_std": bi_rgb,
                    **metrics_local,
                }
                best_mask = mask_crf_local

    if best_cfg is not None and best_mask is None:
        best_mask = refine_with_densecrf(
            img_rgb=img_rgb_ds,
            score_map=score_map_ds,
            threshold_center=threshold_center,
            sh_mask=sh_mask_ds,
            prob_softness=best_cfg["prob_softness"],
            n_iters=n_iters,
            pos_w=best_cfg["pos_w"],
            pos_xy_std=best_cfg["pos_xy_std"],
            bilateral_w=best_cfg["bilateral_w"],
            bilateral_xy_std=best_cfg["bilateral_xy_std"],
            bilateral_rgb_std=best_cfg["bilateral_rgb_std"],
        )
    time_end("crf_grid_search", t0)
    return best_cfg, best_mask

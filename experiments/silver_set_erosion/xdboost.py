import os
import numpy as np
import xgboost as xgb
from skimage.transform import resize

from features import tile_iterator, crop_to_multiple_of_ps, labels_to_patch_masks, tile_feature_path
from metrics_utils import compute_metrics_batch_cpu, compute_metrics_batch_gpu


def build_xgb_dataset(img, labels, ps, tile_size, stride, feature_dir, image_id, pos_frac, max_neg=8000):
    """
    Builds a dataset for XGBoost by iterating over image tiles, extracting features,
    and creating positive/negative samples based on labels.

    Parameters:
    - img (numpy.ndarray): The input image.
    - labels (numpy.ndarray): The label image.
    - ps (int): Patch size for cropping.
    - tile_size (int): Size of each tile.
    - stride (int): Stride for tiling.
    - feature_dir (str): Directory containing precomputed features.
    - image_id (str): Identifier for the image.
    - pos_frac (float): Fraction threshold for positive patches.
    - max_neg (int, optional): Maximum number of negative samples (default: 8000).

    Returns:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Label array (1 for positive, 0 for negative).
    """
    X_pos, X_neg = [], []
    for y, x, img_tile, lab_tile in tile_iterator(img, labels, tile_size, stride):
        img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, lab_tile, ps)
        if h_eff < ps or w_eff < ps:
            continue
        fpath = tile_feature_path(feature_dir, image_id, y, x)
        feats_tile = np.load(fpath)  # already cached from earlier stages
        hp, wp = feats_tile.shape[:2]
        pos_mask, neg_mask = labels_to_patch_masks(lab_c, hp, wp, pos_frac_thresh=pos_frac)
        if pos_mask.any():
            X_pos.append(feats_tile[pos_mask])
        if neg_mask.any():
            X_neg.append(feats_tile[neg_mask])

    X_pos = np.concatenate(X_pos, axis=0)
    X_neg = np.concatenate(X_neg, axis=0) if X_neg else np.empty((0, X_pos.shape[1]))
    # Subsample negatives
    if len(X_neg) > max_neg:
        idx = np.random.default_rng(42).choice(len(X_neg), size=max_neg, replace=False)
        X_neg = X_neg[idx]

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    return X, y


import xgboost as xgb


def train_xgb_classifier(X, y, use_gpu=False, num_boost_round=300, verbose_eval=50):
    """
    Train a binary XGBoost classifier; defaults tuned for high-dimensional embeddings.
    """
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    try:
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtrain, "train")], verbose_eval=verbose_eval)
    except xgb.core.XGBoostError as e:
        if use_gpu and "gpu_hist" in str(e):
            print("[warn] xgboost build does not support GPU; falling back to CPU hist.")
            params["tree_method"] = "hist"
            bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtrain, "train")], verbose_eval=verbose_eval)
        else:
            raise
    return bst


def hyperparam_search_xgb(X,
                          y,
                          use_gpu=False,
                          param_grid=None,
                          num_boost_round=300,
                          val_fraction=0.2,
                          early_stopping_rounds=40,
                          verbose_eval=50,
                          seed: int = 42):
    """
    Hyperparameter search over a list of parameter overrides; picks best by val logloss.
    """
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    if param_grid is None:
        param_grid = [base_params]

    best_model = None
    best_params = None
    best_score = float("inf")  # logloss: lower is better

    for i, overrides in enumerate(param_grid):
        params = base_params.copy()
        params.update(overrides)
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        print(f"[xgb-search] cfg {i+1}/{len(param_grid)}: {params}")
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
        except xgb.core.XGBoostError as e:
            if use_gpu and "gpu_hist" in str(e):
                print("[warn] xgboost build lacks GPU; retrying with CPU hist.")
                params["tree_method"] = "hist"
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
            else:
                raise

        score = float(model.best_score) if hasattr(model, "best_score") else model.evals_result()["val"]["logloss"][-1]
        best_ntree = getattr(model, "best_ntree_limit", getattr(model, "best_iteration", None))
        print(f"[xgb-search] cfg {i+1}: val_logloss={score:.4f} (best_ntree={best_ntree})")

        if score < best_score:
            best_score = score
            best_model = model
            best_params = params.copy()

    return best_model, best_params, best_score


def hyperparam_search_xgb_iou(X,
                              y,
                              thresholds,
                              sh_buffer_mask_b,
                              gt_mask_b,
                              img_b,
                              ps,
                              tile_size,
                              stride,
                              feature_dir,
                              image_id_b,
                              prefetched_tiles=None,
                              device=None,
                              use_gpu=False,
                              param_grid=None,
                              num_boost_round=300,
                              val_fraction=0.2,
                              early_stopping_rounds=40,
                              verbose_eval=50,
                              seed: int = 42):
    """
    Hyperparameter search where the selection metric is IoU on Image B:
    train on Image A patches, score on B, sweep thresholds, pick max IoU.
    """
    best_model = None
    best_params = None
    best_iou = -1.0
    best_metrics = None
    best_thr = None

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    if param_grid is None:
        param_grid = [base_params]

    # simple train/val split (used only for early stopping)
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    for i, overrides in enumerate(param_grid):
        params = base_params.copy()
        params.update(overrides)
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        print(f"[xgb-search-iou] cfg {i+1}/{len(param_grid)}: {params}")
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
        except xgb.core.XGBoostError as e:
            if use_gpu and "gpu_hist" in str(e):
                print("[warn] xgboost build lacks GPU; retrying with CPU hist.")
                params["tree_method"] = "hist"
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
            else:
                raise

        # Evaluate IoU on Image B
        score_full_xgb = xgb_score_image_b(
            img_b,
            model,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_b,
            prefetched_tiles=prefetched_tiles,
        )
        try:
            metrics_list = compute_metrics_batch_gpu(
                score_full_xgb,
                thresholds,
                sh_buffer_mask_b,
                gt_mask_b,
                device=device,
            )
        except Exception:
            metrics_list = compute_metrics_batch_cpu(
                score_full_xgb,
                thresholds,
                sh_buffer_mask_b,
                gt_mask_b,
            )

        best_local = max(metrics_list, key=lambda m: m["iou"])
        print(
            f"[xgb-search-iou] cfg {i+1}: thr={best_local['threshold']:.3f}, IoU={best_local['iou']:.3f}, "
            f"F1={best_local['f1']:.3f}"
        )

        if best_local["iou"] > best_iou:
            best_iou = best_local["iou"]
            best_thr = best_local["threshold"]
            best_metrics = best_local
            best_params = params.copy()
            best_model = model

    return best_model, best_params, best_iou, best_thr, best_metrics


def xgb_score_image_b(img_b, bst, ps, tile_size, stride, feature_dir, image_id_b, prefetched_tiles=None):
    """
    Apply a trained XGBoost model to Image B using cached or prefetched features; returns score map.
    """
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    tile_iter = (
        sorted(prefetched_tiles.items()) if prefetched_tiles is not None
        else tile_iterator(img_b, None, tile_size, stride)
    )

    for tile_entry in tile_iter:
        if prefetched_tiles is not None:
            (y, x), feat_info = tile_entry
            feats_tile = feat_info["feats"]
            h_eff = feat_info["h_eff"]
            w_eff = feat_info["w_eff"]
            hp = feat_info["hp"]
            wp = feat_info["wp"]
        else:
            y, x, img_tile, _ = tile_entry
            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
            if h_eff < ps or w_eff < ps:
                continue
            fpath = tile_feature_path(feature_dir, image_id_b, y, x)
            if not os.path.exists(fpath):
                continue
            feats_tile = np.load(fpath)
            hp, wp = feats_tile.shape[:2]

        dtest = xgb.DMatrix(feats_tile.reshape(-1, feats_tile.shape[-1]))
        scores_patch = bst.predict(dtest).reshape(hp, wp)
        scores_tile = resize(scores_patch, (h_eff, w_eff), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        score_full[y:y + h_eff, x:x + w_eff] += scores_tile
        weight_full[y:y + h_eff, x:x + w_eff] += 1.0

    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    return score_full

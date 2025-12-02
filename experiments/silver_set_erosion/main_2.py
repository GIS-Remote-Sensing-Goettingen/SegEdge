# SPDX-License-Identifier: GPL-2.0
#
# Variant pipeline: replaces kNN with a small CNN head trained on Image A.
# Input to CNN per patch = concatenated DINOv3 patch features + pooled RGB (per patch).
# Output = per-patch probability map, upsampled to pixel grid; rest mirrors main.py.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from skimage.transform import resize

from main import (
    time_start,
    time_end,
    DEBUG_TIMING,
    init_model,
    load_dop20_image,
    reproject_labels_to_image,
    rasterize_vector_labels,
    tile_iterator,
    crop_to_multiple_of_ps,
    labels_to_patch_masks,
    extract_patch_features_single_scale,
    tile_feature_path,
    save_tile_features,
    build_sh_buffer_mask,
    compute_metrics,
    compute_oracle_upper_bound,
    compute_metrics_batch_gpu,
    compute_metrics_batch_cpu,
    refine_with_densecrf,
    crf_grid_search,
    export_mask_to_shapefile,
    consolidate_features_for_image,
)


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------

def patch_rgb_means(img_c: np.ndarray, hp: int, wp: int, ps: int) -> np.ndarray:
    img_float = img_c.astype(np.float32) / 255.0
    rgb_patches = img_float.reshape(hp, ps, wp, ps, 3)
    return rgb_patches.mean(axis=(1, 3))  # (hp, wp, 3)


def build_train_tiles_with_labels(img_a: np.ndarray,
                                  labels_a: np.ndarray,
                                  model,
                                  processor,
                                  device,
                                  ps: int = 16,
                                  tile_size: int = 1024,
                                  stride: int | None = None,
                                  pos_frac_thresh: float = 0.1,
                                  aggregate_layers=None,
                                  feature_dir: str | None = None,
                                  image_id: str | None = None):
    t0 = time_start()
    inputs_dino = []
    inputs_rgb = []
    targets = []
    cached_tiles = computed_tiles = 0

    labels_eroded = labels_a > 0

    for y, x, img_tile, lab_tile in tile_iterator(img_a, labels_eroded, tile_size, stride):
        img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, lab_tile, ps)
        if h_eff < ps or w_eff < ps:
            continue

        feats_tile = None
        hp = wp = None
        if feature_dir is not None and image_id is not None:
            fpath = tile_feature_path(feature_dir, image_id, y, x)
            if os.path.exists(fpath):
                feats_tile = np.load(fpath)
                hp, wp = feats_tile.shape[:2]
                cached_tiles += 1

        if feats_tile is None:
            feats_tile, hp, wp = extract_patch_features_single_scale(
                img_c,
                model,
                processor,
                device,
                ps=ps,
                aggregate_layers=aggregate_layers,
            )
            computed_tiles += 1
            if feature_dir is not None and image_id is not None:
                save_tile_features(feats_tile, feature_dir, image_id, y, x)

        pos_mask, neg_mask = labels_to_patch_masks(lab_c, hp, wp, pos_frac_thresh=pos_frac_thresh)
        label_grid = np.full((hp, wp), -1, dtype=np.int8)
        label_grid[pos_mask] = 1
        label_grid[neg_mask] = 0
        if np.all(label_grid == -1):
            continue  # ignore tiles without confident labels

        inputs_dino.append(feats_tile)  # (hp, wp, C)
        inputs_rgb.append(img_c.astype(np.float32) / 255.0)  # (H, W, 3)
        targets.append(label_grid)

    time_end("build_train_tiles_with_labels", t0)
    print(f"[train-tiles] count={len(inputs_rgb)} cached={cached_tiles} computed={computed_tiles}")
    return inputs_dino, inputs_rgb, targets


class RefineNet(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 96, patch_size: int = 16):
        super().__init__()
        self.ps = patch_size
        self.sem_conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.refine_conv = nn.Sequential(
            nn.Conv2d(hidden_ch + 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x_dino, x_rgb):
        feat = self.sem_conv(x_dino)  # low-res semantic
        feat_up = F.interpolate(feat, scale_factor=self.ps, mode="bilinear", align_corners=False)
        if feat_up.shape[-2:] != x_rgb.shape[-2:]:
            feat_up = F.interpolate(feat_up, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([feat_up, x_rgb], dim=1)
        logits = self.refine_conv(combined)
        return logits


def focal_loss(inputs, targets, alpha=0.75, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    loss = alpha * (1 - pt) ** gamma * bce_loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def train_cnn_head(train_inputs_dino,
                   train_inputs_rgb,
                   train_labels,
                   in_ch: int,
                   device,
                   patch_size: int,
                   hidden_ch: int = 96,
                   epochs: int = 20,
                   lr: float = 1e-3,
                   batch_size: int = 1):
    t0 = time_start()
    t_dino = []
    t_rgb = []
    t_lab = []
    max_hp = max_wp = 0
    max_hpx = max_wpx = 0
    for dino, rgb, lab in zip(train_inputs_dino, train_inputs_rgb, train_labels):
        if (lab >= 0).sum() == 0:
            continue
        td = torch.from_numpy(dino.transpose(2, 0, 1)).float()  # C, Hp, Wp
        tr = torch.from_numpy(rgb.transpose(2, 0, 1)).float()   # 3, H, W
        tl = torch.from_numpy(lab).float()                      # Hp, Wp
        max_hp = max(max_hp, td.shape[1]); max_wp = max(max_wp, td.shape[2])
        max_hpx = max(max_hpx, tr.shape[1]); max_wpx = max(max_wpx, tr.shape[2])
        t_dino.append(td); t_rgb.append(tr); t_lab.append(tl)

    def pad_tensor_3d(t, mh, mw, val=0.0):
        # t: C,H,W
        _, h, w = t.shape
        return F.pad(t, (0, mw - w, 0, mh - h), value=val)

    def pad_tensor_2d(t, mh, mw, val=0.0):
        # t: H,W
        h, w = t.shape
        return F.pad(t, (0, mw - w, 0, mh - h), value=val)

    dino_batch = torch.stack([pad_tensor_3d(t, max_hp, max_wp) for t in t_dino])
    rgb_batch = torch.stack([pad_tensor_3d(t, max_hpx, max_wpx) for t in t_rgb])
    lab_batch = torch.stack([pad_tensor_2d(t, max_hp, max_wp, val=-1.0) for t in t_lab])

    ds = TensorDataset(dino_batch, rgb_batch, lab_batch)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = RefineNet(in_ch=in_ch, hidden_ch=hidden_ch, patch_size=patch_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    print(f"[cnn] Training on {len(ds)} tiles...")
    for ep in range(epochs):
        ep_loss = 0.0
        count = 0
        for bd, br, bl in dl:
            bd, br, bl = bd.to(device), br.to(device), bl.to(device)
            if torch.rand(1).item() < 0.5:
                bd = torch.flip(bd, dims=[3]); br = torch.flip(br, dims=[3]); bl = torch.flip(bl, dims=[2])
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(bd, br).squeeze(1)  # pixel-res logits
                targets_up = F.interpolate(bl.unsqueeze(1), size=logits.shape[-2:], mode="nearest").squeeze(1)
                mask = targets_up >= 0
                if not mask.any():
                    continue
                l_focal = focal_loss(logits[mask], targets_up[mask], alpha=0.75, gamma=2.0)
                probs = torch.sigmoid(logits[mask])
                y_true = targets_up[mask]
                intersection = (probs * y_true).sum()
                dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum() + y_true.sum() + 1e-6)
                loss = 0.5 * l_focal + 0.5 * dice
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            count += 1
        print(f"[cnn] epoch {ep+1}/{epochs} loss={ep_loss / max(1, count):.4f}")

    time_end("train_cnn_head", t0)
    model.eval()
    return model


def cnn_inference_image(img_b: np.ndarray,
                        cnn_model: nn.Module,
                        processor,
                        dino_model,
                        device,
                        ps: int = 16,
                        tile_size: int = 1024,
                        stride: int | None = None,
                        feature_dir: str | None = None,
                        image_id: str | None = None):
    t0 = time_start()
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride):
        t_tile = time_start()
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
        if h_eff < ps or w_eff < ps:
            continue

        feats_tile = None
        hp = wp = None
        if feature_dir is not None and image_id is not None:
            fpath = tile_feature_path(feature_dir, image_id, y, x)
            if os.path.exists(fpath):
                feats_tile = np.load(fpath)
                hp, wp = feats_tile.shape[:2]

        if feats_tile is None:
            feats_tile, hp, wp = extract_patch_features_single_scale(
                img_c,
                dino_model,
                processor,
                device,
                ps=ps,
                aggregate_layers=None,
            )
            if feature_dir is not None and image_id is not None:
                save_tile_features(feats_tile, feature_dir, image_id, y, x)

        rgb_means = patch_rgb_means(img_c, hp, wp, ps)
        x_dino = torch.from_numpy(feats_tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        img_float = img_c.astype(np.float32) / 255.0
        x_rgb = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = cnn_model(x_dino, x_rgb)  # (1,1,Hpx,Wpx)
            score_tile = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

        score_full[y:y + h_eff, x:x + w_eff] += score_tile
        weight_full[y:y + h_eff, x:x + w_eff] += 1.0
        time_end(f"cnn_tile(y={y},x={x})", t_tile)

    mask = weight_full > 0
    score_full[mask] /= weight_full[mask]
    time_end("cnn_inference_image", t0)
    return score_full


def grid_search_thresholds(score_map: np.ndarray,
                           thresholds: list[float],
                           sh_mask: np.ndarray,
                           gt_mask: np.ndarray,
                           device: torch.device):
    if device.type == "cuda":
        metrics_list = compute_metrics_batch_gpu(score_map, thresholds, sh_mask, gt_mask, device=device)
    else:
        metrics_list = compute_metrics_batch_cpu(score_map, thresholds, sh_mask, gt_mask, batch_size=32)

    best_cfg = None
    best_iou = -1.0
    for m in metrics_list:
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_cfg = {"threshold": m["threshold"], **m}
    return best_cfg


def main():
    t0_main = time_start()
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    dino_model, processor, device = init_model(model_name)

    img_path = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/dop20_593000_5979000_1km_20cm.tif"
    img2_path = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/dop20_592000_5982000_1km_20cm.tif"
    lab_path = "/mnt/nvme1n1p5/SH_dataset/planet_labels_2022.tif"
    gt_vector_path = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/labels_final.shp"

    t0_data = time_start()
    img = load_dop20_image(img_path)
    labels_A = reproject_labels_to_image(img_path, lab_path)

    img_b = load_dop20_image(img2_path)
    labels_SH_B = reproject_labels_to_image(img2_path, lab_path)
    gt_mask_B = rasterize_vector_labels(gt_vector_path, img2_path)
    time_end("data_loading_and_reprojection", t0_data)

    # Buffer
    import rasterio
    with rasterio.open(img2_path) as src:
        pixel_size_m = abs(src.transform.a)
    buffer_pixels = int(round(8.0 / pixel_size_m))
    sh_buffer_mask_B = build_sh_buffer_mask(labels_SH_B, buffer_pixels)
    _ = compute_oracle_upper_bound(gt_mask_B, sh_buffer_mask_B)

    feature_dir = os.path.join(os.path.dirname(img_path), "dino_features")
    image_id_a = os.path.splitext(os.path.basename(img_path))[0]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    # Build training tiles (patch-level) and train CNN
    train_inputs_dino, train_inputs_rgb, train_labels = build_train_tiles_with_labels(
        img_a=img,
        labels_a=labels_A,
        model=dino_model,
        processor=processor,
        device=device,
        ps=dino_model.config.patch_size,
        tile_size=1024,
        stride=512,
        pos_frac_thresh=0.1,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=image_id_a,
    )

    in_ch = dino_model.config.hidden_size
    cnn_model = train_cnn_head(
        train_inputs_dino=train_inputs_dino,
        train_inputs_rgb=train_inputs_rgb,
        train_labels=train_labels,
        in_ch=in_ch,
        device=device,
        patch_size=dino_model.config.patch_size,
        hidden_ch=96,
        epochs=10,
        lr=5e-4,
        batch_size=2,
    )

    # Inference on Image B
    score_full = cnn_inference_image(
        img_b=img_b,
        cnn_model=cnn_model,
        processor=processor,
        dino_model=dino_model,
        device=device,
        ps=dino_model.config.patch_size,
        tile_size=1024,
        stride=512,
        feature_dir=feature_dir,
        image_id=image_id_b,
    )

    # Threshold search
    THRESHOLDS = np.linspace(0.01, 0.4, 36).tolist()
    best_thr_cfg = grid_search_thresholds(score_map=score_full,
                                          thresholds=THRESHOLDS,
                                          sh_mask=sh_buffer_mask_B,
                                          gt_mask=gt_mask_B,
                                          device=device)
    thr_best = best_thr_cfg["threshold"]
    mask_raw_best = np.logical_and(score_full >= thr_best, sh_buffer_mask_B)
    print(f"[best-raw-cnn] thr={thr_best:.3f}, IoU={best_thr_cfg['iou']:.3f}, F1={best_thr_cfg['f1']:.3f}")

    # CRF
    PROB_SOFTNESS_VALUES = [0.05]
    POS_W_VALUES = [3.0]
    POS_XY_STD_VALUES = [3.0]
    BILATERAL_W_VALUES = [5.0]
    BILATERAL_XY_STD_VALUES = [25.0, 50.0]
    BILATERAL_RGB_STD_VALUES = [3.0, 5.0]

    best_crf_cfg_inner, best_crf_mask = crf_grid_search(
        img_rgb=img_b,
        score_map=score_full,
        threshold_center=thr_best,
        sh_mask=sh_buffer_mask_B,
        gt_mask=gt_mask_B,
        prob_softness_vals=PROB_SOFTNESS_VALUES,
        pos_w_vals=POS_W_VALUES,
        pos_xy_std_vals=POS_XY_STD_VALUES,
        bilateral_w_vals=BILATERAL_W_VALUES,
        bilateral_xy_std_vals=BILATERAL_XY_STD_VALUES,
        bilateral_rgb_std_vals=BILATERAL_RGB_STD_VALUES,
        n_iters=5,
        max_configs=6,
        downsample_factor=2,
        num_workers=8,
        backend="process",
    )
    if best_crf_mask.shape != img_b.shape[:2]:
        best_crf_mask = resize(
            best_crf_mask.astype(np.float32),
            (img_b.shape[0], img_b.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
    best_crf_config = {"threshold": thr_best, **best_crf_cfg_inner}
    print("\n[crf-cnn] best config:")
    print(best_crf_config)

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    axs[0, 0].imshow(img_b); axs[0, 0].set_title("Image B (RGB)"); axs[0, 0].axis("off")
    axs[0, 1].imshow(labels_SH_B > 0, cmap="gray"); axs[0, 1].set_title("SH_2022 raster (B)"); axs[0, 1].axis("off")
    axs[0, 2].imshow(gt_mask_B > 0, cmap="gray"); axs[0, 2].set_title("Ground truth"); axs[0, 2].axis("off")

    overlay_raw = img_b.copy()
    overlay_raw[mask_raw_best] = (0.5 * overlay_raw[mask_raw_best] + 0.5 * np.array([0, 255, 0])).astype(overlay_raw.dtype)
    axs[1, 0].imshow(overlay_raw)
    axs[1, 0].set_title(f"CNN raw thr={thr_best:.3f}\nIoU={best_thr_cfg['iou']:.3f}, F1={best_thr_cfg['f1']:.3f}")
    axs[1, 0].axis("off")

    overlay_crf = img_b.copy()
    overlay_crf[best_crf_mask] = (0.5 * overlay_crf[best_crf_mask] + 0.5 * np.array([255, 0, 0])).astype(overlay_crf.dtype)
    axs[1, 1].imshow(overlay_crf)
    axs[1, 1].set_title(f"CRF refined\nIoU={best_crf_config['iou']:.3f}, F1={best_crf_config['f1']:.3f}")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(score_full, cmap="magma")
    axs[1, 2].set_title("Score map (CNN prob)")
    axs[1, 2].axis("off")
    plt.tight_layout()
    plt.show()

    # Export shapefiles
    base_name_b = os.path.splitext(os.path.basename(img2_path))[0]
    out_dir_b = os.path.dirname(img2_path)
    shp_raw = os.path.join(out_dir_b, f"{base_name_b}_pred_mask_cnn_raw.shp")
    shp_crf = os.path.join(out_dir_b, f"{base_name_b}_pred_mask_cnn_crf.shp")
    export_mask_to_shapefile(mask_raw_best, img2_path, shp_raw)
    export_mask_to_shapefile(best_crf_mask, img2_path, shp_crf)

    consolidate_features_for_image(feature_dir, image_id_a)
    consolidate_features_for_image(feature_dir, image_id_b)
    time_end("main_cnn (total)", t0_main)


if __name__ == "__main__":
    main()

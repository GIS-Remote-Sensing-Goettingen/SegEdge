# SPDX-License-Identifier: GPL-2.0
#
# Single-scale zero-shot transfer from Image A -> Image B using DINOv3.
# Only the pieces required for run_zero_shot_transfer(...) are kept.

import time
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from skimage.morphology import erosion, disk

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject, Resampling
import rasterio.features as rfeatures

import fiona

from transformers import AutoImageProcessor, AutoModel


# ----------------------------------------------------------------------
# 0. timing utilities
# ----------------------------------------------------------------------

DEBUG_TIMING = True


def time_start():
    """
    Start a timing block.

    Returns a timestamp or None if DEBUG_TIMING is disabled.
    """
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    """
    End a timing block and print a standardized timing line.

    label: short name for the code section.
    t0: value returned by time_start().
    """
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt:.3f} s")


# ----------------------------------------------------------------------
# 1. model setup
# ----------------------------------------------------------------------


def init_model(model_name: str):
    """
    Load DINOv3 model + image processor, move model to CUDA if available.
    Kept small and explicit for reproducibility.
    """
    t0 = time_start()

    processor = AutoImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    time_end("init_model", t0)

    return model, processor, device


# ----------------------------------------------------------------------
# 2. data loading, reprojection, and basic preparation
# ----------------------------------------------------------------------


def load_dop20_image(path: str) -> np.ndarray:
    """
    Load a DOP20 GeoTIFF into an HxWxC uint8-like array.
    Rasterio returns (bands, H, W); we convert to (H, W, C).
    """
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    img = reshape_as_image(arr)  # (H, W, C)
    if img.shape[2] > 3:
        # Many DOP tiles are RGBA; DINOv3 expects 3 channels.
        img = img[:, :, :3]
    return img


def reproject_labels_to_image(ref_img_path: str, labels_path: str) -> np.ndarray:
    """
    Reproject label raster onto the grid of the reference image.
    Returns a single-band label array (H, W).
    """
    with rasterio.open(ref_img_path) as ref, rasterio.open(labels_path) as src:
        dst_meta = ref.meta.copy()
        dst_meta.update(dtype=src.dtypes[0], count=src.count)

        memfile = rasterio.io.MemoryFile()
        with memfile.open(**dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_width=ref.width,
                    dst_height=ref.height,
                    resampling=Resampling.nearest,
                )
            labels_arr = dst.read()

    labels_2d = labels_arr[0]
    return labels_2d


# ----------------------------------------------------------------------
# 3. helpers (normalization, tiling, label-to-patch, feature I/O)
# ----------------------------------------------------------------------


def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize features along the last dimension.
    This mirrors the typical DINOv3 usage before cosine kNN.
    """
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    return feats / norms


def tile_iterator(image_hw3: np.ndarray,
                  labels_hw: np.ndarray | None = None,
                  tile_size: int = 1024,
                  stride: int | None = None):
    """
    Sliding-window tiling over a large HxWxC image (and labels).
    This pattern is standard for ViT on large remote-sensing tiles.
    """
    h, w = image_hw3.shape[:2]
    if stride is None:
        stride = tile_size

    y = 0
    while y < h:
        x = 0
        y_end = min(y + tile_size, h)
        while x < w:
            x_end = min(x + tile_size, w)
            img_tile = image_hw3[y:y_end, x:x_end]

            if labels_hw is not None:
                lab_tile = labels_hw[y:y_end, x:x_end]
            else:
                lab_tile = None

            yield y, x, img_tile, lab_tile
            x += stride
        y += stride


def crop_to_multiple_of_ps(img_tile_hw3: np.ndarray,
                           labels_tile_hw: np.ndarray | None,
                           ps: int):
    """
    Crop tile borders so that H and W are multiples of patch size.
    This makes the ViT patch grid well-defined and avoids silent cropping.
    """
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps

    img_c = img_tile_hw3[:h_eff, :w_eff]
    if labels_tile_hw is not None:
        lab_c = labels_tile_hw[:h_eff, :w_eff]
    else:
        lab_c = None

    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(labels_tile: np.ndarray,
                          hp: int,
                          wp: int,
                          pos_frac_thresh: float = 0.1):
    """
    Convert pixel-level labels to patch-level masks by pooling.
    This preserves thin linear structures better than a naive resize.
    """
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp

    labels_c = labels_tile[:hp * patch_h, :wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)

    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))  # (Hp, Wp)

    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0

    return pos_mask, neg_mask


def tile_feature_path(feature_dir: str,
                      image_id: str,
                      y: int,
                      x: int) -> str:
    """
    Build a unique file path for a tile's DINO features.
    """
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def save_tile_features(feats_tile: np.ndarray,
                       feature_dir: str,
                       image_id: str,
                       y: int,
                       x: int):
    """
    Save per-tile DINO features to disk.

    feats_tile: (Hp, Wp, C) array for this tile.
    """
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))


def consolidate_features_for_image(feature_dir: str,
                                   image_id: str,
                                   output_suffix: str = "_features_full.npy"):
    """
    Look for all tile feature files for a given image_id in feature_dir,
    concatenate them into a single (N, C) array, and save as one .npy.

    Tile files are expected to be named:
        {image_id}_y{y}_x{x}_features.npy

    Returns the output path if any tiles were found, else None.
    """
    t0 = time_start()

    if not os.path.isdir(feature_dir):
        print(f"[warn] feature_dir does not exist: {feature_dir}")
        return None

    prefix = f"{image_id}_y"
    suffix = "_features.npy"
    files = [
        f for f in os.listdir(feature_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]

    if not files:
        print(f"[warn] no feature tiles found for image_id={image_id} in {feature_dir}")
        return None

    files = sorted(files)
    feats_list = []

    for fname in files:
        fpath = os.path.join(feature_dir, fname)
        arr = np.load(fpath)  # (Hp, Wp, C)
        feats_list.append(arr.reshape(-1, arr.shape[-1]))  # (Hp*Wp, C)

    feats_full = np.concatenate(feats_list, axis=0).astype(np.float32)

    out_path = os.path.join(feature_dir, f"{image_id}{output_suffix}")
    np.save(out_path, feats_full)

    time_end(f"consolidate_features_for_image[{image_id}]", t0)
    print(
        f"[info] consolidated {len(files)} tiles for {image_id} -> {out_path}, "
        f"shape={feats_full.shape}"
    )

    return out_path


# ----------------------------------------------------------------------
# 4. single-scale DINOv3 patch feature extraction
# ----------------------------------------------------------------------


def extract_patch_features_single_scale(image_hw3: np.ndarray,
                                        model,
                                        processor,
                                        device,
                                        ps: int = 16,
                                        aggregate_layers=None):
    """
    Single-scale DINOv3 inference.
    We disable internal resize/crop in the processor and rely on external
    cropping to multiples of ps.
    """
    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)

    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape

    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)

    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens:, :]  # drop CLS + registers

    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps

    assert hp * wp == num_tokens, (
        f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    )

    feats = patch_tokens[0].cpu().numpy().reshape(hp, wp, dim)
    feats = l2_normalize(feats)

    return feats, hp, wp


# ----------------------------------------------------------------------
# 5. bank building from Image A (single-scale, with caching)
# ----------------------------------------------------------------------


def build_banks_single_scale(img_a: np.ndarray,
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
    """
    Build positive and negative banks from Image A in a single scale.
    Operates over tiles and converts labels to patch-level masks.

    If feature_dir and image_id are given, per-tile DINO features are
    cached on disk and reused on subsequent runs.
    """
    t0 = time_start()

    pos_list = []
    neg_list = []

    cached_tiles = 0
    computed_tiles = 0

    labels_eroded = erosion((labels_a > 0).astype(bool), disk(2))

    for y, x, img_tile, lab_tile in tile_iterator(img_a,
                                                  labels_eroded,
                                                  tile_size,
                                                  stride):
        img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(
            img_tile, lab_tile, ps
        )
        if h_eff < ps or w_eff < ps:
            continue

        feats_tile = None
        hp = wp = None

        # Try cache first if configured.
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
                save_tile_features(
                    feats_tile,
                    feature_dir=feature_dir,
                    image_id=image_id,
                    y=y,
                    x=x,
                )

        pos_mask, neg_mask = labels_to_patch_masks(
            lab_c,
            hp,
            wp,
            pos_frac_thresh=pos_frac_thresh,
        )

        pos_feats_tile = feats_tile[pos_mask]
        neg_feats_tile = feats_tile[neg_mask]

        if pos_feats_tile.size > 0:
            pos_list.append(pos_feats_tile)
        if neg_feats_tile.size > 0:
            neg_list.append(neg_feats_tile)

    if not pos_list:
        raise ValueError(
            "no positive patches found in Image A; "
            "check labels or pos_frac_thresh"
        )

    pos_bank = np.concatenate(pos_list, axis=0)
    if neg_list:
        neg_bank = np.concatenate(neg_list, axis=0)
    else:
        neg_bank = None

    print(f"Positive bank size: {len(pos_bank)} patches")
    if neg_bank is not None:
        print(f"Negative bank size: {len(neg_bank)} patches")

    time_end("build_banks_single_scale", t0)
    print(
        f"[cache] A: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )

    return pos_bank, neg_bank


# ----------------------------------------------------------------------
# 6. single-scale zero-shot scoring on Image B (GPU kNN + caching)
# ----------------------------------------------------------------------


def zero_shot_knn_single_scale_B_with_saliency(
    img_b: np.ndarray,
    pos_bank: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    k: int = 5,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
):
    """
    Single-scale zero-shot kNN scoring and saliency map on Image B.

    GPU version:
    - Move pos_bank to device once.
    - For each tile, move its patch features to device.
    - Compute cosine similarities via batched matmul and top-k.

    Score: mean cosine similarity of each patch to its k nearest
    positive-bank neighbours.

    Saliency: simple kNN-weighted similarity (higher => contributes
    more strongly to being classified as 'like positives').
    """
    t0 = time_start()

    h_full, w_full = img_b.shape[:2]

    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    saliency_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    cached_tiles = 0
    computed_tiles = 0

    # Move pos_bank once to GPU (or CPU device if no CUDA).
    pos_bank_t = torch.from_numpy(pos_bank.astype(np.float32)).to(device)
    k_eff = min(k, pos_bank_t.shape[0])

    for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride):
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(
            img_tile, None, ps
        )
        if h_eff < ps or w_eff < ps:
            continue

        feats_tile = None
        hp = wp = None

        # Try cache first if configured.
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
                save_tile_features(
                    feats_tile,
                    feature_dir=feature_dir,
                    image_id=image_id,
                    y=y,
                    x=x,
                )

        # Flatten tile patches: (Hp, Wp, C) -> (Nb, C)
        x_feats = feats_tile.reshape(-1, feats_tile.shape[-1]).astype(np.float32)

        # GPU kNN via cosine similarity (dot product on L2-normalized features).
        with torch.no_grad():
            x_feats_t = torch.from_numpy(x_feats).to(device)  # (Nb, C)
            sims_full = x_feats_t @ pos_bank_t.t()            # (Nb, Npos)
            sims_topk, _ = torch.topk(sims_full, k=k_eff, dim=1)
            sims = sims_topk.cpu().numpy()                    # (Nb, k_eff)

        # score: mean similarity over k neighbours
        score_patch = sims.mean(axis=1).reshape(hp, wp)

        # saliency: weighted self-contribution
        weights = sims / (sims.sum(axis=1, keepdims=True) + 1e-8)
        saliency_vals = (weights * sims).sum(axis=1)
        saliency_patch = saliency_vals.reshape(hp, wp)

        score_tile = resize(
            score_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        saliency_tile = resize(
            saliency_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        score_full[y:y + h_eff, x:x + w_eff] += score_tile
        saliency_full[y:y + h_eff, x:x + w_eff] += saliency_tile
        weight_full[y:y + h_eff, x:x + w_eff] += 1.0

    mask_nonzero = weight_full > 0.0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    saliency_full[mask_nonzero] /= weight_full[mask_nonzero]

    time_end("zero_shot_knn_single_scale_B_with_saliency (GPU)", t0)
    print(
        f"[cache] B: cached tiles={cached_tiles}, "
        f"computed tiles={computed_tiles}"
    )

    return score_full, saliency_full


# ----------------------------------------------------------------------
# 7. top-level runner
# ----------------------------------------------------------------------


def run_zero_shot_transfer(
    img_a: np.ndarray,
    labels_a: np.ndarray,
    img_b: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    k: int = 5,
    pos_frac_thresh: float = 0.1,
    threshold: float = 0.75,
    aggregate_layers=None,
    use_crf: bool = False,
    refine_with_crf=None,
    feature_dir: str | None = None,
    image_id_a: str | None = None,
    image_id_b: str | None = None,
    save_features: bool = False,
):
    """
    Public API for zero-shot transfer.
    Single-scale internally, with timing prints for main stages.

    If save_features is True and feature_dir + image_id_* are given,
    per-tile DINO patch features for A and B are cached to disk.
    """
    t0_total = time_start()

    feature_dir_a = feature_dir if save_features else None
    feature_dir_b = feature_dir if save_features else None

    pos_bank, neg_bank = build_banks_single_scale(
        img_a=img_a,
        labels_a=labels_a,
        model=model,
        processor=processor,
        device=device,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        pos_frac_thresh=pos_frac_thresh,
        aggregate_layers=aggregate_layers,
        feature_dir=feature_dir_a,
        image_id=image_id_a,
    )

    score_full, saliency_full = zero_shot_knn_single_scale_B_with_saliency(
        img_b=img_b,
        pos_bank=pos_bank,
        model=model,
        processor=processor,
        device=device,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        k=k,
        aggregate_layers=aggregate_layers,
        feature_dir=feature_dir_b,
        image_id=image_id_b,
    )

    mask_b = score_full >= threshold

    if use_crf and refine_with_crf is not None:
        coarse_labels = mask_b.astype(np.int32)
        refined = refine_with_crf(
            img_b.astype(np.uint8),
            coarse_labels,
            n_classes=2,
        )
        mask_b = (refined == 1)

    time_end("run_zero_shot_transfer (total)", t0_total)

    # basic visualization: A, labels, B+mask, saliency
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    axs[0].imshow(img_a)
    axs[0].set_title("Image A")
    axs[0].axis("off")

    axs[1].imshow(labels_a > 0, cmap="gray")
    axs[1].set_title("Image A: label mask")
    axs[1].axis("off")

    axs[2].imshow(img_b)
    axs[2].imshow(mask_b, cmap="Greens", alpha=0.4)
    axs[2].set_title(f"Image B: zero-shot mask (t={threshold:.2f})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(saliency_full, cmap="hot")
    plt.title("Patch saliency (kNN contribution)")
    plt.axis("off")
    plt.show()

    return {
        "pos_bank": pos_bank,
        "neg_bank": neg_bank,
        "score_full": score_full,
        "saliency_full": saliency_full,
        "mask_b": mask_b,
    }


# ----------------------------------------------------------------------
# 8. shapefile export
# ----------------------------------------------------------------------


def export_mask_to_shapefile(mask: np.ndarray,
                             ref_raster_path: str,
                             out_path: str):
    """
    Polygonize a boolean mask and export as ESRI Shapefile.
    Uses the CRS and transform from ref_raster_path.

    mask: HxW boolean or 0/1 array in the pixel grid of ref_raster_path.
    """
    t0 = time_start()

    mask_uint8 = mask.astype("uint8")

    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        crs = src.crs

    shape_generator = rfeatures.shapes(mask_uint8,
                                       mask=mask_uint8 == 1,
                                       transform=transform)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with fiona.open(
        out_path,
        mode="w",
        driver="ESRI Shapefile",
        crs=crs.to_dict() if crs is not None else None,
        schema=schema,
    ) as shp:
        idx = 0
        for geom, value in shape_generator:
            if value != 1:
                continue
            shp.write({
                "geometry": geom,
                "properties": {
                    "id": int(idx),
                },
            })
            idx += 1

    time_end("export_mask_to_shapefile", t0)
    print(f"[info] shapefile written to: {out_path}")


# ----------------------------------------------------------------------
# 9. script entry point
# ----------------------------------------------------------------------


def main():
    """
    Minimal main that mirrors your original configuration, using
    the cleaned single-scale runner, timing prints, shapefile export,
    and DINO feature caching + consolidation per .tif.
    """
    t0_main = time_start()

    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
    model, processor, device = init_model(model_name)

    t0_data = time_start()

    img_path = (
        "/home/mak/PycharmProjects/SegEdge/experiments/"
        "get_data_from_api/patches_mt/"
        "dop20_593000_5982000_1km_20cm.tif"
    )
    img2_path = (
        "/home/mak/PycharmProjects/SegEdge/experiments/"
        "get_data_from_api/patches_mt/"
        "dop20_592000_5982000_1km_20cm.tif"
    )
    lab_path = (
        "/run/media/mak/Partition of 1TB disk/SH_dataset/"
        "planet_labels_2022.tif"
    )

    img = load_dop20_image(img_path)
    labels_2d = reproject_labels_to_image(img_path, lab_path)
    img_b = np.array(Image.open(img2_path).convert("RGB"))

    time_end("data_loading", t0_data)

    feature_dir = os.path.join(
        os.path.dirname(img_path),
        "dino_features",
    )

    image_id_a = os.path.splitext(os.path.basename(img_path))[0]
    image_id_b = os.path.splitext(os.path.basename(img2_path))[0]

    t0_run = time_start()
    result = run_zero_shot_transfer(
        img_a=img,
        labels_a=labels_2d,
        img_b=img_b,
        model=model,
        processor=processor,
        device=device,
        ps=model.config.patch_size,
        tile_size=1024,
        stride=512,
        k=2,
        pos_frac_thresh=0.1,
        threshold=0.77,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id_a=image_id_a,
        image_id_b=image_id_b,
        save_features=True,
    )
    time_end("run_zero_shot_transfer (from main)", t0_run)

    mask_b = result["mask_b"]

    base_name_b = os.path.splitext(os.path.basename(img2_path))[0]
    out_dir_b = os.path.dirname(img2_path)
    shp_path = os.path.join(out_dir_b, f"{base_name_b}_pred_mask.shp")

    export_mask_to_shapefile(mask=mask_b,
                             ref_raster_path=img2_path,
                             out_path=shp_path)

    consolidate_features_for_image(feature_dir, image_id_a)
    consolidate_features_for_image(feature_dir, image_id_b)

    time_end("main (total)", t0_main)

    return result


if __name__ == "__main__":
    main()

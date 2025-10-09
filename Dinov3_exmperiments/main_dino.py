#!/usr/bin/env python3
"""
Minimal MVP: DINOv3-based (SAT-493M) segmentation and detection of individual trees
from satellite GeoTIFFs. Saves:
  1) binary mask (uint8, 0/1) for tree vs. non-tree
  2) tree crown polygons (GeoPackage)
  3) tree centroids as points (GeoPackage)
Supports two modes:
  - unsup: k-means on DINO dense features + morphology tuned for blob-like tree crowns
  - linear: 1x1 conv "linear probe" if you provide a tiny labeled mask for a few tiles

Requirements (pip):
  torch torchvision timm rasterio numpy scikit-image scikit-learn shapely geopandas tqdm
  opencv-python (for contour extraction)
Optional (for DINOv3): transformers

References:
  - DINOv3 SAT-493M for satellite imagery
  - Watershed segmentation for individual tree crown delineation
"""

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk, dilation, erosion
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch, math
from transformers import AutoModel, AutoImageProcessor
import cv2

# Vectorization
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

CUDA_AVAILABLE = torch.cuda.is_available()


class StageProfiler:
    def __init__(self):
        self.records = OrderedDict()

    @contextmanager
    def track(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            total, count = self.records.get(name, (0.0, 0))
            self.records[name] = (total + duration, count + 1)
            print(f"[TIMER] {name}: {duration:.3f}s (call {count + 1})")

    def summary(self):
        print("\n[TIMER] Summary:")
        for name, (total, count) in self.records.items():
            avg = total / max(count, 1)
            print(f"  - {name}: total={total:.3f}s, calls={count}, avg={avg:.3f}s")


PROFILER = StageProfiler()


def log_cuda(message):
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[CUDA] {message} | allocated={allocated:.1f}MB reserved={reserved:.1f}MB")

# -----------------------------
# Utilities
# -----------------------------
def load_geotiff(path, bands=None, max_edge=4096):
    """Read a GeoTIFF (optionally select bands), optionally downscale for memory."""
    with rio.open(path) as ds:
        src_bands = bands if bands is not None else list(range(1, ds.count + 1))
        # Optional downscale to keep RAM in check
        scale = 1.0
        if max(ds.width, ds.height) > max_edge:
            scale = max_edge / max(ds.width, ds.height)
        out_w, out_h = int(ds.width * scale), int(ds.height * scale)
        arr = ds.read(src_bands, out_shape=(len(src_bands), out_h, out_w),
                      resampling=Resampling.bilinear).astype(np.float32)
        profile = ds.profile
        profile.update(width=out_w, height=out_h, count=len(src_bands),
                       transform=ds.transform * ds.transform.scale(ds.width / out_w, ds.height / out_h))
    # Normalize per-band to [0,1] robustly
    arr = np.clip((arr - np.nanpercentile(arr, 1, axis=(1, 2), keepdims=True)) /
                  (np.nanpercentile(arr, 99, axis=(1, 2), keepdims=True) -
                   np.nanpercentile(arr, 1, axis=(1, 2), keepdims=True) + 1e-6), 0, 1)
    return arr, profile


def to_patches(img, size=896, stride=640):
    """CHW -> list of (patch, y, x). size & stride tuned for ViT dense features throughput."""
    C, H, W = img.shape
    patches = []
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            yy = min(y, H - size);
            xx = min(x, W - size)
            patches.append((img[:, yy:yy + size, xx:xx + size], yy, xx))
    if not patches:
        patches.append((img, 0, 0))
    return patches, (H, W)


def stitch_map(chunks, full_shape, size, stride):
    """Average-blend overlapping patches back into full tensor map."""
    C, H, W = full_shape
    out = np.zeros((C, H, W), dtype=np.float32)
    wts = np.zeros((1, H, W), dtype=np.float32)
    for (feat, y, x) in chunks:
        h, w = feat.shape[-2], feat.shape[-1]
        out[:, y:y + h, x:x + w] += feat
        wts[:, y:y + h, x:x + w] += 1
    out /= np.maximum(wts, 1e-6)
    return out


def postprocess_trees(mask_bin, min_area=25, max_area=10000):
    """Clean mask to emphasize blob-like tree crowns."""
    # Close small gaps within tree crowns
    mask_bin = binary_closing(mask_bin > 0, footprint=disk(2))
    # Remove small noise
    mask_bin = binary_opening(mask_bin, footprint=disk(1))
    # Remove objects outside realistic tree crown size range
    mask_bin = remove_small_objects(mask_bin, min_size=min_area)

    # Remove very large blobs (likely not individual trees)
    labeled = label(mask_bin, connectivity=2)
    for region in regionprops(labeled):
        if region.area > max_area:
            mask_bin[labeled == region.label] = 0

    return mask_bin.astype(np.uint8)


def segment_individual_trees(mask_bin, min_distance=10):
    """
    Use watershed segmentation to separate touching tree crowns.
    Returns labeled image where each tree has a unique ID.
    """
    # Distance transform
    distance = ndi.distance_transform_edt(mask_bin)

    # Find peaks (tree centers) with minimum separation
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask_bin)

    # Create markers from peaks
    mask_markers = np.zeros(distance.shape, dtype=bool)
    mask_markers[tuple(coords.T)] = True
    markers = label(mask_markers)

    # Watershed segmentation
    labels_ws = watershed(-distance, markers, mask=mask_bin)

    return labels_ws


def save_visualizations(image_chw, mask_pp, labels_ws, out_dir, stem, bbox_pix=None):
    """Save an RGB overlay showing mask coverage, watershed boundaries, and optional bounding boxes."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except Exception as exc:
        warnings.warn(f"Skipping overlay visualization: {exc}")
        return None

    if image_chw.ndim != 3:
        warnings.warn("Expected CHW image tensor for visualization.")
        return None

    # Use the first three bands for visualization; repeat single-band inputs.
    if image_chw.shape[0] >= 3:
        base = image_chw[:3]
    else:
        base = np.repeat(image_chw[:1], 3, axis=0)
    base = np.clip(base, 0.0, 1.0)
    img_hwc = base.transpose(1, 2, 0)

    overlay = np.ma.masked_where(mask_pp == 0, mask_pp.astype(float))
    boundaries = find_boundaries(labels_ws, mode="inner")

    overlay_path = out_dir / f"{stem}_overlay.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_hwc)
    if overlay.count() > 0:
        ax.imshow(overlay, cmap="autumn", alpha=0.4, interpolation="nearest")
    if boundaries.any():
        ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="cool", alpha=0.6, interpolation="nearest")
    if bbox_pix:
        for entry in bbox_pix:
            min_row, min_col, max_row, max_col = entry["bbox"]
            width = max_col - min_col
            height = max_row - min_row
            rect = patches.Rectangle(
                (min_col, min_row),
                width,
                height,
                linewidth=1.5,
                edgecolor="#3cff5c",
                facecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(
                min_col,
                max(min_row - 2, 0),
                str(entry["tree_id"]),
                color="black",
                fontsize=6,
                backgroundcolor="#3cff5c",
                alpha=0.8,
            )
    ax.set_title("Tree mask + watershed boundaries")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)
    return overlay_path


def extract_tree_polygons(labels_ws, profile, min_area=25):
    """
    Extract individual tree crown polygons from labeled watershed image.
    Returns a tuple of:
      - GeoDataFrame of crown geometries
      - GeoDataFrame of axis-aligned bounding boxes
      - list of pixel-domain bounding boxes (min_row, min_col, max_row, max_col)
    """
    transform = profile.get('transform')
    crs = profile.get('crs')

    trees = []
    bboxes_geo = []
    bboxes_pix = []
    for region in regionprops(labels_ws):
        if region.area < min_area:
            continue

        # Create binary mask for this tree
        tree_mask = (labels_ws == region.label).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Take largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify and convert to polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Convert pixel coordinates to georeferenced coordinates
        coords = []
        for point in approx:
            px, py = point[0]
            x, y = transform * (px, py)
            coords.append((x, y))

        try:
            poly = Polygon(coords)

            # Calculate centroid in georeferenced coordinates
            centroid_px = region.centroid  # (row, col)
            centroid_x, centroid_y = transform * (centroid_px[1], centroid_px[0])

            trees.append({
                'geometry': poly,
                'tree_id': region.label,
                'area_px': region.area,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity
            })
            min_row, min_col, max_row, max_col = region.bbox
            bbox_coords = [
                transform * (min_col, min_row),
                transform * (min_col, max_row),
                transform * (max_col, max_row),
                transform * (max_col, min_row),
            ]
            bbox_poly = Polygon(bbox_coords)
            bboxes_geo.append({
                'geometry': bbox_poly,
                'tree_id': region.label,
                'area_px': region.area
            })
            bboxes_pix.append({
                'tree_id': region.label,
                'bbox': (min_row, min_col, max_row, max_col)
            })
        except Exception as e:
            warnings.warn(f"Failed to create polygon for tree {region.label}: {e}")
            continue

    if not trees:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
        return empty_gdf, empty_gdf.copy(), []

    gdf = gpd.GeoDataFrame(trees, crs=crs)
    gdf_bbox = gpd.GeoDataFrame(bboxes_geo, crs=crs) if bboxes_geo else gpd.GeoDataFrame(geometry=[], crs=crs)
    return gdf, gdf_bbox, bboxes_pix


def extract_tree_points(gdf_polygons):
    """Extract centroid points from tree polygons."""
    if len(gdf_polygons) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_polygons.crs)

    points = []
    for idx, row in gdf_polygons.iterrows():
        points.append({
            'geometry': Point(row['centroid_x'], row['centroid_y']),
            'tree_id': row['tree_id'],
            'area_px': row['area_px']
        })

    gdf_points = gpd.GeoDataFrame(points, crs=gdf_polygons.crs)
    return gdf_points


# -----------------------------
# DINO backbones
# -----------------------------

class Dinov3SatDenseExtractor(torch.nn.Module):
    """
    Dense token -> H/16 x W/16 grid for DINOv3 SAT-493M ViTs (CLS+4 registers removed).
    Upsamples to input H,W for downstream heads.
    """

    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-sat493m", device="cpu"):
        super().__init__()
        self.proc = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device(device)
        use_fp16 = self.device.type == "cuda" and torch.cuda.is_available()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.dtype = next(self.model.parameters()).dtype
        self.patch = 16
        self.n_regs = 4  # SAT-493M ViTs use 1 CLS + 4 register tokens

    @torch.no_grad()
    def forward(self, img_chw):
        """
        img_chw: np.float32 in [0,1], CxHxW (RGB only for these weights)
        returns: torch.FloatTensor, C_feat x H x W
        """
        if isinstance(img_chw, torch.Tensor):
            tensor = img_chw.detach()
            if tensor.dim() == 4:
                if tensor.shape[0] != 1:
                    raise ValueError("Expected batch of size 1 or CHW tensor.")
                tensor = tensor[0]
            elif tensor.dim() != 3:
                raise ValueError("Expected CHW tensor.")
            tensor = tensor.clamp(0.0, 1.0)
            H, W = tensor.shape[-2:]
            x = (
                tensor[:3, ...]
                .permute(1, 2, 0)
                .mul(255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        elif isinstance(img_chw, np.ndarray):
            arr = img_chw
            if arr.ndim == 4:
                if arr.shape[0] != 1:
                    raise ValueError("Expected batch of size 1 or CHW array.")
                arr = arr[0]
            elif arr.ndim != 3:
                raise ValueError("Expected CHW array.")
            arr = np.clip(arr, 0.0, 1.0)
            H, W = arr.shape[-2:]
            x = (arr[:3, ...].transpose(1, 2, 0) * 255.0).astype("uint8")
        else:
            raise TypeError("img_chw must be a torch.Tensor or np.ndarray.")

        inputs = self.proc(
            images=x,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        if pixel_values.device.type != self.device.type:
            raise RuntimeError(f"Extractor expected tensors on {self.device} but received {pixel_values.device}")
        out = self.model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # [B, 1 + n_regs + Npatch, C]
        B, Ntok, C = tokens.shape
        # remove class + register tokens
        patch_tokens = tokens[:, 1 + self.n_regs:, :]  # [B, Npatch, C]
        _, _, h_model, w_model = pixel_values.shape
        h_tokens = max(1, h_model // self.patch)
        w_tokens = max(1, w_model // self.patch)
        if h_tokens * w_tokens != patch_tokens.shape[1]:
            aspect = h_model / max(w_model, 1)
            h_tokens = max(1, int(round(math.sqrt(patch_tokens.shape[1] * aspect))))
            w_tokens = max(1, patch_tokens.shape[1] // h_tokens)
        if h_tokens * w_tokens != patch_tokens.shape[1]:
            raise RuntimeError(
                f"Token count mismatch—expected {h_tokens * w_tokens}, got {patch_tokens.shape[1]}"
            )
        grid = patch_tokens.permute(0, 2, 1).reshape(B, C, h_tokens, w_tokens)
        dense = torch.nn.functional.interpolate(
            grid,
            size=(h_model, w_model),
            mode="bilinear",
            align_corners=False,
        )
        return dense.squeeze(0)  # C_feat x H x W


def load_sat493m_extractor(device="cpu", large=True):
    name = "facebook/dinov3-vitl16-pretrain-sat493m" if large else "facebook/dinov3-vit7b16-pretrain-sat493m"
    return Dinov3SatDenseExtractor(model_name=name, device=device)


# -----------------------------
# Heads
# -----------------------------
class LinearHead(nn.Module):
    """1x1 conv linear probe over dense features -> 2 classes (tree vs non-tree)."""

    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1, bias=True)

    def forward(self, f):
        return self.conv(f)


# -----------------------------
# Pipeline
# -----------------------------
def extract_dense_features(img_chw, extractor, device, size=896, stride=640, mean=None, std=None):
    with PROFILER.track("extract_dense_features"):
        patches, (H, W) = to_patches(img_chw, size=size, stride=stride)
        out_chunks = []
        for patch, yy, xx in tqdm(patches, desc="Dense features"):
            patch_start = time.perf_counter()
            patch_work = patch
            # Optional normalization for alternative backbones; skipped when mean/std are None.
            if mean is not None and std is not None:
                mean_arr = np.asarray(mean, dtype=patch.dtype).reshape(-1, 1, 1)
                std_arr = np.asarray(std, dtype=patch.dtype).reshape(-1, 1, 1)
                patch_work = (patch_work - mean_arr) / std_arr
            # If multispectral (>3 bands), select 3 for the backbone, default: pseudo-RGB with bands[0:3]
            if patch_work.shape[0] > 3:
                patch_work = patch_work[:3, ...]
            f = extractor(patch_work)  # -> torch.Tensor C,H,W
            log_cuda("Post feature forward")
            out_chunks.append((f.detach().cpu().numpy(), yy, xx))
            patch_dur = time.perf_counter() - patch_start
            print(f"[TIMER] patch ({yy},{xx}) -> {patch_dur:.3f}s")
        feat = stitch_map(out_chunks, (out_chunks[0][0].shape[0], H, W), size, stride)
        return feat  # C,H,W


def _run_kmeans_gpu(X, k=2, iters=20, tol=1e-4):
    N = X.shape[0]
    if N == 0:
        raise ValueError("Empty feature tensor passed to k-means.")
    indices = torch.randperm(N, device=X.device)[:k]
    centers = X[indices].clone()
    for it in range(iters):
        dists = torch.cdist(X, centers, p=2)
        labels = dists.argmin(dim=1)
        log_cuda(f"KMeans iteration {it} distances computed")
        new_centers = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                new_centers.append(X[mask].mean(dim=0))
            else:
                fallback_idx = torch.randint(0, N, (1,), device=X.device)
                new_centers.append(X[fallback_idx].squeeze(0))
        new_centers = torch.stack(new_centers, dim=0)
        shift = (centers - new_centers).norm(dim=1).max()
        centers = new_centers
        log_cuda(f"KMeans iteration {it} centers updated (shift={shift.item():.6f})")
        if shift < tol:
            print(f"[KMEANS] Converged in {it + 1} iterations (shift={shift.item():.6f}).")
            break
    return labels, centers


def segment_unsupervised(feat_chw, image_chw=None, k=2,
                         coverage_bounds=(0.01, 0.7), device="cpu"):
    """
    Run k-means over dense features and pick the cluster likely representing trees.

    The previous implementation always chose the cluster with higher feature variance.
    That fails whenever background (roads, buildings) is more textured than trees,
    producing an all-zero mask after post-processing. We now score clusters using
    a combination of feature texture, activation magnitude, optional greenness cues,
    and a soft prior on expected coverage. This greatly reduces the chance of
    selecting the wrong cluster and returning an empty tree mask.
    """
    C, H, W = feat_chw.shape
    X = feat_chw.reshape(C, -1).T
    if device == "cuda" and CUDA_AVAILABLE:
        with PROFILER.track("kmeans_gpu"):
            feat_tensor = torch.from_numpy(feat_chw).to(device=device, dtype=torch.float32)
            flat_tensor = feat_tensor.reshape(C, -1).transpose(0, 1).contiguous()
            labels_tensor, centers = _run_kmeans_gpu(flat_tensor, k=k)
            y = labels_tensor.view(H, W).cpu().numpy()
            log_cuda("KMeans complete")
    else:
        with PROFILER.track("kmeans_cpu"):
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            y = km.fit_predict(X).reshape(H, W)

    cluster_info = []
    with PROFILER.track("cluster_scoring"):
        for label_idx in range(k):
            cluster_mask = (y == label_idx)
            pix_count = int(cluster_mask.sum())
            if pix_count == 0:
                cluster_info.append({
                    "label": label_idx,
                    "score": float("-inf"),
                    "coverage": 0.0
                })
                continue

            coverage = pix_count / float(H * W)
            feat_cluster = feat_chw[:, cluster_mask]
            # Texture and activation magnitude
            texture = float(np.nanstd(feat_cluster))
            activation = float(np.nanmean(np.linalg.norm(feat_cluster, axis=0)))
            score = texture + 0.1 * activation
            score = float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))

            # Prefer clusters whose coverage is within a plausible tree fraction.
            low_bound, high_bound = coverage_bounds
            if coverage < low_bound:
                score *= 0.5 if coverage > 0 else 0.1
            elif coverage > high_bound:
                score *= 0.1 if coverage > min(0.95, high_bound * 1.5) else 0.5

            # Greenness cue: when at least RGB is available, trees tend to be greener.
            if image_chw is not None and image_chw.shape[0] >= 3:
                red = image_chw[0]
                green = image_chw[1]
                try:
                    vegetation = float(green[cluster_mask].mean() - red[cluster_mask].mean())
                    score += float(vegetation)
                except Exception:
                    pass

            print(f"[SEG] Cluster {label_idx}: pix={pix_count}, coverage={coverage:.3f}, score={score:.3f}")
            cluster_info.append({
                "label": label_idx,
                "score": score,
                "coverage": coverage
            })

    # Pick the cluster with the highest score.
    best = max(cluster_info, key=lambda d: d["score"])
    mask = (y == best["label"]).astype(np.uint8)
    coverage = best["coverage"]

    # Fallback: if coverage is extreme, try the best alternative cluster.
    if coverage <= 0.0 or coverage >= 0.95:
        alternatives = sorted(
            (info for info in cluster_info if info["label"] != best["label"]),
            key=lambda d: (abs(d["coverage"] - sum(coverage_bounds) / 2.0), -d["score"])
        )
        for alt in alternatives:
            if 0.0 < alt["coverage"] < 0.95:
                mask = (y == alt["label"]).astype(np.uint8)
                coverage = alt["coverage"]
                break

    return mask


def segment_linear_probe(feat_chw, head_ckpt):
    head = torch.load(head_ckpt, map_location='cpu')
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(feat_chw[None, ...])).squeeze(0).numpy()
    mask = (logits.argmax(0) == 1).astype(np.uint8)
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="GeoTIFF path")
    p.add_argument("--bands", type=str, default="",
                   help="Comma list (1-based), e.g. '3,2,1' for RGB. Empty=use first 3.")
    p.add_argument("--mode", choices=["unsup", "linear"], default="unsup", help="Unsupervised or linear-probe")
    p.add_argument("--head", type=Path, default=None, help="Path to saved LinearHead weights (for mode=linear)")
    p.add_argument("--max-edge", type=int, default=4096, help="Downscale max image edge to manage memory")
    p.add_argument("--patch", type=int, default=896)
    p.add_argument("--stride", type=int, default=640)
    p.add_argument("--min-area", type=int, default=25, help="Minimum tree crown area in pixels")
    p.add_argument("--max-area", type=int, default=10000, help="Maximum tree crown area in pixels")
    p.add_argument("--min-distance", type=int, default=10, help="Minimum distance between tree centers for watershed")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Select compute device")
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    if device == "cuda":
        cuda_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(cuda_index)
        print(f"Using device: cuda ({gpu_name})")
    else:
        print("Using device: cpu")

    # Read image
    bands = [int(b) for b in args.bands.split(",")] if args.bands else None
    with PROFILER.track("load_geotiff"):
        arr, profile = load_geotiff(args.input, bands=bands, max_edge=args.max_edge)

    extractor = load_sat493m_extractor(device=device, large=True)
    print(f"DINO extractor dtype: {extractor.dtype}, device: {extractor.device}")
    log_cuda("Extractor loaded")

    # Dense features
    feat = extract_dense_features(arr, extractor, device=device, size=args.patch, stride=args.stride, mean=None,
                                  std=None)

    # Segmentation
    if args.mode == "unsup":
        mask = segment_unsupervised(feat, image_chw=arr, k=2, device=device)
    else:
        if args.head is None or not args.head.exists():
            print("ERROR: --head is required for mode=linear and must exist")
            sys.exit(1)
        mask = segment_linear_probe(feat, args.head)

    # Postprocess for tree crown emphasis
    with PROFILER.track("postprocess_trees"):
        mask_pp = postprocess_trees(mask, min_area=args.min_area, max_area=args.max_area)
    if args.mode == "unsup":
        mask_tree_count = int(label(mask_pp, connectivity=2).max())
        alt_mask = (1 - mask).astype(np.uint8)
        with PROFILER.track("postprocess_trees_alt"):
            alt_mask_pp = postprocess_trees(alt_mask, min_area=args.min_area, max_area=args.max_area)
        alt_tree_count = int(label(alt_mask_pp, connectivity=2).max()) if alt_mask_pp.any() else 0
        if alt_mask_pp.sum() > mask_pp.sum() or alt_tree_count > mask_tree_count:
            mask = alt_mask
            mask_pp = alt_mask_pp
            mask_tree_count = alt_tree_count

    # Segment individual trees
    with PROFILER.track("segment_individual_trees"):
        labels_ws = segment_individual_trees(mask_pp, min_distance=args.min_distance)

    # Extract tree polygons
    effective_min_area = args.min_area
    effective_min_distance = args.min_distance
    with PROFILER.track("extract_tree_polygons"):
        gdf_trees, gdf_bboxes, bbox_pix = extract_tree_polygons(labels_ws, profile, min_area=effective_min_area)

    if args.mode == "unsup" and len(gdf_trees) < 5:
        relaxed_min_area = max(5, args.min_area // 2 if args.min_area > 5 else args.min_area)
        relaxed_min_distance = max(3, args.min_distance // 2 if args.min_distance > 3 else args.min_distance)
        if relaxed_min_area != effective_min_area or relaxed_min_distance != effective_min_distance:
            mask_pp_relaxed = postprocess_trees(mask, min_area=relaxed_min_area, max_area=args.max_area)
            labels_relaxed = segment_individual_trees(mask_pp_relaxed, min_distance=relaxed_min_distance)
            gdf_relaxed, bbox_relaxed, bbox_pix_relaxed = extract_tree_polygons(labels_relaxed, profile, min_area=relaxed_min_area)
            if len(gdf_relaxed) > len(gdf_trees):
                print(f"Relaxed thresholds to {len(gdf_relaxed)} trees "
                      f"(min_area={relaxed_min_area}, min_distance={relaxed_min_distance}).")
                mask_pp = mask_pp_relaxed
                labels_ws = labels_relaxed
                gdf_trees = gdf_relaxed
                gdf_bboxes = bbox_relaxed
                bbox_pix = bbox_pix_relaxed
                effective_min_area = relaxed_min_area
                effective_min_distance = relaxed_min_distance

    # Save binary tree mask
    mask_path = args.out_dir / (args.input.stem + "_tree_mask.tif")
    with rio.open(
            mask_path, "w",
            driver="GTiff",
            height=mask_pp.shape[0], width=mask_pp.shape[1],
            count=1, dtype="uint8", crs=profile.get('crs'), transform=profile.get('transform'),
            compress="LZW"
    ) as dst:
        dst.write(mask_pp[None, ...].astype(np.uint8))

    # Save labeled tree image
    labels_path = args.out_dir / (args.input.stem + "_tree_labels.tif")
    with rio.open(
            labels_path, "w",
            driver="GTiff",
            height=labels_ws.shape[0], width=labels_ws.shape[1],
            count=1, dtype="int32", crs=profile.get('crs'), transform=profile.get('transform'),
            compress="LZW"
    ) as dst:
        dst.write(labels_ws[None, ...].astype(np.int32))

    gpkg_path = args.out_dir / (args.input.stem + "_tree_crowns.gpkg")
    if len(gdf_trees) > 0:
        gdf_trees.to_file(gpkg_path, driver="GPKG")
        print(f"Detected {len(gdf_trees)} trees")
        print(f"Saved tree crowns: {gpkg_path}")

        if len(gdf_bboxes) > 0:
            bbox_path = args.out_dir / (args.input.stem + "_tree_bboxes.gpkg")
            gdf_bboxes.to_file(bbox_path, driver="GPKG")
            print(f"Saved tree bounding boxes: {bbox_path}")

        # Extract and save tree centroids
        gdf_points = extract_tree_points(gdf_trees)
        points_path = args.out_dir / (args.input.stem + "_tree_points.gpkg")
        gdf_points.to_file(points_path, driver="GPKG")
        print(f"Saved tree centroids: {points_path}")
        if len(gdf_trees) < 5:
            print("Note: fewer than 5 trees met the filtering thresholds—"
                  "consider tweaking --min-area/--min-distance for your imagery.")
    else:
        print("Warning: no trees detected—check thresholds or imagery resolution.")

    overlay_path = save_visualizations(arr, mask_pp, labels_ws, args.out_dir, args.input.stem, bbox_pix=bbox_pix)
    if overlay_path:
        print(f"Saved overlay preview: {overlay_path}")

    print(f"Effective thresholds -> min_area: {effective_min_area}, min_distance: {effective_min_distance}")
    print(f"Saved binary mask: {mask_path}")
    print(f"Saved labeled image: {labels_path}")

    PROFILER.summary()


if __name__ == "__main__":
    main()

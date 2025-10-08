#!/usr/bin/env python3
"""
Minimal MVP: DINOv3-based (fallback: DINOv2) segmentation of Linear Woody Features (LWFs)
from satellite GeoTIFFs. Saves:
  1) binary mask (uint8, 0/1) for woody vs. non-woody
  2) vectorized LWF polylines (GeoPackage)
Supports two modes:
  - unsup: k-means on DINO dense features + morphology tuned for thin/linear objects
  - linear: 1x1 conv "linear probe" if you provide a tiny labeled mask for a few tiles

Requirements (pip):
  torch torchvision timm rasterio numpy scikit-image scikit-learn shapely geopandas tqdm
Optional (for DINOv3): git+https://github.com/facebookresearch/dinov3
Fallback (DINOv2): torch.hub from facebookresearch/dinov2 (included below)

References (concepts/choices):
  - DINOv3 paper+code (dense features; linear probe seg) – 2025.08.13 & GitHub release
  - DINOv2 linear-seg baseline (1x1 conv on dense feats, upsample) – 2023
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from skimage.morphology import skeletonize, remove_small_objects, binary_opening, disk
from skimage.measure import label
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch, math
from transformers import AutoModel, AutoImageProcessor
# Vectorization
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

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
                       transform=ds.transform * ds.transform.scale(ds.width/out_w, ds.height/out_h))
    # Normalize per-band to [0,1] robustly
    arr = np.clip((arr - np.nanpercentile(arr, 1, axis=(1,2), keepdims=True)) /
                  (np.nanpercentile(arr, 99, axis=(1,2), keepdims=True) -
                   np.nanpercentile(arr, 1, axis=(1,2), keepdims=True) + 1e-6), 0, 1)
    return arr, profile

def to_patches(img, size=896, stride=640):
    """CHW -> list of (patch, y, x). size & stride tuned for ViT dense features throughput."""
    C, H, W = img.shape
    patches = []
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            yy = min(y, H - size); xx = min(x, W - size)
            patches.append((img[:, yy:yy+size, xx:xx+size], yy, xx))
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
        out[:, y:y+h, x:x+w] += feat
        wts[:, y:y+h, x:x+w] += 1
    out /= np.maximum(wts, 1e-6)
    return out

def postprocess_linear(mask_bin, min_obj=64, skel=True):
    """Thin+clean to emphasize linear woody features."""
    if skel:
        mask_bin = skeletonize(mask_bin > 0)
    mask_bin = remove_small_objects(mask_bin, min_size=min_obj)
    mask_bin = binary_opening(mask_bin, footprint=disk(1))
    return mask_bin.astype(np.uint8)

def mask_to_lines(mask_bin, profile, simplify_tolerance=0.0):
    """Vectorize skeleton mask into polylines in image CRS."""
    # Extract connected components and trace pixel centerlines as rough polylines.
    lab = label(mask_bin, connectivity=2)
    lines = []
    H, W = mask_bin.shape
    # Simple raster-to-lines by tracing rows (rudimentary but OK for MVP)
    for lbl in range(1, lab.max()+1):
        ys, xs = np.where(lab == lbl)
        if len(xs) < 5:
            continue
        coords = [(float(x), float(y)) for x, y in zip(xs, ys)]
        try:
            ln = LineString(coords)
            lines.append(ln)
        except Exception:
            continue
    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=profile.get('crs'))

    merged = linemerge(lines)
    if isinstance(merged, LineString):
        geoms = [merged]
    elif isinstance(merged, MultiLineString):
        geoms = list(merged.geoms)
    else:
        geoms = lines

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=profile.get('crs'))
    if simplify_tolerance > 0:
        gdf['geometry'] = gdf.simplify(simplify_tolerance, preserve_topology=True)
    return gdf

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
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).eval().to(device)
        self.patch = 16
        self.n_regs = 4  # SAT-493M ViTs use 1 CLS + 4 register tokens
        # Use the satellite-specific normalization packaged in the processor
        # (mean/std differ from the web-pretrained models)
        # see HF model card / PR noting mean&std update

    @torch.no_grad()
    def forward(self, img_chw):
        """
        img_chw: np.float32 in [0,1], CxHxW (RGB only for these weights)
        returns: torch.FloatTensor, C_feat x H x W
        """
        import numpy as np
        H, W = img_chw.shape[-2:]
        # HF processor expects PIL/np HWC
        x = (img_chw[:3,...].transpose(1,2,0) * 255.0).astype("uint8")
        inputs = self.proc(images=x, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs)
        tokens = out.last_hidden_state  # [B, 1 + n_regs + Npatch, C]
        B, Ntok, C = tokens.shape
        # remove class + register tokens
        patch_tokens = tokens[:, 1 + self.n_regs :, :]  # [B, Npatch, C]
        h_tokens, w_tokens = H // self.patch, W // self.patch
        assert patch_tokens.shape[1] == h_tokens * w_tokens, "Token count mismatch—check resize"
        grid = patch_tokens.permute(0,2,1).reshape(B, C, h_tokens, w_tokens)
        dense = torch.nn.functional.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
        return dense.squeeze(0)  # C_feat x H x W

def load_sat493m_extractor(device="cpu", large=True):
    name = "facebook/dinov3-vitl16-pretrain-sat493m" if large else "facebook/dinov3-vit7b16-pretrain-sat493m"
    return Dinov3SatDenseExtractor(model_name=name, device=device)

# -----------------------------
# Heads
# -----------------------------
class LinearHead(nn.Module):
    """1x1 conv linear probe over dense features -> 2 classes (woody vs non-woody)."""
    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1, bias=True)
    def forward(self, f):
        return self.conv(f)

# -----------------------------
# Pipeline
# -----------------------------
def extract_dense_features(img_chw, extractor, device, size=896, stride=640, mean=None, std=None):
    patches, (H, W) = to_patches(img_chw, size=size, stride=stride)
    out_chunks = []
    for patch, yy, xx in tqdm(patches, desc="Dense features"):
        ten = torch.from_numpy(patch).to(device)
        # If multispectral (>3 bands), select 3 for the backbone, default: pseudo-RGB with bands[0:3]
        if ten.shape[0] > 3:
            ten = ten[:3, ...]
        # normalize to ImageNet stats
        ten = (ten - (mean if mean is not None else 0.5)) / (std if std is not None else 0.5)
        ten = ten.unsqueeze(0)  # B=1
        with torch.no_grad():
            f = extractor(ten)  # 1,C,H,W
        out_chunks.append((f.squeeze(0).cpu().numpy(), yy, xx))
    feat = stitch_map(out_chunks, (out_chunks[0][0].shape[0], H, W), size, stride)
    return feat  # C,H,W

def segment_unsupervised(feat_chw, k=2, woody_label=1):
    C, H, W = feat_chw.shape
    X = feat_chw.reshape(C, -1).T
    km = KMeans(n_clusters=k, n_init='auto', random_state=0)
    y = km.fit_predict(X).reshape(H, W)
    # Heuristic: take cluster with higher average NDVI proxy (use channelwise mean as proxy); flip if needed
    # With generic DINO features we can choose the cluster with higher local variance as "woody"
    var0 = feat_chw[:, y==0].var()
    var1 = feat_chw[:, y==1].var()
    woody = 1 if var1 > var0 else 0
    mask = (y == woody).astype(np.uint8)
    return mask

def segment_linear_probe(feat_chw, head_ckpt):
    head = torch.load(head_ckpt, map_location='cpu')
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(feat_chw[None,...])).squeeze(0).numpy()
    mask = (logits.argmax(0) == 1).astype(np.uint8)
    return mask

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/home/mak/Downloads/super_small.jpg"  , type=Path, required=True, help="GeoTIFF path")
    p.add_argument("--bands", type=str, default="", help="Comma list (1-based), e.g. '3,2,1' for RGB. Empty=use first 3.")
    p.add_argument("--mode", choices=["unsup", "linear"], default="unsup", help="Unsupervised or linear-probe")
    p.add_argument("--head", type=Path, default=None, help="Path to saved LinearHead weights (for mode=linear)")
    p.add_argument("--max-edge", type=int, default=4096, help="Downscale max image edge to manage memory")
    p.add_argument("--patch", type=int, default=896)
    p.add_argument("--stride", type=int, default=640)
    p.add_argument("--min-obj", type=int, default=64)
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument("--simplify", type=float, default=0.0, help="Douglas-Peucker tolerance for line simplify (pixels)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read image
    bands = [int(b) for b in args.bands.split(",")] if args.bands else None
    arr, profile = load_geotiff(args.input, bands=bands, max_edge=args.max_edge)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = load_sat493m_extractor(device=device, large=True)


    # Dense features
    feat = extract_dense_features(arr, extractor, device=device, size=args.patch, stride=args.stride, mean=None,
                                  std=None)
    # Segmentation
    if args.mode == "unsup":
        mask = segment_unsupervised(feat, k=2)
    else:
        if args.head is None or not args.head.exists():
            print("ERROR: --head is required for mode=linear and must exist")
            sys.exit(1)
        mask = segment_linear_probe(feat, args.head)

    # Postprocess for LWF emphasis
    mask_pp = postprocess_linear(mask, min_obj=args.min_obj, skel=True)

    # Save raster mask
    mask_path = args.out_dir / (args.input.stem + "_lwf_mask.tif")
    with rio.open(
        mask_path, "w",
        driver="GTiff",
        height=mask_pp.shape[0], width=mask_pp.shape[1],
        count=1, dtype="uint8", crs=profile.get('crs'), transform=profile.get('transform'),
        compress="LZW"
    ) as dst:
        dst.write(mask_pp[None, ...].astype(np.uint8))

    # Vectorize to lines
    gdf = mask_to_lines(mask_pp, profile, simplify_tolerance=args.simplify)
    gpkg_path = args.out_dir / (args.input.stem + "_lwf.gpkg")
    if len(gdf) > 0:
        gdf.to_file(gpkg_path, driver="GPKG")
    print(f"Saved: {mask_path}")
    if len(gdf) > 0:
        print(f"Saved: {gpkg_path}")
    else:
        print("Warning: no lines extracted—check thresholds or imagery resolution.")

if __name__ == "__main__":
    main()

import os
import logging
import rasterio
import fiona
import numpy as np
import rasterio.features as rfeatures
from rasterio.plot import reshape_as_image
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from skimage.morphology import dilation, disk

from timing_utils import time_start, time_end, DEBUG_TIMING, DEBUG_TIMING_VERBOSE
import config as cfg

logger = logging.getLogger(__name__)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def load_dop20_image(path: str, downsample_factor: int = 1) -> np.ndarray:
    """
    Load a GeoTIFF orthophoto and return an HxWx3 RGB array (clips to first 3 bands).
    Optionally downsample by an integer factor (nearest for labels, bilinear for imagery).
    """
    t0 = time_start()
    with rasterio.open(path) as src:
        if downsample_factor > 1:
            out_h = src.height // downsample_factor
            out_w = src.width // downsample_factor
            arr = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.bilinear,
            )
        else:
            arr = src.read()
    img = reshape_as_image(arr)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    time_end(f"load_dop20_image[{os.path.basename(path)}]", t0)
    return img


def reproject_labels_to_image(ref_img_path: str, labels_path: str, downsample_factor: int = 1) -> np.ndarray:
    """Reproject a raster label map onto the grid/CRS of a reference image using nearest neighbor."""
    t0 = time_start()
    with rasterio.open(ref_img_path) as ref, rasterio.open(labels_path) as src:
        if downsample_factor > 1:
            dst_width = ref.width // downsample_factor
            dst_height = ref.height // downsample_factor
            dst_transform = ref.transform * ref.transform.scale(
                ref.width / dst_width,
                ref.height / dst_height,
            )
        else:
            dst_width = ref.width
            dst_height = ref.height
            dst_transform = ref.transform
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
                    dst_transform=dst_transform,
                    dst_crs=ref.crs,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    resampling=Resampling.nearest,
                )
            labels_arr = dst.read()
    labels_2d = labels_arr[0]
    time_end(f"reproject_labels_to_image[{os.path.basename(labels_path)} -> {os.path.basename(ref_img_path)}]", t0)
    return labels_2d


def rasterize_vector_labels(vector_path: str | list[str],
                            ref_raster_path: str,
                            burn_value: int = 1,
                            downsample_factor: int = 1) -> np.ndarray:
    """
    Rasterize one or more vector layers onto the reference raster grid and union them.

    - If vector_path is a list, each file is rasterized and combined with logical OR.
    - Auto-reprojects geometries if CRS differs from the reference raster CRS.
    """
    t0 = time_start()
    vector_paths = vector_path if isinstance(vector_path, list) else [vector_path]
    with rasterio.open(ref_raster_path) as src:
        if downsample_factor > 1:
            out_shape = (src.height // downsample_factor, src.width // downsample_factor)
            transform = src.transform * src.transform.scale(
                src.width / out_shape[1],
                src.height / out_shape[0],
            )
        else:
            out_shape = (src.height, src.width)
            transform = src.transform
        raster_crs = src.crs
    gt_mask = np.zeros(out_shape, dtype="uint8")

    for vp in vector_paths:
        with fiona.open(vp, "r") as shp:
            vec_crs = shp.crs
            if not vec_crs:
                logger.warning("vector CRS missing for %s; assuming EPSG:4326 (WGS84)", vp)
                vec_crs = CRS.from_epsg(4326).to_dict()
            transformer = None
            if raster_crs and vec_crs and vec_crs != raster_crs.to_dict():
                logger.info("reprojecting vector geometries from %s -> %s for %s", vec_crs, raster_crs.to_dict(), vp)
                transformer = Transformer.from_crs(vec_crs, raster_crs.to_dict(), always_xy=True)
            shapes = []
            for feat in shp:
                geom = feat["geometry"]
                if transformer is not None:
                    geom_obj = shape(geom)
                    geom_obj = shp_transform(transformer.transform, geom_obj)
                    geom = mapping(geom_obj)
                shapes.append((geom, burn_value))

        if not shapes:
            logger.warning("no geometries found in %s", vp)
            continue

        mask_i = rfeatures.rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        gt_mask = np.maximum(gt_mask, mask_i)
    time_end("rasterize_vector_labels", t0)
    return gt_mask


def build_sh_buffer_mask(labels_sh: np.ndarray, buffer_pixels: int) -> np.ndarray:
    """Dilate SH_2022 raster by buffer_pixels using cv2 (fast) or skimage fallback."""
    t0 = time_start()
    base = labels_sh > 0
    if buffer_pixels <= 0:
        time_end("build_sh_buffer_mask", t0)
        return base
    if _HAS_CV2:
        ksize = 2 * buffer_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        buf = cv2.dilate(base.astype(np.uint8), kernel).astype(bool)
    else:
        buf = dilation(base.astype(bool), disk(buffer_pixels))
    time_end("build_sh_buffer_mask", t0)
    return buf


def export_mask_to_shapefile(mask: np.ndarray, ref_raster_path: str, out_path: str):
    """Vectorize a binary mask and save polygons to a shapefile aligned to ref_raster_path CRS."""
    t0 = time_start()
    mask_uint8 = mask.astype("uint8")
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        crs = src.crs
    shape_generator = rfeatures.shapes(mask_uint8, mask=mask_uint8 == 1, transform=transform)
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with fiona.open(out_path, mode="w", driver="ESRI Shapefile", crs=crs.to_dict() if crs is not None else None, schema=schema) as shp:
        idx = 0
        for geom, value in shape_generator:
            if value != 1:
                continue
            shp.write({"geometry": geom, "properties": {"id": int(idx)}})
            idx += 1
    time_end("export_mask_to_shapefile", t0)
    logger.info("shapefile written to: %s", out_path)


def export_masks_to_shapefile_union(masks_with_refs: list[tuple[np.ndarray, str]], out_path: str):
    """
    Write multiple masks (each tied to its own reference raster) into a single shapefile.

    This does not dissolve overlaps; it simply appends all polygons into one layer.
    """
    t0 = time_start()
    if not masks_with_refs:
        logger.warning("export_masks_to_shapefile_union: no masks provided")
        return
    _, first_ref = masks_with_refs[0]
    with rasterio.open(first_ref) as src:
        crs = src.crs
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx = 0
    with fiona.open(out_path, mode="w", driver="ESRI Shapefile", crs=crs.to_dict() if crs is not None else None, schema=schema) as shp:
        for mask, ref_raster_path in masks_with_refs:
            mask_uint8 = mask.astype("uint8")
            with rasterio.open(ref_raster_path) as src:
                transform = src.transform
            shape_generator = rfeatures.shapes(mask_uint8, mask=mask_uint8 == 1, transform=transform)
            for geom, value in shape_generator:
                if value != 1:
                    continue
                shp.write({"geometry": geom, "properties": {"id": int(idx)}})
                idx += 1
    time_end("export_masks_to_shapefile_union", t0)
    logger.info("union shapefile written to: %s (features=%s)", out_path, idx)


def consolidate_features_for_image(feature_dir: str, image_id: str, output_suffix: str = "_features_full.npy"):
    """Concatenate all tile feature .npy files for an image into a single array; return path."""
    t0 = time_start()
    if not os.path.isdir(feature_dir):
        logger.warning("feature_dir does not exist: %s", feature_dir)
        return None
    prefix = f"{image_id}_y"
    suffix = "_features.npy"
    files = [f for f in os.listdir(feature_dir) if f.startswith(prefix) and f.endswith(suffix)]
    if not files:
        logger.warning("no feature tiles found for image_id=%s in %s", image_id, feature_dir)
        return None
    files = sorted(files)
    feats_list = []
    for fname in files:
        fpath = os.path.join(feature_dir, fname)
        arr = np.load(fpath)
        feats_list.append(arr.reshape(-1, arr.shape[-1]))
    feats_full = np.concatenate(feats_list, axis=0).astype(np.float32)
    out_path = os.path.join(feature_dir, f"{image_id}{output_suffix}")
    np.save(out_path, feats_full)
    time_end(f"consolidate_features_for_image[{image_id}]", t0)
    logger.info("consolidated %s tiles for %s -> %s, shape=%s", len(files), image_id, out_path, feats_full.shape)
    return out_path


def export_best_settings(best_raw_config,
                         best_crf_config,
                         model_name,
                         img_path,
                         img2_path,
                         buffer_m,
                         pixel_size_m,
                         shadow_cfg=None,
                         extra_settings: dict | None = None,
                         best_settings_path: str | None = None):
    """
    Write a minimal YAML with the champion configurations and context (paths, buffer, pixel size).
    """
    best_settings = {
        "best_raw_config": best_raw_config,
        "best_crf_config": best_crf_config,
        "best_shadow_config": shadow_cfg,
        "model_name": model_name,
        "img_a": img_path,
        "img_b": img2_path,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
    }
    if extra_settings:
        best_settings["extra"] = extra_settings
    out_path = best_settings_path or cfg.BEST_SETTINGS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        def _write_yaml(d, indent=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    f.write("  " * indent + f"{k}:\n")
                    _write_yaml(v, indent + 1)
                else:
                    f.write("  " * indent + f"{k}: {v}\n")
        _write_yaml(best_settings)
    logger.info("best settings written to %s", out_path)

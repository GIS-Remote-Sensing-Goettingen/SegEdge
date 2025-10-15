#!/usr/bin/env python3
"""
Fetch a Sentinel-2 L2A scene from Earth Search (AWS) for a POINT-centered,
exact-size AOI (default 4 km × 4 km), build a 12-band cube (B01..B12, no B10),
and run the S2DR3 decoder.

Robust choices
--------------
- STAC search: POINT with BBOX + optional time expansion fallbacks.  (pystac-client)
- Collection auto-pick: sentinel-2-c1-l2a (fallbacks retained).
- stackstac: explicit epsg + resolution + **bounds (projected)** + chunksize.  (API)
- FIX: Use dtype=float64 with fill_value=np.nan (stackstac requires this),
       then cast to float32 after loading (np.nan_to_num).               (API)
- 12 bands by default (B01..B12 minus B10); 10-band mode switch if your SR requires it.

References
----------
- stackstac.stack: epsg / resolution / bounds / chunksize; NaN & dtypes. :contentReference[oaicite:1]{index=1}
- Older doc note: specify epsg/resolution when metadata isn’t uniform. :contentReference[oaicite:2]{index=2}
- pystac-client intersects/bbox search usage. :contentReference[oaicite:3]{index=3}
- pyproj Transformer for bbox reprojection. :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Optional
from datetime import datetime, timedelta
import re
import ast

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from pystac_client import Client
import stackstac
from pyproj import Transformer  # for bbox reprojection

from s2dr3_pipeline import S2DR3Pipeline, load_pipeline

# -------------------------
# Globals / constants
# -------------------------
S2_COLLECTION: Optional[str] = None
S2_CATALOG = "https://earth-search.aws.element84.com/v1"

# 12 L2A bands commonly used: B01..B12 (skip B10)
S2_BANDS_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]

# 10-band set (if your SR expects 10)
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B11", "B12"]

# Mapping from band name to preferred Collection-1 asset suffix
S2_ASSET_SUFFIX = {
    "B01": "60m",
    "B02": "10m",
    "B03": "10m",
    "B04": "10m",
    "B05": "20m",
    "B06": "20m",
    "B07": "20m",
    "B08": "10m",
    "B8A": "20m",
    "B09": "60m",
    "B11": "20m",
    "B12": "20m",
}

_GORS_ASSET_MAP = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}


def _rgb_band_indices(bands: Iterable[str]) -> Optional[Tuple[int, int, int]]:
    index_map = {band: idx for idx, band in enumerate(bands)}
    try:
        return index_map["B04"], index_map["B03"], index_map["B02"]
    except KeyError:
        return None


def _log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().numpy()
    logging.info(
        f"{name} stats | min={arr.min():.5f} max={arr.max():.5f} mean={arr.mean():.5f}"
    )


@dataclass
class SearchParams:
    lon: float
    lat: float
    start_date: str
    end_date: str
    size_m: float
    max_cloud: Optional[float] = None
    expand_days: int = 0
    user_agent: str = "S2DR3-pipeline"


# -------------------------
# Logging
# -------------------------
def _setup_logging(verbosity: int) -> None:
    lvl = logging.WARNING
    if verbosity == 1:
        lvl = logging.INFO
    elif verbosity >= 2:
        lvl = logging.DEBUG
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# -------------------------
# Utilities (STAC + geometry)
# -------------------------
def _open_client(user_agent: str) -> Client:
    logging.debug(f"Opening STAC client: {S2_CATALOG}")
    return Client.open(S2_CATALOG, headers={"User-Agent": user_agent})


def _pick_s2_l2a_collection(client: Client, logger=logging) -> str:
    """Prefer 'sentinel-2-c1-l2a'; fallback to 'sentinel-2-l2a' or legacy name."""
    colls = [c.id for c in client.get_collections()]
    logger.debug(f"Available collections: {colls}")
    if "sentinel-2-c1-l2a" in colls:
        logger.info("Using collection: sentinel-2-c1-l2a (current Collection 1 L2A).")
        return "sentinel-2-c1-l2a"
    if "sentinel-2-l2a" in colls:
        logger.warning("Using collection: sentinel-2-l2a (deprecated).")
        return "sentinel-2-l2a"
    if "sentinel-s2-l2a-cogs" in colls:
        logger.warning("Using legacy collection: sentinel-s2-l2a-cogs.")
        return "sentinel-s2-l2a-cogs"
    raise RuntimeError("No Sentinel-2 L2A collection found on this Earth Search endpoint.")


def _bbox_centered_deg(lon: float, lat: float, size_m: float) -> Tuple[float, float, float, float]:
    """WGS84 bbox [minx, miny, maxx, maxy] of exact size around (lon,lat)."""
    half = size_m / 2.0
    m_per_deg_lat = 110_574.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
    dlon = half / max(m_per_deg_lon, 1e-6)
    dlat = half / m_per_deg_lat
    bbox = (float(lon - dlon), float(lat - dlat), float(lon + dlon), float(lat + dlat))
    logging.debug(f"Exact-size bbox (deg) for size_m={size_m}: {bbox}")
    return bbox


def _bbox_deg_to_epsg_bounds(bbox_deg: Tuple[float, float, float, float], epsg: int) -> Tuple[float, float, float, float]:
    """Project WGS84 bbox to EPSG and return (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bbox_deg
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    xs = [minx, maxx, maxx, minx]
    ys = [miny, miny, maxy, maxy]
    X, Y = [], []
    for x, y in zip(xs, ys):
        xx, yy = transformer.transform(x, y)
        X.append(float(xx)); Y.append(float(yy))
    bounds = (min(X), min(Y), max(X), max(Y))
    logging.debug(f"Projected bounds in EPSG:{epsg}: {bounds}")
    return bounds


def _resolve_assets(item, bands: Iterable[str]) -> List[str]:
    available = set(item.assets.keys())
    logging.debug(f"Item assets: {sorted(available)}")
    resolved = []
    for band in bands:
        if band in available:
            resolved.append(band)
            continue
        suffix = S2_ASSET_SUFFIX.get(band)
        candidate = f"{band}_{suffix}" if suffix else None
        if candidate and candidate in available:
            resolved.append(candidate)
            continue
        gors_key = _GORS_ASSET_MAP.get(band)
        if gors_key and gors_key in available:
            resolved.append(gors_key)
            continue
        raise RuntimeError(
            f"Band {band} not present in item assets; available keys: {sorted(available)}"
        )
    return resolved


def _query_items(
    client: Client,
    intersects_geom: Optional[dict],
    datetime_range: str,
    max_items: int = 120,
    max_cloud: Optional[float] = None,
) -> List:
    """Search Earth Search for S2 L2A (POINT/BBOX, optional cloud filter)."""
    query = None
    if max_cloud is not None:
        query = {"eo:cloud_cover": {"lt": max_cloud}}

    logging.info(
        f"STAC search | coll={S2_COLLECTION} | intersects={'geom' if intersects_geom else 'None'} | "
        f"datetime={datetime_range} | max_cloud={max_cloud}"
    )
    search = client.search(
        collections=[S2_COLLECTION],
        intersects=intersects_geom,
        query=query,
        datetime=datetime_range,
        limit=max_items,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    items = list(search.items())
    logging.info(f"Search returned {len(items)} item(s).")
    return items


def _summarize_items(items: List, max_n: int = 10) -> None:
    if not items:
        logging.warning("No items to summarize.")
        return
    logging.info("Top candidate items (most recent first):")
    for i, it in enumerate(items[:max_n]):
        props = it.properties
        dt = props.get("datetime")
        cc = props.get("eo:cloud_cover")
        mgrs = props.get("s2:mgrs_tile")
        logging.info(f"  {i+1:02d}) date={str(dt)[:10]} cloud%={cc} mgrs={mgrs} id={it.id}")


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """UTM EPSG from lon/lat (ignores Norway/Svalbard special cases)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    north = lat >= 0.0
    return (32600 + zone) if north else (32700 + zone)


def _pick_epsg_for_item(item, lon: float, lat: float, override_epsg: Optional[int]) -> int:
    """Use --epsg > proj:epsg > MGRS > lon/lat-derived UTM."""
    if override_epsg:
        logging.info(f"Using override EPSG={override_epsg}.")
        return int(override_epsg)

    props = getattr(item, "properties", {}) or {}
    epsg = props.get("proj:epsg")
    if epsg is not None:
        logging.info(f"Using item proj:epsg={epsg}.")
        return int(epsg)

    mgrs = props.get("s2:mgrs_tile")
    if mgrs:
        m = re.match(r"^(\d{2})([C-X])", mgrs.upper())
        if m:
            zone = int(m.group(1))
            north = m.group(2) >= "N"
            epsg = 32600 + zone if north else 32700 + zone
            logging.info(f"Derived EPSG={epsg} from MGRS={mgrs}.")
            return int(epsg)

    epsg = _utm_epsg_from_lonlat(lon, lat)
    logging.info(f"Derived EPSG={epsg} from lon/lat=({lon:.6f},{lat:.6f}).")
    return int(epsg)


def _select_best_item(params: SearchParams) -> Tuple[object, Tuple[float, float, float, float]]:
    """POINT → BBOX → expanded-window fallbacks; returns (item, bbox_deg)."""
    client = _open_client(params.user_agent)

    global S2_COLLECTION
    if S2_COLLECTION is None:
        S2_COLLECTION = _pick_s2_l2a_collection(client)

    dt_range = f"{params.start_date}/{params.end_date}"
    point_geom = {"type": "Point", "coordinates": [params.lon, params.lat]}
    bbox_deg = _bbox_centered_deg(params.lon, params.lat, params.size_m)

    items = _query_items(client, point_geom, dt_range, max_cloud=params.max_cloud)
    if items:
        _summarize_items(items)
        logging.info("Selecting most recent POINT-intersecting item.")
        return items[0], bbox_deg

    logging.warning("No items for POINT. Trying BBOX polygon...")
    minx, miny, maxx, maxy = bbox_deg
    poly = {"type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}

    items = _query_items(client, poly, dt_range, max_cloud=params.max_cloud)
    if items:
        _summarize_items(items)
        logging.info("Selecting most recent BBOX-intersecting item.")
        return items[0], bbox_deg

    if params.expand_days > 0:
        logging.warning(f"No items. Expanding window by {params.expand_days} days backward.")
        new_start = (datetime.fromisoformat(params.start_date) - timedelta(days=params.expand_days)).date().isoformat()
        dt_range2 = f"{new_start}/{params.end_date}"

        items = _query_items(client, point_geom, dt_range2, max_cloud=params.max_cloud)
        if items:
            _summarize_items(items)
            logging.info("Selecting most recent POINT after expansion.")
            return items[0], bbox_deg

        items = _query_items(client, poly, dt_range2, max_cloud=params.max_cloud)
        if items:
            _summarize_items(items)
            logging.info("Selecting most recent BBOX after expansion.")
            return items[0], bbox_deg

    raise RuntimeError("No S2 L2A items found after POINT, BBOX, and optional expansion.")


# -------------------------
# IO & pipeline
# -------------------------
def _load_cube_from_item(
    item,
    bands: Iterable[str],
    resolution: float,
    bounds_lonlat: Tuple[float, float, float, float],
    chunksize: Optional[object],
    lon: float,
    lat: float,
    override_epsg: Optional[int],
) -> Tuple[torch.Tensor, Affine, int]:
    """
    Clip to exact AOI and stack with explicit grid.
    IMPORTANT FIX: dtype='float64' with fill_value=np.nan to satisfy stackstac,
    then cast to float32 after loading.
    """
    epsg_out = _pick_epsg_for_item(item, lon, lat, override_epsg)
    bounds_epsg = _bbox_deg_to_epsg_bounds(bounds_lonlat, epsg_out)

    logging.info("Preparing stackstac array...")
    logging.debug(f"bounds (EPSG:{epsg_out})={bounds_epsg}, resolution={resolution}, "
                  f"bands={bands}, chunksize={chunksize}")

    resolved_assets = _resolve_assets(item, bands)
    logging.debug(f"Resolved asset order: {resolved_assets}")

    da = stackstac.stack(
        [item],
        assets=resolved_assets,
        epsg=epsg_out,
        resolution=resolution,
        dtype="float64",
        fill_value=np.nan,
        bounds=bounds_epsg,
        chunksize=chunksize,
    )

    if da.sizes.get("time", 0) == 0 or da.sizes.get("y", 0) == 0 or da.sizes.get("x", 0) == 0:
        raise RuntimeError(
            "Stacked array is empty for the chosen AOI/resolution. Try increasing --size-m, "
            "loosening the date window, or checking the item coverage."
        )

    da = da.isel(time=0, band=slice(0, len(bands)))
    da = da.assign_coords(band=list(bands))

    logging.debug(f"xarray dims: {da.dims} sizes: {da.sizes}")
    transform = da.attrs.get("transform")
    epsg = da.attrs.get("epsg") or epsg_out
    if transform is None:
        raise RuntimeError("stackstac DataArray missing 'transform'.")
    logging.info(f"Found EPSG={epsg}.")
    logging.debug(f"Affine transform: {transform}")

    # Load into memory, replace NaNs with 0, then cast to float32 for the model
    data64 = da.transpose("band", "y", "x").data
    if hasattr(data64, "compute"):
        data64 = data64.compute()
    cube_np = data64
    np.nan_to_num(cube_np, copy=False)
    cube_np = cube_np.astype("float32", copy=False)
    logging.info(f"Loaded cube shape: {cube_np.shape} (bands, H, W)")

    return torch.from_numpy(cube_np).unsqueeze(0), transform, int(epsg)


def _write_geotiff(
    path: Path,
    array: torch.Tensor,
    transform_in: Affine,
    epsg: int,
    sr_scale: float,
    dtype: str,
    scale_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Write GeoTIFF with transform scaled by SR factor."""
    arr = array.squeeze(0).cpu().numpy()
    bands, height, width = arr.shape

    if sr_scale <= 0:
        raise ValueError("--sr-scale must be positive.")

    transform_sr = transform_in * Affine.scale(1.0 / sr_scale, 1.0 / sr_scale)

    if dtype == "uint16":
        if scale_range is not None:
            src_min, src_max = scale_range
        else:
            src_min, src_max = float(arr.min()), float(arr.max())
        logging.info(f"Scaling output to uint16 using range [{src_min}, {src_max}]")
        arr = np.clip((arr - src_min) / max(src_max - src_min, 1e-6), 0.0, 1.0)
        arr = (arr * 65535.0 + 0.5).astype(np.uint16, copy=False)
        out_dtype = "uint16"
        predictor = 2
    else:
        arr = arr.astype("float32", copy=False)
        out_dtype = "float32"
        predictor = 3

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": out_dtype,
        "crs": f"EPSG:{epsg}",
        "transform": transform_sr,
        "compress": "deflate",
        "predictor": predictor,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    logging.info(f"Writing GeoTIFF: {path} | shape={arr.shape} | sr_scale={sr_scale} | dtype={out_dtype}")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # spatial targeting by coordinates
    parser.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    # temporal window
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD); default=today (UTC)")
    parser.add_argument("--expand-days", type=int, default=0, help="If no hits, expand window backward by this many days and retry.")
    parser.add_argument("--max-cloud", type=float, default=None, help="Optional max eo:cloud_cover (0..100).")
    # AOI size (exact square)
    parser.add_argument("--size-m", type=float, default=4000.0, help="Square AOI size in meters (default 4000 = 4 km).")
    # band set
    parser.add_argument("--bands-mode", type=str, choices=["12", "10"], default="12",
                        help="12 bands (B01..B12, no B10) or 10 bands (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12).")
    # pipeline bits
    parser.add_argument("--backbone", required=True, help="Path to gedrm_*.safetensors")
    parser.add_argument("--weights-dir", required=True, help="Directory of streamed decoder weights")
    parser.add_argument("--output", required=True, help="Destination GeoTIFF for super-res output")
    parser.add_argument("--lr-output", default=None, help="Optional GeoTIFF path to save the low-resolution cube before SR")
    parser.add_argument("--lr-rgb-output", default=None, help="Optional RGB GeoTIFF (B04/B03/B02) for low-resolution cube")
    parser.add_argument("--sr-rgb-output", default=None, help="Optional RGB GeoTIFF (B04/B03/B02) for super-res output")
    parser.add_argument("--resolution", type=float, default=10.0, help="Ingest resolution in metres (10 reprojects 20/60m bands).")
    parser.add_argument("--sr-scale", type=float, default=4.0, help="SR scale factor to update transform (e.g., 4).")
    parser.add_argument("--device", default="cpu", help="torch device for inference (e.g., 'cpu', 'cuda', 'cuda:0').")
    parser.add_argument("--output-dtype", choices=["float32", "uint16"], default="float32",
                        help="Output GeoTIFF data type. Use 'uint16' with scaling to shrink file size.")
    parser.add_argument("--output-scale", nargs=2, type=float, metavar=("SRC_MIN", "SRC_MAX"),
                        help="Optional range to scale floats when --output-dtype=uint16 (values outside are clipped).")
    parser.add_argument("--epsg", type=int, default=None,
                        help="Force output EPSG (e.g., 32632). If omitted, infer from item or lon/lat.")
    # stackstac chunking
    parser.add_argument("--chunksize", default="1024",
                        help="stackstac 'chunksize' (e.g., 1024, '(1,1,512,512)', 'auto', '15 MB').")
    # logging
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v: info, -vv: debug")
    args = parser.parse_args()
    args.chunksize = _parse_chunksize_arg(args.chunksize)
    return args


def _parse_chunksize_arg(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = int(value)
        return {"y": parsed, "x": parsed}
    text = str(value).strip()
    if text.lower() in {"none", ""}:
        return None
    if text.lower() == "auto":
        return "auto"
    if text.lower().endswith("mb"):
        return text
    try:
        parsed = int(text)
        return {"y": parsed, "x": parsed}
    except ValueError:
        pass
    if text.startswith("(") or text.startswith("["):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return text


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    try:
        target_device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        logging.error(f"Invalid --device '{args.device}': {exc}")
        raise
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA device is available.")
    if args.output_scale and args.output_dtype != "uint16":
        logging.warning("--output-scale is ignored unless --output-dtype=uint16")

    # Default window: last 180 days up to today (UTC)
    if args.end is None:
        args.end = datetime.utcnow().date().isoformat()
    if args.start is None:
        start_dt = datetime.fromisoformat(args.end) - timedelta(days=180)
        args.start = start_dt.date().isoformat()

    # choose band list
    bands = S2_BANDS_12 if args.bands_mode == "12" else S2_BANDS_10
    rgb_indices = _rgb_band_indices(bands)
    if rgb_indices is None and (args.lr_rgb_output or args.sr_rgb_output):
        logging.warning("RGB outputs requested but required bands (B04,B03,B02) are missing from the selected band set.")
    logging.info(f"Band set: {bands}")

    logging.info(f"Search window: {args.start} .. {args.end}")
    params = SearchParams(
        lon=args.lon,
        lat=args.lat,
        start_date=args.start,
        end_date=args.end,
        size_m=args.size_m,
        max_cloud=args.max_cloud,
        expand_days=args.expand_days,
    )

    # Select item with fallbacks
    try:
        item, bbox_deg = _select_best_item(params)
    except Exception:
        logging.error("Selection failed.", exc_info=True)
        raise

    # Report chosen item
    props = item.properties
    logging.info(
        f"Chosen item: id={item.id} date={props.get('datetime')} cloud%={props.get('eo:cloud_cover')} mgrs={props.get('s2:mgrs_tile')}"
    )

    # Load cube
    try:
        cube, transform, epsg = _load_cube_from_item(
            item=item,
            bands=bands,
            resolution=args.resolution,
            bounds_lonlat=bbox_deg,
            chunksize=args.chunksize,
            lon=args.lon,
            lat=args.lat,
            override_epsg=args.epsg,
        )
    except Exception:
        logging.error("stackstac load failed.", exc_info=True)
        raise

    if args.lr_output:
        try:
            lr_path = Path(args.lr_output).expanduser()
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            _write_geotiff(
                lr_path,
                cube,
                transform,
                epsg,
                sr_scale=1.0,
                dtype="float32",
            )
            logging.info(f"Low-resolution cube written to {lr_path}")
        except Exception:
            logging.error("Failed to write low-resolution GeoTIFF.", exc_info=True)
            raise
    if args.lr_rgb_output and rgb_indices is not None:
        try:
            lr_rgb_path = Path(args.lr_rgb_output).expanduser()
            lr_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            lr_rgb = cube[:, list(rgb_indices), :, :]
            _write_geotiff(
                lr_rgb_path,
                lr_rgb,
                transform,
                epsg,
                sr_scale=1.0,
                dtype="float32",
            )
            logging.info(f"Low-resolution RGB written to {lr_rgb_path}")
        except Exception:
            logging.error("Failed to write low-resolution RGB GeoTIFF.", exc_info=True)
            raise

    # Load pipeline + run SR
    try:
        artifacts = load_pipeline(
            backbone_ckpt=Path(args.backbone).expanduser(),
            decoder_dir=Path(args.weights_dir).expanduser(),
            device=target_device,
        )
        pipeline = S2DR3Pipeline(artifacts, device=target_device)
        pipeline_device = target_device
        logging.info(f"S2DR3 pipeline ready (device={pipeline_device}).")
    except Exception:
        logging.error("Failed to initialize S2DR3 pipeline.", exc_info=True)
        raise

    # IMPORTANT: if your decoder expects 10 channels but you requested 12,
    # re-run with --bands-mode 10 or adapt the model.
    try:
        with torch.inference_mode():
            sr = pipeline.super_resolve(cube)
        logging.info(f"SR cube shape: {tuple(sr.shape)} (1, bands, H_sr, W_sr)")
        _log_tensor_stats("LR cube", cube)
        _log_tensor_stats("SR cube", sr)
    except Exception:
        logging.error("Super-resolution failed (channel mismatch?). "
                      "If your model expects 10 bands, run again with --bands-mode 10.", exc_info=True)
        raise

    # Write output
    try:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_geotiff(
            output_path,
            sr,
            transform,
            epsg,
            sr_scale=args.sr_scale,
            dtype=args.output_dtype,
            scale_range=tuple(args.output_scale) if args.output_scale else None,
        )
        logging.info(f"Super-resolved cube written to {output_path}")
        print(str(output_path))  # simple path echo for pipelines
    except Exception:
        logging.error("Failed to write GeoTIFF.", exc_info=True)
        raise

    if args.sr_rgb_output and rgb_indices is not None:
        try:
            sr_rgb_path = Path(args.sr_rgb_output).expanduser()
            sr_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            sr_rgb = sr[:, list(rgb_indices), :, :]
            _write_geotiff(
                sr_rgb_path,
                sr_rgb,
                transform,
                epsg,
                sr_scale=args.sr_scale,
                dtype=args.output_dtype,
                scale_range=tuple(args.output_scale) if args.output_scale else None,
            )
            logging.info(f"Super-resolved RGB written to {sr_rgb_path}")
        except Exception:
            logging.error("Failed to write super-resolved RGB GeoTIFF.", exc_info=True)
            raise


if __name__ == "__main__":
    main()

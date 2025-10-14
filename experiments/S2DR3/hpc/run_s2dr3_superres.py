#!/usr/bin/env python3
"""
Download a Sentinel-2 L2A scene from the public STAC catalog and run the
reverse-engineered S2DR3 super-resolution decoder on a cropped 10-band cube.

Example
-------
python run_s2dr3_superres.py \
    --tile 32UNC \
    --date 2025-04-15 \
    --weights-dir ~/.cache/s2dr3/sandbox/weights \
    --backbone ~/.cache/s2dr3/sandbox/local/S2DR3/gedrm_17igu4jxui \
    --crop-size 512 \
    --output /tmp/s2dr3_superres.tif

Requirements
------------
- pystac-client
- stackstac
- xarray, rasterio
- PyTorch (already required by the decoder/backbone)

Notes
-----
* The script pulls the 10 multispectral bands (B02–B12) and resamples everything
  to the requested resolution (default 10 m).  It crops the tile to keep memory
  usage under control before handing the cube to the decoder.
* Only the decoder is used here; the ViT backbone is loaded so you can reuse the
  tokens if needed, but super-resolution operates solely on the multispectral
  cube.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio
from datetime import datetime, timedelta

from pystac_client import Client
import stackstac
import torch
from rasterio.transform import Affine

from s2dr3_pipeline import S2DR3Pipeline, load_pipeline

S2_COLLECTION = "sentinel-s2-l2a-cogs"
S2_CATALOG = "https://earth-search.aws.element84.com/v1"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def _select_item(tile: str, date: str):
    client = Client.open(S2_CATALOG, headers={"User-Agent": "s2dr3-research/0.1"})
    mgrs = tile.upper().lstrip("T")
    target = datetime.fromisoformat(date)
    window_start = (target - timedelta(days=180)).isoformat()
    search = client.search(
        collections=[S2_COLLECTION],
        query={"s2:mgrs_tile": {"eq": mgrs}},
        datetime=f"{window_start}/{date}",
        limit=1,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A items for tile {tile} in the last 180 days (up to {date})."
        )
    item = items[0]
    if item.properties["datetime"][:10] != date:
        print(
            f"[info] Requested {date} missing; using {item.properties['datetime'][:10]} instead."
        )
    return item


def _load_cube(
    item,
    bands: Iterable[str],
    resolution: float,
    crop_size: int,
) -> Tuple[torch.Tensor, Affine]:
    da = stackstac.stack(
        [item],
        assets=list(bands),
        resolution=resolution,
        dtype="float32",
        fill_value=np.nan,
        chunks=None,
    ).isel(time=0, band=slice(0, len(bands)))  # dims: (band, y, x)

    if crop_size:
        y_mid = da.sizes["y"] // 2
        x_mid = da.sizes["x"] // 2
        half = crop_size // 2
        da = da.isel(
            y=slice(y_mid - half, y_mid + half),
            x=slice(x_mid - half, x_mid + half),
        )

    data = da.transpose("band", "y", "x").data  # load into memory
    cube = np.nan_to_num(data, nan=0.0)
    transform = da.attrs["transform"]
    return torch.from_numpy(cube).unsqueeze(0), transform  # (1, bands, H, W)


def _write_geotiff(path: Path, array: torch.Tensor, transform: Affine, crs: str) -> None:
    array = array.squeeze(0).cpu().numpy()
    bands, height, width = array.shape
    transform_sr = transform * Affine.scale(1 / 10, 1 / 10)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform_sr,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", required=True, help="Sentinel-2 MGRS tile (e.g. 32UNC)")
    parser.add_argument("--date", required=True, help="Acquisition date (YYYY-MM-DD)")
    parser.add_argument("--backbone", required=True, help="Path to gedrm_*.safetensors")
    parser.add_argument("--weights-dir", required=True, help="Directory of streamed decoder weights")
    parser.add_argument("--output", required=True, help="Destination GeoTIFF for super-res output")
    parser.add_argument("--resolution", type=float, default=10.0, help="Input resolution in metres")
    parser.add_argument("--crop-size", type=int, default=512, help="Square crop size before super-resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item = _select_item(args.tile, args.date)
    cube, transform = _load_cube(item, S2_BANDS, args.resolution, args.crop_size)

    artifacts = load_pipeline(
        backbone_ckpt=Path(args.backbone).expanduser(),
        decoder_dir=Path(args.weights_dir).expanduser(),
    )
    pipeline = S2DR3Pipeline(artifacts)

    with torch.inference_mode():
        sr = pipeline.super_resolve(cube)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geotiff(output_path, sr, transform, item.common_metadata.epsg)
    print(f"Super-resolved cube written to {output_path}")


if __name__ == "__main__":
    main()

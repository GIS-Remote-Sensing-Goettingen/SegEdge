#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform as warp_transform

import opensr_model
import opensr_utils


def compute_centroid_lat_lon(tif_path: Path) -> tuple[float, float]:
    """
    Derive the geographic centroid (latitude, longitude) of a GeoTIFF.

    Steps
    -----
    1. Open the raster and ensure it carries a valid CRS.
    2. Convert the pixel-space midpoint into the raster’s native coordinates.
    3. Reproject the single point into geographic WGS84 (EPSG:4326).

    Args:
        tif_path: Path to the raster file whose centroid should be computed.

    Returns:
        tuple[float, float]: Centroid latitude and longitude in decimal degrees.

    Examples
    --------
    >>> compute_centroid_lat_lon(Path("example.tif"))  # doctest: +SKIP
    (51.5074, -0.1278)
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Dataset at {tif_path} lacks a CRS; cannot compute centroid.")

        center_row = src.height / 2.0
        center_col = src.width / 2.0
        x, y = src.transform * (center_col, center_row)
        lon, lat = warp_transform(src.crs, "EPSG:4326", [x], [y])
        return float(lat[0]), float(lon[0])


def compress_and_rename(src_path: Path, lat: float, lon: float, dest_dir: Path) -> Path:
    """
    Compress the SR output and rename it using the centroid coordinates.

    Steps
    -----
    1. Build a descriptive filename embedding the centroid coordinates.
    2. Copy the original GeoTIFF into the new name using high-level ZSTD compression.
    3. Remove the uncompressed source file to keep storage tidy.

    Args:
        src_path: Path to the uncompressed SR GeoTIFF (typically ``sr.tif``).
        lat: Centroid latitude in decimal degrees.
        lon: Centroid longitude in decimal degrees.
        dest_dir: Directory where the renamed file should be written.

    Returns:
        Path: Filesystem path to the compressed, renamed GeoTIFF.

    Examples
    --------
    >>> compress_and_rename(Path("sr.tif"), 51.5, -0.12, Path("outputs"))  # doctest: +SKIP
    PosixPath('output_SR_image_51.500000_-0.120000.tif')
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"output_SR_image_{lat:.6f}_{lon:.6f}.tif"
    new_path = dest_dir / new_name

    print(f"[INFO] Compressing {src_path} → {new_path}")
    rio_copy(
        src_path,
        new_path,
        driver="GTiff",
        COMPRESS="ZSTD",
        ZSTD_LEVEL=22,
        PREDICTOR=2,
        TILED=True,
        BLOCKXSIZE=512,
        BLOCKYSIZE=512,
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
    )

    print(f"[INFO] Removing original uncompressed file: {src_path}")
    src_path.unlink(missing_ok=True)
    return new_path


def main():
    """
    Orchestrate SR inference, centroid retrieval, and compressed renaming.

    Steps
    -----
    1. Parse CLI arguments and load the LDSR configuration.
    2. Instantiate the SR model and run the large_file_processing helper.
    3. Locate the final ``sr.tif`` output, compute its centroid, and re-encode it.

    Args:
        None explicitly; arguments are consumed from ``sys.argv``.

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tif", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--window-size", type=int, nargs=2, default=(128, 128))
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--eliminate-border-px", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--save-preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    cfg_path = Path(__file__).with_name("config_10m.yaml")
    cfg = OmegaConf.load(cfg_path)

    model = opensr_model.SRLatentDiffusion(cfg, device=device)
    model.load_pretrained(cfg.ckpt_version)
    assert model.training is False

    runner = opensr_utils.large_file_processing(
        root=args.input_tif,
        model=model,
        window_size=tuple(args.window_size),
        factor=args.factor,
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        device=device,
        gpus=args.gpus,
        save_preview=args.save_preview,
        debug=False,
    )

    final_sr_path = getattr(runner, "final_sr_path", None)
    if final_sr_path is None:
        # Fallback: assume default naming inside output directory.
        final_sr_path = Path(args.output_dir) / "sr.tif"
        print(f"[WARN] final_sr_path not set by runner; falling back to {final_sr_path}")
    else:
        final_sr_path = Path(final_sr_path)

    if not final_sr_path.exists():
        raise FileNotFoundError(f"Expected SR output at {final_sr_path} but it does not exist.")

    lat, lon = compute_centroid_lat_lon(final_sr_path)
    print(f"[INFO] Output centroid latitude: {lat:.6f}, longitude: {lon:.6f}")
    output_dir = Path(args.output_dir)
    new_path = compress_and_rename(final_sr_path, lat, lon, output_dir)
    print(f"[INFO] Final SR file available at: {new_path}")


if __name__ == "__main__":
    main()

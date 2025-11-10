#!/usr/bin/env python3
"""
Utility to mosaic per-chip Planet SH label TIFFs into a single GeoTIFF.

Example
-------
python main.py \
    --input-dir "/run/media/mak/Partition of 1TB disk/SH_dataset/Planet_2022_SH_labels" \
    --output "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from osgeo import gdal

logger = logging.getLogger("labels_unifier")

DEFAULT_CREATION_OPTIONS = ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]


def merge_creation_options(user_options: Sequence[str]) -> List[str]:
    merged: dict[str, str] = {}
    for option in DEFAULT_CREATION_OPTIONS:
        key = option.split("=", 1)[0].upper()
        merged[key] = option
    for option in user_options:
        key = option.split("=", 1)[0].upper()
        merged[key] = option
    return list(merged.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine a directory full of chip label GeoTIFFs into a single mosaic."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory that contains the chip GeoTIFFs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("planet_labels_mosaic.tif"),
        help="Destination GeoTIFF for the mosaic (default: %(default)s).",
    )
    parser.add_argument(
        "--vrt-path",
        type=Path,
        default=None,
        help="Optional explicit path for the intermediate VRT. Defaults to <output>.vrt.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search for chips (enabled by default).",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N chips (useful for smoke tests).",
    )
    parser.add_argument(
        "--resample",
        default="nearest",
        choices=["nearest", "average", "bilinear", "cubic", "cubicspline", "lanczos"],
        help="Resampling algorithm passed to GDAL BuildVRT.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=0.0,
        help="NoData value to enforce in the mosaic (default: %(default)s).",
    )
    parser.add_argument(
        "--co",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra GDAL creation option for the output GeoTIFF (repeatable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they exist.",
    )
    parser.add_argument(
        "--keep-vrt",
        action="store_true",
        help="Keep the intermediate VRT instead of deleting it after translation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list how many chips would be mosaicked, without running GDAL.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def collect_sources(input_dir: Path, recursive: bool, limit: int | None) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    patterns = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    candidates: List[Path] = []
    for pattern in patterns:
        iterator: Iterable[Path]
        if recursive:
            iterator = input_dir.rglob(pattern)
        else:
            iterator = input_dir.glob(pattern)
        for path in iterator:
            if path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            if path.name.endswith(".aux.xml"):
                continue
            if not path.is_file():
                continue
            candidates.append(path.resolve())
            if limit is not None and len(candidates) >= limit:
                logger.debug("Limit reached (%d files); stopping search.", limit)
                return sorted(candidates)
    return sorted(candidates)


def ensure_write_target(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to replace it."
            )
        if path.is_dir():
            raise IsADirectoryError(f"{path} is a directory, expected a file path.")
        logger.warning("Overwriting existing file: %s", path)
        path.unlink()
    if not path.parent.exists():
        logger.debug("Creating destination directory %s", path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)


def infer_gdal_dtype(sample_path: Path) -> int:
    dataset = gdal.Open(str(sample_path))
    if dataset is None:
        raise RuntimeError(f"Failed to open sample dataset {sample_path}")
    band = dataset.GetRasterBand(1)
    dtype = band.DataType
    dataset = None
    return dtype


def build_vrt(vrt_path: Path, sources: Sequence[Path], nodata: float, resample: str) -> None:
    logger.info("Building VRT with %d source files -> %s", len(sources), vrt_path)
    ensure_write_target(vrt_path, overwrite=True)
    options = gdal.BuildVRTOptions(
        resampleAlg=resample,
        srcNodata=nodata,
        VRTNodata=nodata,
    )
    result = gdal.BuildVRT(str(vrt_path), [str(p) for p in sources], options=options)
    if result is None:
        raise RuntimeError("GDAL.BuildVRT returned None (see logs above for details).")
    result.FlushCache()
    result = None


def translate_to_gtiff(
    vrt_path: Path,
    output_path: Path,
    nodata: float,
    gdal_dtype: int,
    creation_options: Sequence[str],
    overwrite: bool,
) -> None:
    logger.info("Translating VRT -> %s", output_path)
    ensure_write_target(output_path, overwrite=overwrite)
    co = merge_creation_options(creation_options)
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=co,
        outputType=gdal_dtype,
        noData=nodata,
    )
    result = gdal.Translate(str(output_path), str(vrt_path), options=translate_options)
    if result is None:
        raise RuntimeError("GDAL.Translate returned None (see logs above for details).")
    result.FlushCache()
    result = None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    gdal.UseExceptions()

    try:
        sources = collect_sources(args.input_dir, args.recursive, args.limit)
    except Exception as exc:
        logger.error("Failed to gather source TIFFs: %s", exc)
        return 1

    if not sources:
        logger.error("No GeoTIFF chips found in %s", args.input_dir)
        return 1

    logger.info("Discovered %d chip(s).", len(sources))
    if args.dry_run:
        logger.info("Dry-run requested, exiting before running GDAL.")
        return 0

    output_path = args.output.resolve()
    vrt_path = (
        args.vrt_path.resolve()
        if args.vrt_path is not None
        else output_path.with_suffix(".vrt")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.keep_vrt:
        # We'll always rebuild; ensure leftover file doesn't trip up BuildVRT.
        if vrt_path.exists():
            logger.debug("Removing pre-existing VRT: %s", vrt_path)
            vrt_path.unlink()

    try:
        build_vrt(vrt_path, sources, args.nodata, args.resample)
        dtype = infer_gdal_dtype(sources[0])
        translate_to_gtiff(
            vrt_path,
            output_path,
            args.nodata,
            dtype,
            creation_options=args.co,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        logger.error("Failed to build mosaic: %s", exc)
        return 1
    finally:
        if not args.keep_vrt and vrt_path.exists():
            logger.debug("Cleaning up temporary VRT %s", vrt_path)
            vrt_path.unlink()

    logger.info("All done. Mosaic written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

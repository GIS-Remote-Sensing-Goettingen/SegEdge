#!/usr/bin/env python3
"""
Drive pipeline.sh over a grid of Sentinel-2 cutouts.

This helper script receives two extreme geographic coordinates (any order) and
expands them into a grid of square patches sized according to the EDGE_SIZE
used by pipeline.sh. Each patch is launched sequentially by exporting the
expected LATITUDE/LONGITUDE/EDGE_SIZE environment variables before invoking
pipeline.sh.

The script prints detailed progress information because it is intended to run
on a cluster where interactive debugging is harder.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List

DEFAULT_RESOLUTION_METERS = 10.0  # Default Sentinel-2 pixel size (meters) at 10 m resolution
OVERLAP_METERS = 128.0  # Requested overlap between adjacent patches


@dataclass(frozen=True)
class Patch:
    """Container describing a single patch launch."""

    row_index: int
    row_count: int
    column_index: int
    column_count: int
    latitude: float
    longitude: float
    edge_size: int


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """
    Configure and parse command-line arguments.

    Steps
    -----
    1. Instantiate an ``ArgumentParser`` with a descriptive help message.
    2. Register every CLI option that mirrors a pipeline.sh environment variable.
    3. Parse the provided iterable into a populated :class:`argparse.Namespace`.

    Args:
        argv: Iterable containing the raw command-line tokens to parse.

    Returns:
        argparse.Namespace: Object with attributes for every supported option.

    Examples
    --------
    >>> ns = parse_args(["--lat1", "0", "--lon1", "0", "--lat2", "1", "--lon2", "1"])
    >>> ns.lat1, ns.lon2
    (0.0, 1.0)
    """
    # Accept optional overrides for all pipeline.sh environment variables.
    ap = argparse.ArgumentParser(
        description="Invoke pipeline.sh across a grid of patches defined by two extreme coordinates."
    )
    ap.add_argument(
        "--lat1",
        type=float,
        required=True,
        help="Latitude of the first extreme corner (degrees).",
    )
    ap.add_argument(
        "--lon1",
        type=float,
        required=True,
        help="Longitude of the first extreme corner (degrees).",
    )
    ap.add_argument(
        "--lat2",
        type=float,
        required=True,
        help="Latitude of the opposite extreme corner (degrees).",
    )
    ap.add_argument(
        "--lon2",
        type=float,
        required=True,
        help="Longitude of the opposite extreme corner (degrees).",
    )
    ap.add_argument(
        "--edge-size",
        type=int,
        default=4096,
        help="Patch edge size in pixels (default: 4096).",
    )
    ap.add_argument(
        "--pipeline",
        default="pipeline.sh",
        help="Path to pipeline.sh (default: ./pipeline.sh).",
    )
    ap.add_argument(
        "--start-date",
        help="Override START_DATE passed to pipeline.sh (optional).",
    )
    ap.add_argument(
        "--end-date",
        help="Override END_DATE passed to pipeline.sh (optional).",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        help="Override RESOLUTION passed to pipeline.sh (meters per pixel; default: 10).",
    )
    ap.add_argument(
        "--env-path",
        help="Override SEGEDGE_CONDA_ENV passed to pipeline.sh (optional).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print patch information without invoking pipeline.sh.",
    )
    return ap.parse_args(list(argv))


def meters_to_lat_deg(distance_m: float) -> float:
    """
    Approximate conversion from meters to degrees latitude.

    Steps
    -----
    1. Assume the average meridional scale of approximately 111,320 meters/degree.
    2. Divide the requested distance by that constant to obtain degrees.

    Args:
        distance_m: Distance in meters along the north-south direction.

    Returns:
        float: Estimated angular displacement in decimal degrees.

    Examples
    --------
    >>> round(meters_to_lat_deg(111_320), 3)
    1.0
    """
    # Latitude degrees per meter are constant enough for this workflow.
    return distance_m / 111_320.0


def meters_to_lon_deg(distance_m: float, latitude_deg: float) -> float:
    """
    Approximate conversion from meters to degrees longitude at a given latitude.

    Steps
    -----
    1. Compute the number of meters per degree of longitude using ``cos(latitude)``.
    2. Validate that the denominator is positive, avoiding singularities near poles.
    3. Divide the distance by the computed scale to produce degrees.

    Args:
        distance_m: East-west separation in meters.
        latitude_deg: Latitude (degrees) at which to evaluate the scale factor.

    Returns:
        float: Estimated longitudinal displacement in decimal degrees.

    Examples
    --------
    >>> round(meters_to_lon_deg(111_320, 0.0), 3)
    1.0
    >>> round(meters_to_lon_deg(111_320, 60.0), 3)
    2.0
    """
    # Longitude degrees shrink with cosine(latitude); guard against poles.
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude_deg))
    if meters_per_degree_lon <= 0:
        raise ValueError(
            f"Cannot compute longitude degrees at latitude {latitude_deg:.6f}; check the coordinates."
        )
    return distance_m / meters_per_degree_lon


def clamp_center(
    suggested_center: float, min_value: float, max_value: float, half_extent: float
) -> float:
    """
    Ensure the patch center stays within bounds while still covering the edges.

    The center is clamped so that the square defined by center ± half_extent
    continues to cover the requested range.

    Steps
    -----
    1. Compute the feasible interval for the patch center.
    2. Handle the degenerate case where the requested span is smaller than the patch.
    3. Clamp the suggested center into the valid interval.

    Args:
        suggested_center: Desired center coordinate prior to clamping.
        min_value: Minimum extent that must be covered by the patch.
        max_value: Maximum extent that must be covered by the patch.
        half_extent: Half-width (in degrees) of the patch.

    Returns:
        float: Adjusted center coordinate that satisfies the coverage constraint.

    Examples
    --------
    >>> clamp_center(5.0, 0.0, 10.0, 3.0)
    5.0
    >>> clamp_center(1.0, 0.0, 4.0, 3.0)
    2.0
    """
    # Determine the allowable interval for the patch center.
    lower_limit = min_value + half_extent
    upper_limit = max_value - half_extent
    if lower_limit > upper_limit:
        # Requested span is smaller than the patch size; place the center midway.
        return (min_value + max_value) / 2.0
    return max(lower_limit, min(suggested_center, upper_limit))


def compute_centers(
    min_value: float, max_value: float, patch_deg: float, step_deg: float
) -> List[float]:
    """
    Produce evenly spaced patch centers covering the requested span.

    Steps
    -----
    1. Validate that the patch and step sizes are positive.
    2. Return a single clamped center if one patch already covers the span.
    3. Iterate with the requested stride to generate and clamp additional centers.
    4. Deduplicate centers produced by clamping near the edges.

    Args:
        min_value: Lower extreme of the interval (degrees).
        max_value: Upper extreme of the interval (degrees).
        patch_deg: Full patch size expressed in degrees.
        step_deg: Desired step between centers in degrees.

    Returns:
        list[float]: Sorted list of center coordinates covering the span.

    Examples
    --------
    >>> compute_centers(0.0, 1.0, 0.6, 0.5)
    [0.3, 0.7]
    >>> compute_centers(0.0, 0.0, 0.6, 0.5)
    [0.0]
    """
    if patch_deg <= 0 or step_deg <= 0:
        raise ValueError("patch_deg and step_deg must be positive.")

    half_extent = patch_deg / 2.0
    span = max(0.0, max_value - min_value)

    if span <= patch_deg:
        # A single patch can cover the span.
        center = (min_value + max_value) / 2.0
        return [clamp_center(center, min_value, max_value, half_extent)]

    # Determine how many steps we need so that the final patch still covers the max.
    count = int(math.ceil((span - patch_deg) / step_deg)) + 1
    centers: List[float] = []
    for index in range(count):
        candidate = min_value + half_extent + index * step_deg
        candidate = clamp_center(
            candidate, min_value, max_value, half_extent
        )
        if centers and abs(candidate - centers[-1]) < 1e-12:
            continue
        centers.append(candidate)

    if not centers:
        center = (min_value + max_value) / 2.0
        centers.append(
            clamp_center(center, min_value, max_value, half_extent)
        )

    return centers


def build_patches(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    edge_size: int,
    resolution_m: float,
) -> List[Patch]:
    """
    Expand the bounding box into a list of patch descriptors.

    Steps
    -----
    1. Normalize corner inputs into north/south and west/east extremes.
    2. Translate the pixel edge size into physical meters and angular degrees using ``resolution_m``.
    3. Compute latitude centers, then per-row longitude centers with overlap.
    4. Create :class:`Patch` instances for every grid position.

    Args:
        lat1: Latitude of one extreme corner (degrees).
        lon1: Longitude of the same extreme corner (degrees).
        lat2: Latitude of the opposite extreme corner (degrees).
        lon2: Longitude of the opposite extreme corner (degrees).
        edge_size: Patch edge length in pixels.
        resolution_m: Pixel size in meters used to convert the edge into ground distance.

    Returns:
        list[Patch]: Patches ordered row-wise (north to south, west to east).

    Examples
    --------
    >>> patches = build_patches(0.0, 0.0, 0.0, 0.0, 512, 10.0)
    [DEBUG] Bounding box (lat): 0.0 → 0.0
    [DEBUG] Bounding box (lon): 0.0 → 0.0
    [DEBUG] Pixel resolution (meters): 10.0
    [DEBUG] Patch size (meters): 5120.0
    [DEBUG] Patch size (deg lat): 0.045993532159540065
    [DEBUG] Estimated number of latitude rows: 1
    [DEBUG] Row 1/1: lat_center=0.000000, patch_lon_deg=0.045994, lon_step_deg=0.044844, lon columns=1
    >>> len(patches)
    1
    """
    # Ensure consistent ordering of extremes.
    lat_min, lat_max = sorted((lat1, lat2))
    lon_min, lon_max = sorted((lon1, lon2))

    if edge_size <= 0:
        raise ValueError(f"EDGE_SIZE must be positive; received {edge_size}.")

    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive; received {resolution_m}.")

    print("[DEBUG] Bounding box (lat):", lat_min, "→", lat_max)
    print("[DEBUG] Bounding box (lon):", lon_min, "→", lon_max)
    print("[DEBUG] Pixel resolution (meters):", resolution_m)

    patch_size_m = edge_size * resolution_m
    if OVERLAP_METERS >= patch_size_m:
        raise ValueError(
            f"Overlap {OVERLAP_METERS} m must be smaller than patch size {patch_size_m} m."
        )

    patch_step_m = patch_size_m - OVERLAP_METERS
    patch_lat_deg = meters_to_lat_deg(patch_size_m)
    lat_step_deg = meters_to_lat_deg(patch_step_m)
    lat_centers = compute_centers(lat_min, lat_max, patch_lat_deg, lat_step_deg)
    lat_rows = len(lat_centers)

    patches: List[Patch] = []

    print("[DEBUG] Patch size (meters):", patch_size_m)
    print("[DEBUG] Patch size (deg lat):", patch_lat_deg)
    print("[DEBUG] Estimated number of latitude rows:", lat_rows)

    for row_index, lat_center in enumerate(lat_centers):
        # Longitude degrees depend on latitude, recompute per row.
        patch_lon_deg = meters_to_lon_deg(patch_size_m, lat_center)
        lon_step_deg = meters_to_lon_deg(patch_step_m, lat_center)
        lon_centers = compute_centers(lon_min, lon_max, patch_lon_deg, lon_step_deg)
        lon_cols = len(lon_centers)

        print(
            f"[DEBUG] Row {row_index + 1}/{lat_rows}: lat_center={lat_center:.6f}, "
            f"patch_lon_deg={patch_lon_deg:.6f}, lon_step_deg={lon_step_deg:.6f}, "
            f"lon columns={lon_cols}"
        )

        for column_index, lon_center in enumerate(lon_centers):
            # Capture every patch descriptor for later execution.
            patches.append(
                Patch(
                    row_index=row_index,
                    row_count=lat_rows,
                    column_index=column_index,
                    column_count=lon_cols,
                    latitude=lat_center,
                    longitude=lon_center,
                    edge_size=edge_size,
                )
            )

    return patches


def run_patch(
    patch: Patch, args: argparse.Namespace, patch_number: int, total_patches: int
) -> None:
    """
    Launch pipeline.sh for the supplied patch.

    Steps
    -----
    1. Resolve the absolute path to ``pipeline.sh`` and ensure it exists.
    2. Build the environment with per-patch overrides (lat, lon, edge size, etc.).
    3. Select the appropriate command (direct execution or via ``bash``).
    4. Run the command inside the pipeline directory and report the outcome.

    Args:
        patch: Descriptor with geographic coordinates and grid indices.
        args: Parsed CLI namespace providing runtime overrides.
        patch_number: One-based index of the current patch.
        total_patches: Total number of patches that will be launched.

    Returns:
        None

    Raises:
        FileNotFoundError: If ``pipeline.sh`` cannot be found.
        subprocess.CalledProcessError: If the subprocess exits with an error.

    Examples
    --------
    The function prints its status and runs the external script, so doctesting
    the call itself is not practical. Instead, we document the expected inputs:

    >>> dummy_patch = Patch(0, 1, 0, 1, 51.0, -0.1, 512)
    >>> class DummyArgs:
    ...     pipeline = "pipeline.sh"
    ...     start_date = end_date = resolution = env_path = None
    ...     dry_run = False
    ...     resolution_m = 10.0
    >>> run_patch(dummy_patch, DummyArgs(), 1, 1)  # doctest: +SKIP
    """
    # Resolve the pipeline path so we can change directories safely.
    pipeline_path = os.path.abspath(args.pipeline)
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"pipeline.sh not found at {pipeline_path}")
    if not os.access(pipeline_path, os.X_OK):
        print(
            f"[WARN] pipeline.sh at {pipeline_path} is not executable; attempting to run via bash.",
            file=sys.stderr,
        )

    pipeline_dir = os.path.dirname(pipeline_path) or "."
    env = os.environ.copy()
    resolved_resolution = int(round(args.resolution_m))
    env.update(
        {
            "LATITUDE": f"{patch.latitude:.8f}",
            "LONGITUDE": f"{patch.longitude:.8f}",
            "EDGE_SIZE": str(patch.edge_size),
            "RESOLUTION": str(resolved_resolution),
        }
    )

    if args.start_date:
        env["START_DATE"] = args.start_date
    if args.end_date:
        env["END_DATE"] = args.end_date
    if args.env_path:
        env["SEGEDGE_CONDA_ENV"] = args.env_path

    print(
        f"[INFO] Launching patch {patch_number}/{total_patches} "
        f"(row {patch.row_index + 1}/{patch.row_count}, "
        f"col {patch.column_index + 1}/{patch.column_count})"
    )
    print(
        f"[INFO] LAT={patch.latitude:.6f}, LON={patch.longitude:.6f}, "
        f"EDGE_SIZE={patch.edge_size}, RESOLUTION={resolved_resolution}"
    )
    print(f"[INFO] Working directory for pipeline.sh: {pipeline_dir}")

    cmd: List[str]
    if os.access(pipeline_path, os.X_OK):
        cmd = [pipeline_path]
    else:
        cmd = ["bash", pipeline_path]

    subprocess.run(cmd, check=True, cwd=pipeline_dir, env=env)
    print(f"[INFO] Patch {patch_number}/{total_patches} completed successfully.")


def main(argv: Iterable[str]) -> int:
    """
    Entry point for command-line execution.

    Steps
    -----
    1. Parse CLI arguments into a structured namespace.
    2. Compute the grid of patches that covers the requested bounding box.
    3. Either list the patches (dry run) or run ``pipeline.sh`` for each patch.
    4. Return zero on success or the failing subprocess exit code.

    Args:
        argv: Iterable containing command-line arguments (excluding program name).

    Returns:
        int: Zero for success, otherwise non-zero on subprocess failure.

    Examples
    --------
    >>> import contextlib, io
    >>> capture = io.StringIO()
    >>> with contextlib.redirect_stdout(capture):
    ...     exit_code = main(["--lat1", "0", "--lon1", "0", "--lat2", "0", "--lon2", "0", "--dry-run"])
    >>> exit_code
    0
    >>> "Dry run requested; listing patches without execution." in capture.getvalue()
    True
    """
    # Convert the CLI arguments into the patch execution plan.
    args = parse_args(argv)

    print("[DEBUG] Parsed arguments:", args)

    resolution_m = float(args.resolution) if args.resolution is not None else DEFAULT_RESOLUTION_METERS
    args.resolution_m = resolution_m
    patches = build_patches(
        args.lat1,
        args.lon1,
        args.lat2,
        args.lon2,
        args.edge_size,
        resolution_m,
    )
    print(f"[INFO] Total patches to process: {len(patches)}")

    if args.dry_run:
        print("[INFO] Dry run requested; listing patches without execution.")
        for idx, patch in enumerate(patches, start=1):
            print(
                f"[DRY-RUN] Patch {idx}: row {patch.row_index + 1}/{patch.row_count}, "
                f"col {patch.column_index + 1}/{patch.column_count}, "
                f"lat={patch.latitude:.6f}, lon={patch.longitude:.6f}"
            )
        return 0

    total_patches = len(patches)
    for index, patch in enumerate(patches, start=1):
        try:
            run_patch(patch, args, index, total_patches)
        except subprocess.CalledProcessError as exc:
            print(
                f"[ERROR] pipeline.sh failed for lat={patch.latitude:.6f}, "
                f"lon={patch.longitude:.6f} with exit code {exc.returncode}",
                file=sys.stderr,
            )
            return exc.returncode
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR] Unexpected error while running patch at "
                f"lat={patch.latitude:.6f}, lon={patch.longitude:.6f}: {exc}",
                file=sys.stderr,
            )
            return 1

    print("[INFO] All patches completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

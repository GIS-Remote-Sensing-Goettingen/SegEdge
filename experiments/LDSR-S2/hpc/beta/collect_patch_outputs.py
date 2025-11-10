#!/usr/bin/env python3
"""
Collect per-patch SR deliverables into a shared output directory.

Run this after your SLURM jobs finish to copy the final
`output_SR_image_<lat>_<lon>.tif` files (and optionally logs/LR tiles) out of
their individual `output_patches/patch_lat_...` workspaces into a single folder
living next to `pipeline.sh`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Configure command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("output_patches"),
        help="Root directory containing per-patch workspaces (default: ./output_patches).",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).with_name("collected_outputs"),
        help="Destination directory for consolidated files (default: ./collected_outputs).",
    )
    ap.add_argument(
        "--include-logs",
        action="store_true",
        help="Also copy each patch's logs directory into the destination.",
    )
    ap.add_argument(
        "--include-lr",
        action="store_true",
        help="Also copy each patch's data_sentinel2 directory into the destination.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files.",
    )
    return ap.parse_args()


def copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    """Copy an entire directory tree if the source exists."""
    if not src.exists():
        return
    if dry_run:
        print(f"[DRY-RUN] Would copy {src} -> {dst}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    args = parse_args()
    source_root = args.source
    dest_root = args.dest

    if not source_root.exists():
        print(f"[WARN] Source root {source_root} does not exist; nothing to do.")
        return 0

    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for patch_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        outputs_dir = patch_dir / "outputs"
        if not outputs_dir.exists():
            print(f"[SKIP] {patch_dir} has no outputs directory.")
            continue

        for tif_path in outputs_dir.glob("output_SR_image_*.tif"):
            target_path = dest_root / tif_path.name
            if args.dry_run:
                print(f"[DRY-RUN] Would copy {tif_path} -> {target_path}")
            else:
                shutil.copy2(tif_path, target_path)
                print(f"[INFO] Copied {tif_path} -> {target_path}")
            copied += 1

        if args.include_logs:
            copy_tree(patch_dir / "logs", dest_root / f"{patch_dir.name}_logs", args.dry_run)

        if args.include_lr:
            copy_tree(patch_dir / "data_sentinel2", dest_root / f"{patch_dir.name}_data", args.dry_run)

    if copied == 0:
        print("[WARN] No output SR files were found.")
    else:
        print(f"[INFO] Copied {copied} SR files into {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


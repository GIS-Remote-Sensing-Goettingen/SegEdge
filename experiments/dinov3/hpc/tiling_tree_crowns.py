#!/usr/bin/env python3
"""
HPC-friendly driver for the DINOv3 tree crown segmentation pipeline.

This script loads a YAML configuration (see ``configs/dinov3``), normalises the
values, and forwards them to ``sege.pipelines.dinov3_tree_crowns``. It keeps the
cluster submission wrapper lightweight while guaranteeing that all core logic
lives inside the shared pipeline module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from sege.pipelines import dinov3_tree_crowns


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return data


def normalise_path(value: Any, base: Path) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str((base / value).resolve() if not Path(value).is_absolute() else Path(value))
    raise TypeError(f"Expected path-like value, got {type(value)!r}: {value}")


def build_argv(cfg: dict[str, Any]) -> list[str]:
    args: list[str] = []

    def push(flag: str, value: Any) -> None:
        if value in (None, "", [], ()):
            return
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])

    push("--input", cfg.get("input"))
    bands = cfg.get("bands")
    if isinstance(bands, Iterable) and not isinstance(bands, (str, bytes)):
        bands = ",".join(str(b) for b in bands)
    push("--bands", bands)
    push("--mode", cfg.get("mode"))
    push("--head", cfg.get("head"))
    push("--max-edge", cfg.get("max_edge"))
    push("--patch", cfg.get("patch"))
    push("--stride", cfg.get("stride"))
    push("--min-area", cfg.get("min_area"))
    push("--max-area", cfg.get("max_area"))
    push("--min-distance", cfg.get("min_distance"))
    push("--device", cfg.get("device"))
    push("--log-file", cfg.get("log_file"))
    push("--out_dir", cfg.get("out_dir"))
    return args


def merge_config(config_path: Path, overrides: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = config_path.parent.parent.parent.resolve()

    for attr in ["input", "head", "log_file", "out_dir"]:
        value = getattr(overrides, attr, None)
        if value is not None:
            cfg[attr] = normalise_path(value, root)
        elif attr in cfg:
            cfg[attr] = normalise_path(cfg[attr], root)

    for attr in ["bands", "mode", "device"]:
        value = getattr(overrides, attr, None)
        if value is not None:
            cfg[attr] = value

    for attr in ["max_edge", "patch", "stride", "min_area", "max_area", "min_distance"]:
        value = getattr(overrides, attr, None)
        if value is not None:
            cfg[attr] = value

    bands_override = getattr(overrides, "bands", None)
    if isinstance(bands_override, str):
        cfg["bands"] = [b.strip() for b in bands_override.split(",") if b.strip()]

    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dinov3/default.yaml"),
        help="YAML configuration describing the pipeline invocation.",
    )
    parser.add_argument("--input", type=Path, help="Override the GeoTIFF to process.")
    parser.add_argument("--head", type=Path, help="Optional linear probe checkpoint.")
    parser.add_argument("--log-file", type=Path, help="Override log output path.")
    parser.add_argument("--out_dir", type=Path, help="Override artifact output directory.")
    parser.add_argument("--bands", type=str, help="Comma-separated band indices.")
    parser.add_argument("--mode", choices=("unsup", "linear"), help="Pipeline mode override.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Device override.")
    parser.add_argument("--max-edge", type=int, help="Maximum edge length before downscaling.")
    parser.add_argument("--patch", type=int, help="Patch size for dense feature extraction.")
    parser.add_argument("--stride", type=int, help="Stride between patches.")
    parser.add_argument("--min-area", type=int, help="Minimum tree crown area.")
    parser.add_argument("--max-area", type=int, help="Maximum tree crown area.")
    parser.add_argument("--min-distance", type=int, help="Watershed seed separation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = merge_config(args.config, args)
    cli_args = build_argv(cfg)
    dinov3_tree_crowns.main(cli_args)


if __name__ == "__main__":
    main(sys.argv[1:])

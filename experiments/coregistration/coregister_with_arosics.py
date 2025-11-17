#!/usr/bin/env python3
"""
Coregister two GeoTIFFs with AROSICS and report the applied alignment matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from arosics import COREG, COREG_LOCAL


def translation_matrix(dx: float, dy: float) -> List[List[float]]:
    """Return a 3x3 homogeneous translation matrix for dx/dy."""
    return [
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0],
    ]


def main() -> None:


    workdir = Path(__file__).resolve().parent
    ref_img = workdir / "20240724_noon_orthomosaic_rgb.tif"
    tgt_img = workdir / "20240724_noon_orthomosaic_tir.tif"

    if not ref_img.exists() or not tgt_img.exists():
        raise SystemExit(
            f"Expected both {ref_img.name} and {tgt_img.name} to exist in {workdir}"
        )

    output_img = workdir / "20240724_noon_orthomosaic_tir_coreg.tif"

    # Allow larger displacements during matching to avoid false failures.
    coreg = COREG_LOCAL(
        str(ref_img),
        str(tgt_img),
        path_out=str(output_img),
        fmt_out="GTIFF",
        max_shift=40,
        grid_res=50,
        q=False,
        progress=True,
    )

    coreg.correct_shifts()

    # Cool infos
    info: Dict = coreg.coreg_info
    px_dx = float(info["corrected_shifts_px"]["x"])
    px_dy = float(info["corrected_shifts_px"]["y"])
    map_dx = float(info["corrected_shifts_map"]["x"])
    map_dy = float(info["corrected_shifts_map"]["y"])

    matrices = {
        "pixel_shift_matrix": translation_matrix(px_dx, px_dy),
        "map_shift_matrix": translation_matrix(map_dx, map_dy),
    }

    #LOG
    summary = {
        "reference_image": str(ref_img.name),
        "target_image": str(tgt_img.name),
        "coregistered_image": str(output_img.name),
        "pixel_shift": {"dx": px_dx, "dy": px_dy},
        "map_shift_degrees": {"dx": map_dx, "dy": map_dy},
        "affine_matrices": matrices,
        "success": bool(info.get("success", True)),
    }

    summary_path = workdir / "coregistration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Coregistration complete.")
    print(f"  Reference image : {ref_img.name}")
    print(f"  Target image    : {tgt_img.name}")
    print(f"  Output image    : {output_img.name}")
    print(f"  Pixel shift     : dx={px_dx:.4f} px, dy={px_dy:.4f} px (image  coordinates)"  )
    print(f"  Map shift       : dx={map_dx:.8f}°, dy={map_dy:.8f}° (geographic coordinates)"
    )
    print("  Translation matrices:")
    for name, matrix in matrices.items():
        print(f"    {name}:")
        for row in matrix:
            print(f"      {row}")
    print(f"  Summary saved to {summary_path.name}")


if __name__ == "__main__":
    main()

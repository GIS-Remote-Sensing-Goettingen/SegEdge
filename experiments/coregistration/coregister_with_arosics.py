#!/usr/bin/env python3
"""
Coregister two GeoTIFFs with AROSICS and report the applied alignment matrix.
"""

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


    ref_img = "20240724_evening_orthomosaic_tir.tif"
    tgt_img =  "20240724_noon_orthomosaic_tir.tif"

    output_img = "20240724_noon_orthomosaic_tir_coreg.tif"

    # Allow larger displacements during matching to avoid false failures.
    coreg = COREG_LOCAL(
        str(ref_img),
        str(tgt_img),
        path_out=str(output_img),
        fmt_out="GTIFF",
        max_shift=40,
        max_points=700,
        grid_res=200,
        q=False,
        progress=True,
    )

    coreg.correct_shifts()

    # Cool infos
    info: Dict = coreg.coreg_info
    print(f"Success: {info['success']}")
    print(f"Mean horizontal shift: {info['mean_shifts_px']['x']:.2f} pixels")
    print(f"Mean vertical shift: {info['mean_shifts_px']['y']:.2f} pixels")
    print(f"Total GCPs: {len(info['GCPList'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Coregister two GeoTIFFs with AROSICS and report the applied alignment matrix.
"""

import json
from pathlib import Path
from typing import Dict

from arosics import COREG, COREG_LOCAL

def main() -> None:


    ref_img = "20240726_noon_orthomosaic.tif"
    tgt_img =  "20240726_evening_orthomosaic.tif"

    output_img = "20240726_evening_orthomosaic_coreg.tif"

    coreg = COREG_LOCAL(
        str(ref_img),
        str(tgt_img),
        path_out=str(output_img),
        fmt_out="GTIFF",
        max_shift=100,
        max_points=2000,
        grid_res=100,
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

"""
Reusable inference pipelines for the SegEdge project.

Available modules
-----------------

``dinov3_tree_crowns``
    DINOv3-based unsupervised tree crown delineation for high-resolution GeoTIFFs.

``sam2_farmland``
    SAM 2 based prompt-driven farmland segmentation workflow for multispectral imagery.
"""

from . import dinov3_tree_crowns, sam2_farmland  # noqa: F401

__all__ = ["dinov3_tree_crowns", "sam2_farmland"]


"""
SegEdge core package.

This namespace exposes reusable segmentation pipelines and shared utilities.
Individual command-line entry points live under ``sege.pipelines`` or
``sege.cli``. Import the desired pipeline module directly, e.g.
``from sege.pipelines.dinov3_tree_crowns import main``.
"""

__all__ = ["pipelines"]


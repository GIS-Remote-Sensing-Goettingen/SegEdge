"""
Factories for segmentation heads.

This keeps the main script simple: pick a head by string name and instantiate
it with consistent defaults.
"""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from .base import SegmentationHead
from .maskformer import DinoMaskFormerHead
from .unet import DinoUNetHead
from .unet_v2 import DinoUNetV2Head
from .UnetLite import DinoUNetLiteHead

HeadBuilder = Callable[[int, int], SegmentationHead]


def available_heads() -> Dict[str, HeadBuilder]:
    """
    Return the set of supported segmentation head builders.

    >>> sorted(available_heads().keys())
    ['maskformer', 'unet', 'unet_lite', 'unet_v2']
    """

    return {
        "unet": DinoUNetHead,
        "unet_v2": DinoUNetV2Head,
        "maskformer": DinoMaskFormerHead,
        "unet_lite": DinoUNetLiteHead,
    }


def build_head(name: str, num_classes: int, dino_channels: int) -> SegmentationHead:
    """
    Build a segmentation head by name.

    >>> head = build_head("unet", num_classes=2, dino_channels=1024)
    >>> isinstance(head, nn.Module)
    True
    """

    registry = available_heads()
    if name not in registry:
        raise ValueError(f"Unknown head '{name}'. Choose from: {sorted(registry)}")
    return registry[name](num_classes=num_classes, dino_channels=dino_channels)

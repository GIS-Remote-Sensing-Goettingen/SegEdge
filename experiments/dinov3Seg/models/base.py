"""
Base interfaces for segmentation heads.

The Linux kernel style emphasizes clear contracts, so this module keeps the
abstract interface tiny and well documented.
"""

from __future__ import annotations

import torch
from torch import nn


class SegmentationHead(nn.Module):
    """
    Base segmentation head that consumes raw RGB tensors and multiscale DINO
    features, then returns logits.

    >>> class DummyHead(SegmentationHead):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(2, 2, 1)
    ...     def forward(self, image, features):
    ...         _ = image
    ...         return self.conv(features[0])
    >>> image = torch.randn(1, 3, 4, 4)
    >>> feats = [torch.randn(1, 2, 4, 4)]
    >>> head = DummyHead()
    >>> out = head(image, feats)
    >>> tuple(out.shape)
    (1, 2, 4, 4)
    """

    def forward(
        self,
        image: torch.Tensor,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Subclasses must override this call to map fused inputs to logits.

        >>> class EchoHead(SegmentationHead):
        ...     def forward(self, image, features):
        ...         return image.sum(dim=1, keepdim=True) + features[0]
        >>> img = torch.ones(1, 3, 2, 2)
        >>> feats = [torch.zeros(1, 1, 2, 2)]
        >>> EchoHead()(img, feats).shape
        torch.Size([1, 1, 2, 2])
        """

        raise NotImplementedError("SegmentationHead subclasses must implement forward")

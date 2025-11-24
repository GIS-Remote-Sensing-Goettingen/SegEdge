"""
DinoUNet variant with Spatial Prior Module and Fidelity-Aware projections.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .base import SegmentationHead


class SpatialPriorModule(nn.Module):
    """
    Lightweight CNN that extracts edge-aware priors from RGB inputs.

    >>> spm = SpatialPriorModule(in_channels=3, base_channels=8)
    >>> img = torch.randn(1, 3, 32, 32)
    >>> h2, h4 = spm(img)
    >>> tuple(h2.shape), tuple(h4.shape)
    ((1, 8, 16, 16), (1, 16, 8, 8))
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        """
        Initialize the SPM stack.

        >>> SpatialPriorModule(3, 4)
        SpatialPriorModule(
          (stem): Sequential(
            (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU(inplace=True)
          )
          (layer2): Sequential(
            (0): Conv2d(4, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU(inplace=True)
          )
        )
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce feature maps at H/2 and H/4 resolution.

        >>> spm = SpatialPriorModule(3, 4)
        >>> img = torch.randn(1, 3, 64, 64)
        >>> out = spm(img)
        >>> len(out)
        2
        """

        c1 = self.stem(x)
        c2 = self.layer2(c1)
        return c1, c2


class FidelityAwareProjection(nn.Module):
    """
    Compresses the 1024-channel DINO maps into smaller tensors with channel
    attention, keeping the semantics intact.

    >>> proj = FidelityAwareProjection(16, 4)
    >>> feat = torch.randn(1, 16, 8, 8)
    >>> tuple(proj(feat).shape)
    (1, 4, 8, 8)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Build the projector and attention path.

        >>> FidelityAwareProjection(8, 4)
        FidelityAwareProjection(
          (proj): Conv2d(8, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=4, out_features=1, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=1, out_features=4, bias=False)
            (3): Sigmoid()
          )
        )
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(out_channels // 4, 1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply projection and attention weighting.

        >>> proj = FidelityAwareProjection(4, 4)
        >>> feat = torch.randn(1, 4, 4, 4)
        >>> proj(feat).shape
        torch.Size([1, 4, 4, 4])
        """

        x_proj = self.bn(self.proj(x))
        b, c, _, _ = x_proj.size()
        weights = self.fc(self.avg_pool(x_proj).view(b, c)).view(b, c, 1, 1)
        return x_proj * weights


class DoubleConv(nn.Module):
    """
    Helper block mirroring ConvBlock but without reuse to keep shapes explicit.

    >>> block = DoubleConv(2, 2)
    >>> x = torch.randn(1, 2, 8, 8)
    >>> block(x).shape
    torch.Size([1, 2, 8, 8])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Build the double convolution.

        >>> DoubleConv(3, 3)
        DoubleConv(
          (conv): Sequential(
            (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU(inplace=True)
          )
        )
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the block over inputs.

        >>> block = DoubleConv(2, 4)
        >>> x = torch.randn(1, 2, 4, 4)
        >>> block(x).shape
        torch.Size([1, 4, 4, 4])
        """

        return self.conv(x)


class DinoUNetV2Head(SegmentationHead):
    """
    Enhanced DinoUNet with SPM and FAPM, including deep supervision.

    >>> head = DinoUNetV2Head(num_classes=2, dino_channels=16)
    >>> img = torch.randn(1, 3, 256, 256)
    >>> feats = [
    ...     torch.randn(1, 16, 32, 32),
    ...     torch.randn(1, 16, 16, 16),
    ...     torch.randn(1, 16, 8, 8),
    ...     torch.randn(1, 16, 4, 4),
    ... ]
    >>> logits, deep = head.forward_with_aux(img, feats)
    >>> tuple(logits.shape), tuple(deep.shape)
    ((1, 2, 256, 256), (1, 2, 32, 32))
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """
        Build the V2 head.

        >>> DinoUNetV2Head(2, 32)  # doctest: +ELLIPSIS
        DinoUNetV2Head(
          (spm): SpatialPriorModule...
        )
        """

        super().__init__()
        self.spm = SpatialPriorModule(in_channels=3, base_channels=32)
        self.fapm1 = FidelityAwareProjection(dino_channels, 512)
        self.fapm2 = FidelityAwareProjection(dino_channels, 256)
        self.fapm3 = FidelityAwareProjection(dino_channels, 128)
        self.fapm4 = FidelityAwareProjection(dino_channels, 64)
        self.bottleneck = DoubleConv(512, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(128 + 128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(64 + 64, 64)
        self.ds_head1 = nn.Conv2d(64, num_classes, 1)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up4_extra = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = DoubleConv(32 + 32, 32)
        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning main logits and deep supervision head.

        >>> head = DinoUNetV2Head(2, 64)
        >>> img = torch.randn(1, 3, 256, 256)
        >>> feats = [
        ...     torch.randn(1, 64, 32, 32),
        ...     torch.randn(1, 64, 16, 16),
        ...     torch.randn(1, 64, 8, 8),
        ...     torch.randn(1, 64, 4, 4),
        ... ]
        >>> main, aux = head.forward_with_aux(img, feats)
        >>> tuple(main.shape[2:]) == (256, 256)
        True
        """

        spm_h2, spm_h4 = self.spm(image)
        d_shallow = self.fapm4(features[0])
        d_mid1 = self.fapm3(features[1])
        d_mid2 = self.fapm2(features[2])
        d_deep = self.fapm1(features[3])
        x = self.bottleneck(d_deep)
        x = self.conv1(self._concat(self.up1(x), d_mid2))
        x = self.conv2(self._concat(self.up2(x), d_mid1))
        x = self.conv3(self._concat(self.up3(x), d_shallow))
        ds_out = self.ds_head1(x)
        x = self.up4(x)
        if x.shape[-1] < spm_h4.shape[-1]:
            x = self.up4_extra(x)
        x = self.conv4(self._concat(x, spm_h4))
        x = self.conv5(self._concat(self.up5(x), spm_h2))
        logits = self.final_conv(self.final_up(x))
        return logits, ds_out

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward returning only the main logits to respect the base interface.

        >>> head = DinoUNetV2Head(2, 64)
        >>> img = torch.randn(1, 3, 256, 256)
        >>> feats = [
        ...     torch.randn(1, 64, 32, 32),
        ...     torch.randn(1, 64, 16, 16),
        ...     torch.randn(1, 64, 8, 8),
        ...     torch.randn(1, 64, 4, 4),
        ... ]
        >>> head(img, feats).shape
        torch.Size([1, 2, 256, 256])
        """

        logits, _ = self.forward_with_aux(image, features)
        return logits

    def _concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Align spatial dimensions between tensors before concatenation.
        """

        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

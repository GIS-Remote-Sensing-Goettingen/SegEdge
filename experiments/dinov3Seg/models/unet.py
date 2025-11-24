"""
Classic U-Net style decoder that consumes frozen DINO features.

The forward path mirrors the baseline experiment provided in the original
scripts.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .base import SegmentationHead


class ConvBlock(nn.Module):
    """
    Two Conv-BN-ReLU stages, matching the Linux kernel ethos of simplicity.

    >>> block = ConvBlock(4, 8)
    >>> x = torch.randn(1, 4, 16, 16)
    >>> tuple(block(x).shape)
    (1, 8, 16, 16)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Create the convolutional block.

        >>> ConvBlock(3, 3)
        ConvBlock(
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional sub-network.

        >>> block = ConvBlock(2, 2)
        >>> t = torch.ones(1, 2, 4, 4)
        >>> block(t).shape
        torch.Size([1, 2, 4, 4])
        """

        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block that mirrors U-Net semantics.

    >>> up = UpBlock(8, 4, 4)
    >>> x = torch.randn(1, 8, 8, 8)
    >>> skip = torch.randn(1, 4, 16, 16)
    >>> tuple(up(x, skip).shape)
    (1, 4, 16, 16)
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Construct an upsampling block.

        >>> UpBlock(4, 2, 2)
        UpBlock(
          (up): ConvTranspose2d(4, 2, kernel_size=(2, 2), stride=(2, 2))
          (conv): ConvBlock(
            (conv): Sequential(
              (0): Conv2d(4, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
              (3): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (4): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): ReLU(inplace=True)
            )
          )
        )
        """

        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Upsample, align, concatenate, and refine the feature map.

        >>> up = UpBlock(4, 2, 2)
        >>> x = torch.randn(1, 4, 4, 4)
        >>> skip = torch.randn(1, 2, 8, 8)
        >>> up(x, skip).shape
        torch.Size([1, 2, 8, 8])
        """

        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DinoUNetHead(SegmentationHead):
    """
    Original DinoUNet configuration adapted into the modular head format.

    >>> head = DinoUNetHead(num_classes=2, dino_channels=1024)
    >>> img = torch.randn(1, 3, 64, 64)
    >>> feats = [torch.randn(1, 1024, 4, 4) for _ in range(4)]
    >>> out = head(img, feats)
    >>> tuple(out.shape)
    (1, 2, 64, 64)
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """
        Build the DinoUNet head.

        >>> head = DinoUNetHead(2, 512)
        >>> isinstance(head.final_conv, nn.Conv2d)
        True
        """

        super().__init__()
        decoder_channels = [512, 256, 128, 64]
        self.bottleneck = ConvBlock(dino_channels, decoder_channels[0])
        skip_connection_channels: List[int] = [dino_channels] * 3 + [3]
        self.up_blocks = nn.ModuleList()
        in_channels = decoder_channels[0]
        for idx, skip_channels in enumerate(skip_connection_channels):
            out_channels = (
                decoder_channels[idx + 1]
                if idx + 1 < len(decoder_channels)
                else decoder_channels[-1]
            )
            self.up_blocks.append(UpBlock(in_channels, skip_channels, out_channels))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse the DINO features and RGB image to obtain logits.

        >>> head = DinoUNetHead(2, 256)
        >>> img = torch.randn(1, 3, 32, 32)
        >>> feats = [torch.randn(1, 256, 2, 2) for _ in range(4)]
        >>> head(img, feats).shape
        torch.Size([1, 2, 32, 32])
        """

        features_reversed = features[::-1]
        x = self.bottleneck(features_reversed[0])
        skip_connections = [features_reversed[1], features_reversed[2], features_reversed[3], image]
        for idx, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[idx])
        return self.final_conv(x)

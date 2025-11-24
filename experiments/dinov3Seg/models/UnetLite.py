"""
Lite DinoUNet head: smaller and cheaper than DinoUNetV2Head.

Design goals:
- Keep the same interface and spatial structure as DinoUNetV2Head
- Reduce channel widths in SPM, FAP, and decoder to save memory/FLOPs
- Preserve high-resolution decoding for thin linear woody features
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .base import SegmentationHead
from .unet_v2 import SpatialPriorModule, FidelityAwareProjection, DoubleConv


# We assume SpatialPriorModule, FidelityAwareProjection, DoubleConv are defined
# in the same module as in your snippet. If they live elsewhere, adjust imports.


class DinoUNetLiteHead(SegmentationHead):
    """
    Lighter DinoUNet-style head with SPM + FAP, but reduced channels.

    This is intended as a drop-in replacement for DinoUNetV2Head when
    GPU memory or throughput are limiting factors.

    >>> head = DinoUNetLiteHead(num_classes=2, dino_channels=64)
    >>> img = torch.randn(1, 3, 256, 256)
    >>> feats = [
    ...     torch.randn(1, 64, 32, 32),
    ...     torch.randn(1, 64, 16, 16),
    ...     torch.randn(1, 64, 8, 8),
    ...     torch.randn(1, 64, 4, 4),
    ... ]
    >>> logits, aux = head.forward_with_aux(img, feats)
    >>> tuple(logits.shape)
    (1, 2, 256, 256)
    >>> tuple(aux.shape)
    (1, 2, 32, 32)
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """
        Build the lightweight head.

        Architectural changes vs. DinoUNetV2Head:
          - SPM base_channels = 16 (was 32) to shrink RGB prior path.
          - FAP outputs: [128, 64, 32, 16] (was [512, 256, 128, 64]).
          - Decoder channels follow FAP widths, roughly halved / quartered.
          - Deep supervision kept, but on a 16-channel feature map.
        """
        super().__init__()

        # --- 1) Spatial prior path (on RGB) ---------------------------------
        # Smaller base width = fewer parameters and cheaper high-res features.
        # Still provides H/2 and H/4 priors for boundaries.
        self.spm = SpatialPriorModule(in_channels=3, base_channels=16)

        # --- 2) Fidelity-aware projections from DINO backbone ----------------
        # We compress DINO features aggressively but keep attention to retain
        # the most informative channels for segmentation.
        self.fapm1 = FidelityAwareProjection(dino_channels, 128)  # deepest
        self.fapm2 = FidelityAwareProjection(dino_channels, 64)
        self.fapm3 = FidelityAwareProjection(dino_channels, 32)
        self.fapm4 = FidelityAwareProjection(dino_channels, 16)   # shallowest

        # --- 3) Bottleneck at the deepest resolution (e.g. 4x4) -------------
        # Reduced to 128 channels to cut cost while still allowing some
        # non-linear mixing of deep DINO semantics.
        self.bottleneck = DoubleConv(128, 128)

        # --- 4) Decoder: upsample + skip connection blocks -------------------
        # Note: channel counts must align with concatenation:
        #   up1: 128 -> 64, concat with 64 -> DoubleConv(128 -> 64)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(64 + 64, 64)

        #   up2: 64 -> 32, concat with 32 -> DoubleConv(64 -> 32)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(32 + 32, 32)

        #   up3: 32 -> 16, concat with 16 -> DoubleConv(32 -> 16)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv3 = DoubleConv(16 + 16, 16)

        # Deep supervision head at 32x32 on a compact 16-channel feature map.
        self.ds_head1 = nn.Conv2d(16, num_classes, kernel_size=1)

        # --- 5) Fuse decoder with SPM priors at H/4 and H/2 ------------------
        # After up3, feature map is 32x32; up4 -> 64x64 (H/4).
        # SPM H/4 has 32 channels (base_channels * 2).
        self.up4 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.up4_extra = nn.ConvTranspose2d(16, 16, 2, stride=2)
        # Input to conv4: decoder(16) + spm_h4(32) -> 48
        self.conv4 = DoubleConv(16 + 32, 16)

        # Next we go from H/4 to H/2 and fuse with SPM H/2 (16 channels).
        # up5: 16 -> 16, concat with 16 -> DoubleConv(32 -> 16).
        self.up5 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.conv5 = DoubleConv(16 + 16, 16)

        # --- 6) Final upsample to full resolution and classification ---------
        # final_up: H/2 -> H, while staying at 16 channels.
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    # ---------------------------------------------------------------------    # Forward variants
    # ---------------------------------------------------------------------
    def forward_with_aux(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning main logits and deep supervision head.

        Args:
            image:    (B, 3, H, W)
            features: list of 4 DINO feature maps, ordered from shallowest
                      to deepest, as in DinoUNetV2Head:
                        features[0]: (B, C, H/8,  W/8)
                        features[1]: (B, C, H/16, W/16)
                        features[2]: (B, C, H/32, W/32)
                        features[3]: (B, C, H/64, W/64)

        Returns:
            logits: (B, num_classes, H, W)
            ds1:    (B, num_classes, H/8, W/8) deep supervision prediction
        """
        # --- SPM priors (RGB-based spatial context) -------------------------
        spm_h2, spm_h4 = self.spm(image)  # H/2, H/4

        # --- Project backbone features with fidelity-aware attention --------
        d_shallow = self.fapm4(features[0])  # H/8,  16ch
        d_mid1    = self.fapm3(features[1])  # H/16, 32ch
        d_mid2    = self.fapm2(features[2])  # H/32, 64ch
        d_deep    = self.fapm1(features[3])  # H/64, 128ch

        # --- Bottleneck at the deepest scale --------------------------------
        x = self.bottleneck(d_deep)  # (B, 128, H/64, W/64)

        # --- Decoder: merge deep DINO semantics -----------------------------
        # up1: H/64 -> H/32, then fuse with mid2 (H/32).
        x = self.up1(x)
        x = self.conv1(self._concat(x, d_mid2))

        # up2: H/32 -> H/16, fuse with mid1 (H/16).
        x = self.up2(x)
        x = self.conv2(self._concat(x, d_mid1))

        # up3: H/16 -> H/8, fuse with shallow DINO features (H/8).
        x = self.up3(x)
        x = self.conv3(self._concat(x, d_shallow))

        # Deep supervision prediction at H/8 (e.g. 32x32 for H=256).
        ds_out = self.ds_head1(x)

        # --- Fuse with SPM priors at higher resolutions ---------------------
        # up4: H/8 -> H/4; account for possible rounding mismatches.
        x = self.up4(x)
        if x.shape[-1] < spm_h4.shape[-1]:
            # In case of odd input sizes, we might be one pixel short.
            x = self.up4_extra(x)

        # Fuse decoder with SPM H/4 (boundary-aware prior).
        x = self.conv4(self._concat(x, spm_h4))

        # up5: H/4 -> H/2, fuse with SPM H/2 (high-res spatial prior).
        x = self.up5(x)
        x = self.conv5(self._concat(x, spm_h2))

        # Final upsample: H/2 -> H, then 1x1 conv to logits.
        x = self.final_up(x)
        logits = self.final_conv(x)

        return logits, ds_out

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward returning only the main logits to respect the base interface.
        """
        logits, _ = self.forward_with_aux(image, features)
        return logits

    # ------------------------------------------------------------------    # Helper
    # ------------------------------------------------------------------
    def _concat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Align spatial dimensions between tensors before concatenation.

        This keeps robustness to small rounding differences arising from
        strided convolutions or odd-sized inputs.
        """
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

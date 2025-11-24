"""
MaskFormer-style transformer head sitting on top of DINOv3 features.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .base import SegmentationHead


class SpatialPriorModule(nn.Module):
    """
    Downsamples the RGB input to H/4 to complement DINO features.

    >>> spm = SpatialPriorModule(in_channels=3, dim=32)
    >>> img = torch.randn(1, 3, 32, 32)
    >>> tuple(spm(img).shape)
    (1, 32, 8, 8)
    """

    def __init__(self, in_channels: int = 3, dim: int = 128) -> None:
        """
        Build the module with two stride-2 convolutions.

        >>> SpatialPriorModule(3, 16)
        SpatialPriorModule(
          (stem): Sequential(
            (0): Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
          )
        )
        """

        super().__init__()
        half_dim = dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, half_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_dim),
            nn.ReLU(),
            nn.Conv2d(half_dim, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce the H/4 prior tensor.

        >>> spm = SpatialPriorModule(3, 32)
        >>> spm(torch.randn(1, 3, 32, 32)).shape[-2:]
        torch.Size([8, 8])
        """

        return self.stem(x)


class PixelDecoder(nn.Module):
    """
    Projects multiscale DINO features and fuses them with SPM output.

    >>> decoder = PixelDecoder(dino_dim=8, embed_dim=32)
    >>> img = torch.randn(1, 3, 32, 32)
    >>> feats = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    >>> tuple(decoder(img, feats).shape)
    (1, 32, 8, 8)
    """

    def __init__(self, dino_dim: int = 1024, embed_dim: int = 256) -> None:
        """
        Build projector layers plus the SPM module.

        >>> PixelDecoder(4, 32)  # doctest: +ELLIPSIS
        PixelDecoder(
          (proj_l23): Conv2d(4, 32, kernel_size=(1, 1), stride=(1, 1))
          ...
        )
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.proj_l23 = nn.Conv2d(dino_dim, embed_dim, 1)
        self.proj_l17 = nn.Conv2d(dino_dim, embed_dim, 1)
        self.proj_l11 = nn.Conv2d(dino_dim, embed_dim, 1)
        self.proj_l5 = nn.Conv2d(dino_dim, embed_dim, 1)
        self.spm = SpatialPriorModule(in_channels=3, dim=embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1)
        groups = min(32, embed_dim)
        while embed_dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, embed_dim)

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Return fused pixel embeddings at H/4.

        >>> decoder = PixelDecoder(16, 32)
        >>> img = torch.randn(1, 3, 64, 64)
        >>> feats = [torch.randn(1, 16, 4, 4) for _ in range(4)]
        >>> decoder(img, feats).shape[1]
        32
        """

        feat_spm = self.spm(image)
        target_size = feat_spm.shape[-2:]
        proj_feats = []
        for proj, feat in zip(
            [self.proj_l5, self.proj_l11, self.proj_l17, self.proj_l23], features
        ):
            proj_feats.append(F.interpolate(proj(feat), size=target_size, mode="bilinear"))
        feat_dino = sum(proj_feats)
        fused = torch.cat([feat_spm, feat_dino], dim=1)
        return self.norm(self.fusion(fused))


class MaskTransformerHead(nn.Module):
    """
    Transformer decoder that produces per-class mask logits.

    >>> head = MaskTransformerHead(num_classes=2, embed_dim=8, num_heads=2, num_layers=1)
    >>> pixel_emb = torch.randn(1, 8, 8, 8)
    >>> out = head(pixel_emb)
    >>> out.shape
    torch.Size([1, 2, 8, 8])
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        """
        Build the transformer decoder stack.

        >>> MaskTransformerHead(2, 16, 2, 1)  # doctest: +ELLIPSIS
        MaskTransformerHead(
          (class_queries): Embedding(2, 16)
          ...
        )
        """

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_queries = nn.Embedding(num_classes, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, pixel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Produce logits via query-to-pixel dot products.

        >>> head = MaskTransformerHead(2, 8, 2, 1)
        >>> emb = torch.randn(1, 8, 8, 8)
        >>> head(emb).shape
        torch.Size([1, 2, 8, 8])
        """

        b, c, h, w = pixel_embeddings.shape
        memory = pixel_embeddings.flatten(2).permute(0, 2, 1)
        queries = self.class_queries.weight.unsqueeze(0).expand(b, -1, -1)
        refined = self.transformer_decoder(tgt=queries, memory=memory)
        refined = self.output_mlp(refined)
        mask_logits = torch.bmm(refined, memory.transpose(1, 2)).view(b, self.num_classes, h, w)
        return mask_logits


class DinoMaskFormerHead(SegmentationHead):
    """
    Combines PixelDecoder and MaskTransformer head with an upsampling step.

    >>> head = DinoMaskFormerHead(num_classes=2, dino_channels=16)
    >>> img = torch.randn(1, 3, 64, 64)
    >>> feats = [torch.randn(1, 16, 4, 4) for _ in range(4)]
    >>> head(img, feats).shape
    torch.Size([1, 2, 64, 64])
    """

    def __init__(self, num_classes: int, dino_channels: int) -> None:
        """
        Build the pix2mask pipeline.

        >>> DinoMaskFormerHead(2, 32)  # doctest: +ELLIPSIS
        DinoMaskFormerHead(
          (pixel_decoder): PixelDecoder...
        )
        """

        super().__init__()
        embed_dim = 256
        self.pixel_decoder = PixelDecoder(dino_dim=dino_channels, embed_dim=embed_dim)
        self.mask_head = MaskTransformerHead(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Produce segmentation logits at the original image resolution.

        >>> head = DinoMaskFormerHead(2, 16)
        >>> img = torch.randn(1, 3, 32, 32)
        >>> feats = [torch.randn(1, 16, 4, 4) for _ in range(4)]
        >>> head(img, feats).shape
        torch.Size([1, 2, 32, 32])
        """

        pixel_emb = self.pixel_decoder(image, features)
        logits = self.mask_head(pixel_emb)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

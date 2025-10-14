"""Reconstruction of the proprietary S2x10NetS decoder from dumped weights.

The vendor binary instantiates a ViT backbone followed by a heavy RRDB-style
super-resolution head that upsamples the multispectral tokens by 10×.  During
runtime we intercepted the module tree and dumped its `state_dict` from
`module_098_s2dr3_s2dr3infer.S2x10NetS.pt`.  This file recreates the matching
PyTorch modules so we can load and use the decoder offline.

The structure, deduced from parameter shapes and forward traces, is:

    - Stem conv (10 ➔ 160 channels) with LeakyReLU.
    - 23 RRDB blocks, each composed of 3 ResidualDenseBlocks (growth 80).
    - Five `Resample` stages that upsample by 2× using bilinear resize + conv
      (channel schedule: 160➔80➔40➔20➔10➔10).

All convolution weights are 3×3 with padding=1; activations are LeakyReLU with
the default 0.01 slope, matching the runtime trace we captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.01, inplace=True)


class StemConv(nn.Module):
    """Initial 3×3 conv block (records weights under `encoder.0.conv`)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _leaky_relu(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """5-layer dense block with growth rate 80 and residual scaling."""

    def __init__(self, channels: int, growth: int, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth * 2, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth * 3, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth * 4, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = _leaky_relu(self.conv1(x))
        x2 = _leaky_relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = _leaky_relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = _leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        out = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.res_scale * out


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3 nested RDBs)."""

    def __init__(self, channels: int, growth: int, res_scale: float = 0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth, res_scale)
        self.rdb2 = ResidualDenseBlock(channels, growth, res_scale)
        self.rdb3 = ResidualDenseBlock(channels, growth, res_scale)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.res_scale * (out - x)


class Resample(nn.Module):
    """Upsample-by-2 block with 3×3 conv (weights saved as `generator.*`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.upscale = upscale
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upscale:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return _leaky_relu(self.conv(x))


@dataclass
class S2x10Config:
    in_channels: int = 10
    embed_channels: int = 160
    growth: int = 80
    rrdb_blocks: int = 23


class S2x10NetS(nn.Module):
    """Faithful reproduction of `s2dr3.s2dr3infer.S2x10NetS`."""

    def __init__(self, cfg: S2x10Config = S2x10Config()):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleList()
        self.encoder.append(StemConv(cfg.in_channels, cfg.embed_channels))
        trunk = nn.ModuleList(
            [RRDB(cfg.embed_channels, cfg.growth) for _ in range(cfg.rrdb_blocks)]
        )
        self.encoder.append(trunk)

        self.generator = nn.ModuleList(
            [
                Resample(cfg.embed_channels, 80),
                Resample(80, 40),
                Resample(40, 20),
                Resample(20, 10),
                Resample(10, 10, upscale=False, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder[0](x)
        for block in self.encoder[1]:
            x = block(x)
        for block in self.generator:
            x = block(x)
        return x


def load_s2x10_from_state(path: Path | str) -> S2x10NetS:
    """Load the reconstructed decoder from a monolithic state_dict file."""

    path = Path(path)
    state = torch.load(path, map_location="cpu")
    model = S2x10NetS()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in decoder state: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing keys when loading decoder: {missing}")
    model.eval()
    return model


def load_s2x10_from_chunks(root: Path | str) -> S2x10NetS:
    """Load the decoder by streaming per-module dumps (reduces peak RAM usage).

    Parameters are expected to be stored under ``root`` using the naming scheme
    emitted by the shim hooks, e.g. ``module_004_s2dr3_s2dr3infer.RRDB.pt``.
    Only the RRDB and Resample files are required; standalone
    ``ResidualDenseBlock`` dumps are ignored.
    """

    root = Path(root)
    model = S2x10NetS()

    # Stem conv (10 -> 160)
    stem_path = root / "module_000_s2dr3_s2dr3infer.Resample.pt"
    if not stem_path.exists():
        raise FileNotFoundError(f"Stem weights not found: {stem_path}")
    stem_state = torch.load(stem_path, map_location="cpu")
    model.encoder[0].load_state_dict(stem_state, strict=False)

    # 23 RRDB blocks
    rrdb_paths = sorted(
        root.glob("module_*_s2dr3_s2dr3infer.RRDB.pt"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if len(rrdb_paths) != len(model.encoder[1]):
        raise RuntimeError(
            f"Expected {len(model.encoder[1])} RRDB dumps, found {len(rrdb_paths)}"
        )
    for block, path in zip(model.encoder[1], rrdb_paths):
        state = torch.load(path, map_location="cpu")
        block.load_state_dict(state, strict=True)

    # Generator resample stages (skip the stem entry).
    resample_paths = sorted(
        (
            p
            for p in root.glob("module_*_s2dr3_s2dr3infer.Resample.pt")
            if p.name != stem_path.name
        ),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if len(resample_paths) != len(model.generator):
        raise RuntimeError(
            f"Expected {len(model.generator)} generator dumps, found {len(resample_paths)}"
        )
    for module, path in zip(model.generator, resample_paths):
        state = torch.load(path, map_location="cpu")
        module.load_state_dict(state, strict=False)

    model.eval()
    return model


if __name__ == "__main__":
    print(
        "This module is meant to be imported. "
        "Use load_s2x10_from_state(<path>) for the monolithic dump or "
        "load_s2x10_from_chunks(<dir>) to stream per-module weights."
    )

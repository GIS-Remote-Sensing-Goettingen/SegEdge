"""Reverse-engineered Vision Transformer backbone used by S2DR3.

This module recreates the architecture implied by the decrypted checkpoint
(`gedrm_*.safetensors`).  It focuses on the feature extractor; downstream heads
remain vendor-specific and are therefore omitted.

Shapes recovered from the checkpoint:
    - Patch embedding: conv (3 -> 1024) with 16x16 kernel/stride.
    - 24 transformer blocks (layer.0 ... layer.23).
    - Each block uses LayerScale (lambda1/lambda2) and a SwiGLU-style MLP with
      an expansion factor of 4 (4096 units).
    - Attention projections are 1024x1024, consistent with 16 heads.
    - Learnable cls + 4 register tokens + mask token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.nn import functional as F


@dataclass
class VitConfig:
    img_size: int = 1024  # nominal square crop
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    register_tokens: int = 4
    layer_scale_init: float = 1e-5


class PatchEmbed(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=cfg.in_chans,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # N, L, C
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda1 * x


class Attention(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # Original checkpoint omits a bias term on the key projection.
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # B, heads, N, head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(x.shape)
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.up_proj = nn.Linear(cfg.embed_dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, cfg.embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = Attention(cfg)
        self.layer_scale1 = LayerScale(cfg.embed_dim, cfg.layer_scale_init)

        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg)
        self.layer_scale2 = LayerScale(cfg.embed_dim, cfg.layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x))
        x = x + self.layer_scale1(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.layer_scale2(mlp_out)
        return x


class VisionTransformerS2DR3(nn.Module):
    """Backbone recreation matching the S2DR3 checkpoint parameterization."""

    def __init__(self, cfg: Optional[VitConfig] = None):
        super().__init__()
        self.cfg = cfg or VitConfig()
        self.patch_embed = PatchEmbed(self.cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.cfg.register_tokens, self.cfg.embed_dim)
        )
        self.pos_dropout = nn.Identity()  # no absolute position embeddings present
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.depth)])
        self.norm = nn.LayerNorm(self.cfg.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # No learned positional embedding is stored in the checkpoint; rely on implicit scheme.
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
        x = self.pos_dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.forward_tokens(x)
        cls = tokens[:, 0]
        return cls, tokens


def load_backbone_from_safetensors(path: str, device: Optional[torch.device] = None) -> VisionTransformerS2DR3:
    """Instantiate the backbone and load the safetensor weights."""
    raw_state = load_safetensors(path)
    remapped = {}
    for key, tensor in raw_state.items():
        new_key = key
        if key.startswith("embeddings.patch_embeddings."):
            new_key = key.replace("embeddings.patch_embeddings.", "patch_embed.proj.")
        elif key.startswith("embeddings."):
            new_key = key.replace("embeddings.", "")
        elif key.startswith("layer."):
            parts = key.split(".")
            block_idx = parts[1]
            new_key = ".".join(["blocks", block_idx] + parts[2:])
        elif key.startswith("norm."):
            new_key = key
        else:
            # Keep tails like layer_scale.* as-is after block prefix substitution above.
            new_key = key
        remapped[new_key] = tensor

    model = VisionTransformerS2DR3()
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading backbone: {unexpected}")
    if missing:
        print(f"[warn] Missing keys (likely head parameters not part of backbone): {missing}")
    if device is not None:
        model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Lightweight self-check to ensure the reconstructed architecture ingests the safetensor weights.
    ckpt = "/home/mak/.cache/s2dr3/sandbox/local/S2DR3/gedrm_17igu4jxui"
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    model = load_backbone_from_safetensors(ckpt, device=device)
    dummy = torch.randn(1, 3, 1024, 1024, device=device)
    cls, tokens = model(dummy)
    print("CLS shape:", cls.shape, "tokens:", tokens.shape)

# S2DR3 Backbone Recreation

This directory contains a scratch reimplementation of the Vision Transformer
backbone that powers Gamma Earth's S2DR3 model. The goal is to make the vendor
checkpoint usable inside our own experiments while documenting the structure we
have inferred from encrypted blobs recovered during runtime.

## What We Know So Far

- **Model family**: ViT-L/16 with 24 encoder blocks and 1024-dimensional tokens.
- **Tokens**: 1 `cls_token`, 4 `register_tokens`, and 1 `mask_token` are learned.
- **Patch embed**: `Conv2d(3 → 1024, kernel=16, stride=16)`.
- **Attention**: 16 heads, linear projections are 1024×1024. The key projection
  has _no_ bias term (the other projections do), which now matches the
  checkpoint exactly.
- **LayerScale**: Two learned per-block scalars (lambda1/lambda2) initialized to
  1e-5.
- **MLP**: Two-layer 1024 → 4096 → 1024 feed-forward with GELU; no evidence of
  extra gating parameters in the state dict.
- **Norms**: Pre-normalization using `LayerNorm` before attention and MLP, plus
  a final `LayerNorm` head.
- **Positional encoding**: No absolute embedding tensors are present; the model
  appears to rely on implicit or relative position handling downstream.

## Decoder Reconstruction

After instrumenting the proprietary runner, we dumped the tensors powering the
`S2x10NetS` super-resolution head. The decoder is reproduced in
`s2dr3_decoder.py`, and the streamed loader
`load_s2x10_from_chunks("/home/mak/.cache/s2dr3/sandbox/weights")` assembles the
model without exhausting RAM.

Key attributes:
- Stem conv: 10 bands at 480×480 → 160 channels.
- 23 RRDB blocks (each = 3 ResidualDenseBlocks with growth factor 80).
- Five upsample stages doubling spatial size, yielding a 10× super-res cube.

## End-to-End Usage

`s2dr3_pipeline.py` wires the pieces together:

```python
from pathlib import Path
from s2dr3_pipeline import load_pipeline, S2DR3Pipeline

artifacts = load_pipeline(
    backbone_ckpt=Path("~/.cache/s2dr3/sandbox/local/S2DR3/gedrm_17igu4jxui").expanduser(),
    decoder_dir=Path("~/.cache/s2dr3/sandbox/weights").expanduser(),
)
pipeline = S2DR3Pipeline(artifacts)
sr_cube = pipeline.super_resolve(low_res_cube)  # (B, 10, 480, 480) -> (B, 10, 4800, 4800)
```

The helper `pipeline.extract_tokens(rgb)` exposes the ViT features if you need
them for downstream tasks (TCI/NDVI heads remain proprietary).

## Usage

```bash
python vit_s2dr3_backbone.py
```

The script will instantiate the backbone, load the decrypted safetensors blob
from the sandbox cache, and run a single forward pass to verify shapes.
Alternatively, import `load_backbone_from_safetensors` and point it at the blob
you have mounted.

## Gaps and Next Steps

- The original downstream heads (segmentation, super-resolution, spectral
  indices) remain proprietary. We will need lightweight heads that mimic their
  behaviour.
- The shim still fakes several GDAL warp calls; fixing those wrappers will make
  it easier to validate outputs locally.
- No training-time regularisation (dropout, stochastic depth) is present yet.
  Re-introduce them if we plan to fine-tune.

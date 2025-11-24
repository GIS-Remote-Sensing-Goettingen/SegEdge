import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import rasterio

# -----------------------------
# 0. GLOBAL DEBUGGING UTILITIES
# -----------------------------

# Global flag to toggle all debug prints in one place.
DEBUG = True

def debug_print(*args, **kwargs):
    """
    Centralized debug printer.
    Use this instead of print() so you can easily switch debugging off.
    """
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


# -----------------------------
# 1. MODEL BUILDING BLOCKS
# -----------------------------

class ConvBlock(nn.Module):
    """
    Standard U-Net style conv block:
    Conv → BN → ReLU → Conv → BN → ReLU
    Justification: this is exactly the same pattern you used in training,
    so the state_dict from your checkpoint will load correctly and the
    inductive bias (local smoothing + nonlinearity) is preserved.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        debug_print("ConvBlock input shape:", tuple(x.shape))
        x = self.conv(x)
        debug_print("ConvBlock output shape:", tuple(x.shape))
        return x


class UpBlock(nn.Module):
    """
    U-Net upsampling block:
    1) ConvTranspose2d for upsampling (learned upsampling)
    2) Concatenate with skip tensor
    3) ConvBlock for local refinement
    Justification: mirrors your training decoder so spatial
    semantics stay consistent.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        debug_print("UpBlock input x:", tuple(x.shape), "skip:", tuple(skip.shape))
        x = self.up(x)
        debug_print("UpBlock after upsample x:", tuple(x.shape))
        # Handle possible off-by-one spatial mismatches
        if x.shape[-2:] != skip.shape[-2:]:
            debug_print(
                "UpBlock: spatial mismatch, interpolating skip from",
                skip.shape[-2:], "to", x.shape[-2:]
            )
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        debug_print("UpBlock after concat:", tuple(x.shape))
        x = self.conv(x)
        debug_print("UpBlock output:", tuple(x.shape))
        return x


class DinoUNet(nn.Module):
    """
    Decoder taking:
      - multiscale DINO feature maps as hierarchical skips
      - raw RGB tile as final skip
    Justification: matches your training architecture exactly (bottleneck
    on deepest DINO features + upsampling over progressively finer maps).
    """
    def __init__(self, dinov3_feature_channels, unet_decoder_channels, num_classes):
        super().__init__()
        self.dinov3_feature_channels = dinov3_feature_channels
        self.decoder_channels = unet_decoder_channels

        # Bottleneck on deepest DINO feature map
        self.bottleneck = ConvBlock(dinov3_feature_channels[-1], self.decoder_channels[0])

        # Build upsampling path, with skips from DINO features (coarsest→finest) + image
        self.up_blocks = nn.ModuleList()
        skip_connection_channels = self.dinov3_feature_channels[:-1][::-1] + [3]
        in_channels = self.decoder_channels[0]

        for i, skip_channels in enumerate(skip_connection_channels):
            out_channels = (
                self.decoder_channels[i + 1]
                if i + 1 < len(self.decoder_channels)
                else self.decoder_channels[-1]
            )
            debug_print(f"UpBlock {i}: in={in_channels}, skip={skip_channels}, out={out_channels}")
            self.up_blocks.append(UpBlock(in_channels, skip_channels, out_channels))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, image, dinov3_features):
        """
        image:            (B, 3, H, W)
        dinov3_features:  list of 4 tensors, each (B, C, Hp_i, Wp_i)
        """
        debug_print("DinoUNet forward: image shape:", tuple(image.shape))
        debug_print("DinoUNet forward: #features:", len(dinov3_features))
        for i, f in enumerate(dinov3_features):
            debug_print(f"  feature[{i}] shape:", tuple(f.shape))

        features_reversed = dinov3_features[::-1]

        bottleneck_input = features_reversed[0]   # deepest feature map
        debug_print("Bottleneck input shape:", tuple(bottleneck_input.shape))
        x = self.bottleneck(bottleneck_input)

        # 3 remaining feature maps + raw image
        skip_connections = [features_reversed[1], features_reversed[2], features_reversed[3], image]
        for i, up_block in enumerate(self.up_blocks):
            debug_print(f"Calling UpBlock {i}")
            x = up_block(x, skip_connections[i])

        logits = self.final_conv(x)
        debug_print("DinoUNet final logits shape:", tuple(logits.shape))
        return logits


# -----------------------------
# 2. DINO FEATURE EXTRACTION
# -----------------------------

def extract_multiscale_features(image_hw3, model, processor, device, layers, ps=16):
    """
    image_hw3: numpy array (H, W, 3), raw values (same as during training).
    Returns:   list of feature maps [ (C, Hp, Wp), ... ] on CPU.

    Justification: replicates your training-time feature extraction, including
    processor behavior and token->spatial reshaping, so that the decoder sees
    exactly the same type of inputs it saw during training.
    """
    debug_print("extract_multiscale_features: input image shape:", image_hw3.shape,
                "dtype:", image_hw3.dtype, "min:", image_hw3.min(), "max:", image_hw3.max())

    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False
    ).to(device)

    debug_print("Processor pixel_values shape:", tuple(inputs["pixel_values"].shape),
                "dtype:", inputs["pixel_values"].dtype)

    R = getattr(model.config, "num_register_tokens", 0)
    debug_print("Model num_register_tokens:", R)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
        debug_print("Total hidden_states:", len(hidden_states))

    _, _, Hproc, Wproc = inputs["pixel_values"].shape
    debug_print("Hproc, Wproc:", Hproc, Wproc)
    feature_maps = []

    for layer_idx in layers:
        layer_output = hidden_states[layer_idx]          # (B, 1+R+N, C)
        debug_print(f"Layer {layer_idx} output shape:", tuple(layer_output.shape))

        patch_tokens = layer_output[:, 1 + R:, :]        # remove cls + register tokens
        debug_print(f"Layer {layer_idx} patch_tokens shape:", tuple(patch_tokens.shape))

        Hp, Wp = Hproc // ps, Wproc // ps
        N, C = patch_tokens.shape[1], patch_tokens.shape[2]
        debug_print(f"Layer {layer_idx}: Hp={Hp}, Wp={Wp}, N={N}, C={C}")

        assert N == Hp * Wp, f"Unexpected token count: N={N}, Hp={Hp}, Wp={Wp}"

        feats = patch_tokens.reshape(1, Hp, Wp, C).permute(0, 3, 1, 2)  # (1, C, Hp, Wp)
        debug_print(f"Layer {layer_idx} reshaped feats:", tuple(feats.shape))
        feature_maps.append(feats.squeeze(0).cpu())                      # (C, Hp, Wp)

    debug_print("extract_multiscale_features: returning", len(feature_maps), "feature maps")
    return feature_maps


# -----------------------------
# 3. TILE-BASED INFERENCE
# -----------------------------

def infer_full_image(
    input_tif,
    checkpoint_path,
    output_tif,
    model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    tile_size=512,
    device=None
):
    """
    Runs sliding-window inference on a georeferenced 3-band GeoTIFF and
    writes a 1-band mask GeoTIFF aligned to the input.

    Justification: large scenes likely don't fit into GPU memory at once,
    so tiling preserves spatial alignment while keeping memory bounded.
    """
    debug_print("infer_full_image called with:")
    debug_print("  input_tif:", input_tif)
    debug_print("  checkpoint_path:", checkpoint_path)
    debug_print("  output_tif:", output_tif)
    debug_print("  model_name:", model_name)
    debug_print("  tile_size:", tile_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Using device:", device)

    # 3.1 Load input GeoTIFF and metadata
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        debug_print("Input profile:", profile)
        img_full = src.read()          # (C, H, W)
        debug_print("Raw read image_full shape (C,H,W):", img_full.shape,
                    "dtype:", img_full.dtype)
        img_full = np.transpose(img_full, (1, 2, 0))  # (H, W, 3)
        height, width, channels = img_full.shape
        debug_print("Transposed img_full shape (H,W,C):", img_full.shape)

    assert channels == 3, f"Expected 3 bands, got {channels}"

    # Output mask array (int64 or uint8)
    pred_full = np.zeros((height, width), dtype=np.uint8)
    debug_print("Initialized pred_full with shape:", pred_full.shape)

    # 3.2 Build backbone + UNet and load weights
    layers = [5, 11, 17, 23]
    dinov3_channels = [1024, 1024, 1024, 1024]
    unet_channels = [512, 256, 128, 64]
    num_classes = 2

    debug_print("Loading processor and backbone...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).eval().to(device)
    debug_print("Backbone loaded; device:", next(backbone.parameters()).device)

    debug_print("Building DinoUNet...")
    unet = DinoUNet(dinov3_channels, unet_channels, num_classes=num_classes).to(device)
    debug_print("DinoUNet created; device:", next(unet.parameters()).device)

    # Count parameters for sanity
    n_params = sum(p.numel() for p in unet.parameters())
    debug_print("DinoUNet total parameters:", n_params)

    debug_print("Loading checkpoint state_dict from:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    debug_print("State_dict loaded. Missing keys:", missing)
    debug_print("State_dict loaded. Unexpected keys:", unexpected)
    unet.eval()

    # Patch size for DINO (16 for vitl16, 14 for vitl14)
    ps = 14 if "vitl14" in model_name else 16
    debug_print("Using patch size ps:", ps)

    # 3.3 Sliding-window over the full image
    n_tiles_y = math.ceil(height / tile_size)
    n_tiles_x = math.ceil(width / tile_size)
    debug_print("Number of tiles (y,x):", n_tiles_y, n_tiles_x)

    tile_counter = 0
    for ty, y in enumerate(range(0, height, tile_size)):
        for tx, x in enumerate(range(0, width, tile_size)):
            tile_counter += 1
            y_min, x_min = y, x
            y_max, x_max = y + tile_size, x + tile_size

            # Shift last tiles to fit exactly in image (same logic as training)
            if y_max > height:
                y_min, y_max = height - tile_size, height
            if x_max > width:
                x_min, x_max = width - tile_size, width

            debug_print(
                f"Tile {tile_counter} (grid {ty},{tx}):",
                "y:", (y_min, y_max), "x:", (x_min, x_max)
            )

            # Extract tile
            img_tile = img_full[y_min:y_max, x_min:x_max, :]   # (tile, tile, 3)
            debug_print("  img_tile shape:", img_tile.shape,
                        "dtype:", img_tile.dtype,
                        "min:", img_tile.min(), "max:", img_tile.max())

            # Skip empty tiles (optional; same as your training heuristic)
            if np.max(img_tile) == 0:
                debug_print("  Skipping tile: max == 0")
                continue

            # 3.3.1 Prepare inputs
            img_tile_raw = img_tile.astype(np.float32)              # for DINO
            img_tile_norm = (img_tile_raw / 255.0).astype(np.float32)  # for UNet

            debug_print("  img_tile_raw stats: min", img_tile_raw.min(),
                        "max", img_tile_raw.max())
            debug_print("  img_tile_norm stats: min", img_tile_norm.min(),
                        "max", img_tile_norm.max())

            img_t = (
                torch.from_numpy(img_tile_norm)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )  # (1,3,H,W)
            debug_print("  img_t shape:", tuple(img_t.shape),
                        "dtype:", img_t.dtype, "device:", img_t.device)

            # 3.3.2 DINO multiscale features for this tile
            try:
                feats = extract_multiscale_features(
                    img_tile_raw,
                    model=backbone,
                    processor=processor,
                    device=device,
                    layers=layers,
                    ps=ps
                )
            except Exception as e:
                debug_print("  ERROR during feature extraction on tile", tile_counter, ":", repr(e))
                raise

            # Add batch dimension expected by DinoUNet
            feats_batched = []
            for i, f in enumerate(feats):
                debug_print(f"  feature[{i}] before batch shape:", f.shape, "dtype:", f.dtype)
                fb = f.to(device).unsqueeze(0)
                debug_print(f"  feature[{i}] batched shape:", fb.shape, "device:", fb.device)
                feats_batched.append(fb)

            # 3.3.3 Forward pass
            try:
                with torch.no_grad():
                    logits = unet(img_t, feats_batched)   # (1,2,h,w)
            except Exception as e:
                debug_print("  ERROR during UNet forward on tile", tile_counter, ":", repr(e))
                raise

            debug_print("  logits shape:", tuple(logits.shape),
                        "dtype:", logits.dtype, "device:", logits.device)

            # Resize logits back to tile spatial size if needed
            if logits.shape[-2:] != img_t.shape[-2:]:
                debug_print("  Resizing logits from", logits.shape[-2:],
                            "to", img_t.shape[-2:])
                logits = F.interpolate(logits, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
                debug_print("  logits after resize shape:", tuple(logits.shape))

            pred_tile = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H,W)
            debug_print("  pred_tile shape:", pred_tile.shape,
                        "unique labels:", np.unique(pred_tile))

            # 3.3.4 Write into full prediction canvas
            pred_full[y_min:y_max, x_min:x_max] = pred_tile

    # 3.4 Save prediction as GeoTIFF aligned to input
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0
    )
    debug_print("Output profile:", profile)

    out_dir = os.path.dirname(output_tif) or "."
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(pred_full, 1)

    debug_print("Final pred_full stats: min", pred_full.min(),
                "max", pred_full.max(), "unique:", np.unique(pred_full))
    print(f"Saved prediction to: {output_tif}")


# -----------------------------
# 4. ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print("Main: using device:", DEVICE)

    INPUT_TIF = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/dop20_599000_5984000_1km_20cm.tif"
    CHECKPOINT = "/home/mak/PycharmProjects/SegEdge/experiments/dinov3Seg/weights/dinounet_best.pth"
    OUTPUT_TIF = "/home/mak/PycharmProjects/SegEdge/experiments/dinov3Seg/test/output_prediction.tif"

    infer_full_image(
        input_tif=INPUT_TIF,
        checkpoint_path=CHECKPOINT,
        output_tif=OUTPUT_TIF,
        model_name="facebook/dinov3-vitl16-pretrain-sat493m",
        tile_size=512,
        device=DEVICE
    )

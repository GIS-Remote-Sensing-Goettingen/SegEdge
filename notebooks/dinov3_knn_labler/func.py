
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imread

def l2_normalize(feats, eps=1e-8):
    """
    L2-normalize features along the last dimension.
    This is standard before cosine-sim / kNN with cosine metric.
    """
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    return feats / norms

def extract_patch_features_single_scale(
    image_hw3,
    model,
    processor,
    device,
    ps=16,
    aggregate_layers=None,
):
    """
    Extract DINOv3 patch features for a single scale.
    We disable internal resizing/cropping in the processor and rely
    on external cropping to multiples of ps.
    Returns:
        feats: (Hp, Wp, C) numpy, L2-normalized
        Hp, Wp: patch-grid size
    """
    # processor: keep resolution
    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False
    ).to(device)

    px = inputs["pixel_values"]
    _, _, Hproc, Wproc = px.shape

    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            layers = [hs[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)

    R = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + R:, :]  # drop CLS + registers

    N, C = patch_tokens.shape[1], patch_tokens.shape[2]
    Hp = Hproc // ps
    Wp = Wproc // ps
    assert Hp * Wp == N, f"Patch-grid mismatch: {Hp}*{Wp} != {N}"

    feats = patch_tokens[0].cpu().numpy().reshape(Hp, Wp, C)
    feats = l2_normalize(feats)
    return feats, Hp, Wp
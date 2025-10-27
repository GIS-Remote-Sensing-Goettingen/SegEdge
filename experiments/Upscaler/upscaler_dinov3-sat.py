"""
Feature upscaling and clustering using DINOv3 and AnyUp.

This script:
1. Loads an image and extracts DINOv3 features
2. Pads the image to multiples of patch size (16)
3. Upscales features using AnyUp
4. Clusters upscaled features with k-means
5. Visualizes the cluster map
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import MiniBatchKMeans


# Constants
PATCH_SIZE = 16
IMG_SIZE = 800
N_CLUSTERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = "to_upscale.png"  # Path to input image

def load_dinov3_model(device: str):
    """
    Load DINOv3 ViT-L/16 model and processor.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (processor, model)
    """
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-sat493m"
    )
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-sat493m",
        token=True
    ).to(device).eval()

    return processor, model


def pad_image_to_patch_size(img: Image.Image, patch_size: int, processor):
    """
    Pad image to next multiple of patch size for clean tiling.

    Args:
        img: PIL Image
        patch_size: ViT patch size (typically 16)
        processor: DINOv3 processor for normalization stats

    Returns:
        Tuple of (padded normalized tensor, original width, original height)
    """
    W, H = img.size

    # compute next multiples so patches tile cleanly
    Wm = ((W + patch_size - 1) // patch_size) * patch_size
    Hm = ((H + patch_size - 1) // patch_size) * patch_size
    pad_w, pad_h = Wm - W, Hm - H

    print(f"original: {W}x{H}  →  next multiples of {patch_size}: {Wm}x{Hm}  "
          f"(add {pad_w} px width, {pad_h} px height)")

    # padding: (left, top, right, bottom)
    pad_to_multiple = transforms.Pad((0, 0, pad_w, pad_h), fill=0)
    to_tensor = transforms.Compose([pad_to_multiple, transforms.ToTensor()])

    # convert to tensor
    img_tensor = to_tensor(img)
    print("tensor shape:", tuple(img_tensor.shape))

    # normalize using DINOv3 processor stats
    normalize = transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
    img_norm = normalize(img_tensor)

    return img_norm, W, H


def extract_dinov3_features(hr_image: torch.Tensor, model, device: str):
    """
    Extract spatial patch features from DINOv3 model.

    Args:
        hr_image: Normalized image tensor (B, C, H, W)
        model: DINOv3 model
        device: Device for computation

    Returns:
        Feature tensor of shape (B, D, Hp, Wp)
    """
    # forward pass to get patch tokens
    with torch.no_grad():
        out = model(pixel_values=hr_image)

    # drop CLS and register tokens
    R = getattr(model.config, "num_register_tokens", 0)
    patch = out.last_hidden_state[:, 1 + R:, :]
    print("patch shape (1,P,D):", tuple(patch.shape))

    # compute patch grid dims
    ps, Hx, Wx = PATCH_SIZE, hr_image.shape[2], hr_image.shape[3]
    Hp, Wp = Hx // ps, Wx // ps

    # reshape to spatial grid (B, Hp, Wp, D) then to (B, D, Hp, Wp) for upsampler
    patch_grid = patch.reshape(1, Hp, Wp, patch.shape[-1])
    lr_features = patch_grid.permute(0, 3, 1, 2).contiguous()

    print(f"lr_features shape: {tuple(lr_features.shape)}")

    return lr_features


def upsample_features(hr_image: torch.Tensor, lr_features: torch.Tensor):
    """
    Upsample features using AnyUp model.

    Args:
        hr_image: High-resolution image tensor (B, C, H, W)
        lr_features: Low-resolution features (B, D, Hp, Wp)

    Returns:
        Upsampled features (B, D, H, W)
    """
    # now feed hr_image and lr_features to the upsampler
    upsampler = torch.hub.load(
        "wimmerth/anyup",
        "anyup",
        verbose=False
    ).cpu().eval()

    hr_image_cpu = hr_image.cpu()
    lr_features_cpu = lr_features.cpu()

    with torch.no_grad():
        hr_features = upsampler(hr_image_cpu, lr_features_cpu, q_chunk_size=64)

    print("hr_features shape:", tuple(hr_features.shape))

    return hr_features


def cluster_features(hr_features: torch.Tensor, n_clusters: int = 8):
    """
    Cluster upsampled features using k-means.

    Args:
        hr_features: Upsampled feature tensor (B, C, H, W)
        n_clusters: Number of clusters for k-means

    Returns:
        Tuple of (cluster labels, height, width)
    """
    B, C, H, W = hr_features.shape

    # we'll cluster in feature space, so build a 2D matrix X = (num_pixels × feature_dim)
    X = hr_features.permute(0, 2, 3, 1).reshape(B, -1, C)
    print("X shape:", X.shape)

    # L2-normalize each patch vector so distance ~ cosine distance
    # (helps k-means separate textures like roads/roofs)
    X = X[0].detach().cpu().numpy().astype("float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # print sanity checks before clustering
    print("X shape:", X.shape, " mean L2 norm:", float(np.linalg.norm(X, axis=1).mean()))

    # we'll use MiniBatchKMeans to group similar patch embeddings
    # (fast, memory-friendly) into k semantic clusters
    # choose an initial cluster count (k=8 is a reasonable start to separate
    # roads/roofs/veg/shadows etc.)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=12,
        batch_size=2048,
        max_iter=100
    )

    # run clustering on the L2-normalized embeddings to obtain one label per pixel (0..k-1)
    labels = kmeans.fit_predict(X)

    # report how many pixels fell into each cluster so we can gauge balance
    # before mapping to the image grid
    print("labels shape:", labels.shape, " | counts per cluster:",
          np.bincount(labels, minlength=n_clusters))

    return labels, H, W


def visualize_clusters(labels: np.ndarray, height: int, width: int):
    """
    Visualize cluster assignments as a spatial map.

    Args:
        labels: Cluster labels for each pixel
        height: Image height
        width: Image width
    """
    label_grid = labels.reshape(height, width)

    # quick sanity: report grid shape and how many distinct clusters are present
    print("label_grid:", label_grid.shape, " unique clusters:",
          np.unique(label_grid).size)

    # render the label grid with a categorical colormap and no smoothing between pixels
    plt.imshow(label_grid, cmap="tab20", interpolation="nearest")

    # print a quick status so you know what you're seeing (grid size and number of clusters)
    print(f"visualizing clusters @ {label_grid.shape[0]}x{label_grid.shape[1]} "
          f"with {label_grid.max()+1} clusters")

    # save the figure
    plt.axis("off")
    plt.savefig("cluster_map.png", bbox_inches="tight", pad_inches=0, dpi=300)


def main():
    """Main execution function."""
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")

    # Load image
    img = Image.open(IMG).convert("RGB")

    # Load DINOv3 model and processor
    processor, model = load_dinov3_model(DEVICE)

    # Pad image to patch size multiples and normalize
    img_norm, orig_w, orig_h = pad_image_to_patch_size(img, PATCH_SIZE, processor)

    # add batch dim and move to device (this is hr_image for AnyUp)
    hr_image = img_norm.unsqueeze(0).to(DEVICE)
    print("hr_image for upsampler:", tuple(hr_image.shape), " device:", hr_image.device)

    # Extract DINOv3 features
    lr_features = extract_dinov3_features(hr_image, model, DEVICE)

    # Upsample features
    hr_features = upsample_features(hr_image, lr_features)

    # Cluster features
    labels, H, W = cluster_features(hr_features, n_clusters=N_CLUSTERS)

    # Visualize clusters
    visualize_clusters(labels, H, W)


if __name__ == "__main__":
    main()


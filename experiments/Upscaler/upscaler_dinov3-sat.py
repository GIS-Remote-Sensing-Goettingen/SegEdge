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
import os

# Constants
PATCH_SIZE = 16
IMG_SIZE = 800
N_CLUSTERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = "to_upscale.png"  # Path to input image

def load_dinov3_model(device: str):
    """
    Load DINOv3 ViT-L/16 model and processor from local cache.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (processor, model)
    """
    print("Loading DINOv3 processor from cache...")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-sat493m",
        local_files_only=True  # Force using cached files
    )
    print("Processor loaded successfully!")

    print("Loading DINOv3 model from cache...")
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-sat493m",
        local_files_only=True,  # Force using cached files
        token=True
    ).to(device).eval()
    print(f"Model loaded successfully and moved to {device}!")

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


def upsample_features(hr_image: torch.Tensor, lr_features: torch.Tensor, device: str):
    """
    Upsample features using AnyUp model in tiles.

    Args:
        hr_image: High-resolution image tensor (B, C, H, W)
        lr_features: Low-resolution features (B, D, Hp, Wp)
        device: Device for computation

    Returns:
        Upsampled features (B, D, H, W)
    """
    _, _, H, W = hr_image.shape
    tile_size = 512  # Process 512×512 tiles
    min_tile_size = 64  # Minimum tile size to avoid padding errors

    print("Loading AnyUp model from cache...")
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", force_reload=False).to(device).eval()
    print("Loaded AnyUp model successfully!")

    hr_features_list = []

    for i in range(0, H, tile_size):
        row_features = []
        for j in range(0, W, tile_size):
            # Extract tile boundaries
            h_end = min(i + tile_size, H)
            w_end = min(j + tile_size, W)

            # Check if tile is too small
            tile_h = h_end - i
            tile_w = w_end - j

            # If edge tile is too small, extend backwards to get minimum size
            if tile_h < min_tile_size:
                i_start = max(0, h_end - min_tile_size)
            else:
                i_start = i

            if tile_w < min_tile_size:
                j_start = max(0, w_end - min_tile_size)
            else:
                j_start = j

            # Extract tiles and make them contiguous
            hr_tile = hr_image[:, :, i_start:h_end, j_start:w_end].contiguous()
            lr_tile = lr_features[:, :, i_start // 16:(h_end // 16), j_start // 16:(w_end // 16)].contiguous()

            print(f"Processing tile [{i_start}:{h_end}, {j_start}:{w_end}] - HR: {hr_tile.shape}, LR: {lr_tile.shape}")

            # Process tile
            with torch.no_grad():
                tile_features = upsampler(hr_tile, lr_tile, q_chunk_size=64)

            # If we extended the tile, crop back to original boundary
            if i_start < i or j_start < j:
                crop_i = i - i_start if i_start < i else 0
                crop_j = j - j_start if j_start < j else 0
                tile_features = tile_features[:, :, crop_i:, crop_j:]

            row_features.append(tile_features)

        hr_features_list.append(torch.cat(row_features, dim=3))

        # Print memory status after each row
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024 ** 3
            allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
            free = total - allocated

            print(f"GPU Memory - Total: {total:.2f} GB | Allocated: {allocated:.2f} GB | "
                  f"Reserved: {reserved:.2f} GB | Free: {free:.2f} GB")

    hr_features = torch.cat(hr_features_list, dim=2)
    print("Final hr_features shape:", tuple(hr_features.shape))

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

    # Reshape to (num_pixels, feature_dim) - still on CPU as tensor
    X = hr_features.permute(0, 2, 3, 1).reshape(B, -1, C)
    print("X shape:", X.shape)

    # Convert to numpy in smaller chunks to avoid OOM
    print("Converting to numpy and normalizing in batches...")
    chunk_size = 100000  # Smaller chunks for safety
    num_vectors = X.shape[1]
    num_chunks = (num_vectors + chunk_size - 1) // chunk_size

    # Pre-allocate numpy array
    X_np = np.empty((num_vectors, C), dtype=np.float32)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_vectors)

        # Convert chunk to numpy
        chunk = X[0, start_idx:end_idx].numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
        X_np[start_idx:end_idx] = chunk / norms

        if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            print(f"  Processed {end_idx}/{num_vectors} vectors")

    print("X_np shape:", X_np.shape, " mean L2 norm:", float(np.linalg.norm(X_np[:10000], axis=1).mean()))

    # Use MiniBatchKMeans for memory efficiency
    print(f"Running MiniBatchKMeans with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=8,
        batch_size=2048,
        max_iter=100,
        verbose=1
    )

    labels = kmeans.fit_predict(X_np)

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
    hr_features = upsample_features(hr_image, lr_features, DEVICE)

    # Cluster features
    labels, H, W = cluster_features(hr_features, n_clusters=N_CLUSTERS)

    # Visualize clusters
    visualize_clusters(labels, H, W)


if __name__ == "__main__":
    main()


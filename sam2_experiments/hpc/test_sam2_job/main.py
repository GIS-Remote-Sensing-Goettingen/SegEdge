import time
from matplotlib import pyplot as plt
import cv2
import torch
import numpy as np
import supervision as sv
import copy
import gc

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def generate_patches(image, patch_size=1024, overlap=128):
    """
    Split image into overlapping patches.

    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch
        overlap: Overlap between adjacent patches

    Returns:
        List of (patch, x_offset, y_offset) tuples
    """
    h, w = image.shape[:2]

    # Validate minimum size
    if h < overlap or w < overlap:
        raise ValueError(f"Image too small ({h}x{w}) for overlap {overlap}")

    stride = patch_size - overlap
    patches = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            # Extract patch
            patch = image[y:y_end, x:x_end]

            # Pad if needed
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch.copy()
                patch = padded

            patches.append((patch, x, y))

    print(f"[INFO] Generated {len(patches)} patches of size {patch_size}x{patch_size} with {overlap}px overlap")
    return patches


def merge_masks(all_masks, image_shape, overlap=128):
    """
    Merge masks from overlapping patches with NMS in overlap regions.

    Args:
        all_masks: List of mask dicts with 'segmentation', 'bbox', 'offset_x', 'offset_y'
        image_shape: (H, W) of original image
        overlap: Overlap size

    Returns:
        Merged list of masks
    """
    print(f"[INFO] Merging {len(all_masks)} masks from patches")

    # Translate mask coordinates to global image space
    global_masks = []
    for mask_dict in all_masks:
        mask = copy.deepcopy(mask_dict)
        offset_x = mask.pop('offset_x', 0)
        offset_y = mask.pop('offset_y', 0)

        # Adjust bbox
        x, y, w, h = mask['bbox']
        mask['bbox'] = [x + offset_x, y + offset_y, w, h]

        # Create full-size segmentation mask
        full_seg = np.zeros(image_shape, dtype=bool)
        seg = mask['segmentation']
        y_end = min(offset_y + seg.shape[0], image_shape[0])
        x_end = min(offset_x + seg.shape[1], image_shape[1])
        full_seg[offset_y:y_end, offset_x:x_end] = seg[:y_end - offset_y, :x_end - offset_x]
        mask['segmentation'] = full_seg

        global_masks.append(mask)

    # Simple deduplication: remove masks with high IoU in overlap regions
    if len(global_masks) > 1:
        keep = []
        for i, mask in enumerate(global_masks):
            should_keep = True
            for kept_mask in keep:
                intersection = np.logical_and(mask['segmentation'], kept_mask['segmentation']).sum()
                union = np.logical_or(mask['segmentation'], kept_mask['segmentation']).sum()
                iou = intersection / union if union > 0 else 0

                if iou > 0.5:  # Duplicate threshold
                    should_keep = False
                    break

            if should_keep:
                keep.append(mask)

        print(f"[INFO] Kept {len(keep)}/{len(global_masks)} masks after deduplication")
        return keep

    return global_masks


def print_vram_usage(device=0):
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_allocated(device)
    print(f"[VRAM] allocated: {alloc/1024**2:.2f} MiB, reserved: {reserved/1024**2:.2f} MiB, free: {free/1024**2:.2f} MiB, total: {total/1024**2:.2f} MiB, peak: {peak/1024**2:.2f} MiB")



def plot_images_and_save(images, grid_size, titles, save_path):
    print("[DEBUG] plot_images_and_save called")
    print("  grid_size:", grid_size)
    print("  #images:", len(images))
    for i, img in enumerate(images):
        print(f"    image {i} shape:", img.shape if hasattr(img, "shape") else type(img))
    t0 = time.time()
    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]
            # assume BGR images
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[WARN] cvtColor failed for image {idx}: {e}")
                rgb = img
            ax.imshow(rgb)
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    dt = time.time() - t0
    print(f"[DEBUG] Saved grid image to {save_path} in {dt:.3f}s")


if __name__ == '__main__':
    print("[INFO] Starting script")
    t_start = time.time()
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("[INFO] GPU device properties:", props)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device:", DEVICE)
    print_vram_usage()

    CHECKPOINT = "/user/davide.mattioli/u20330/SegEdge/sam2_experiments/Models/sam2_hiera_large.pt"
    CONFIG = "sam2_hiera_l.yaml"
    print("[INFO] Building SAM2 model")
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

    print("[INFO] Instantiating mask generator")
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=256,
        points_per_batch=64,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        min_mask_region_area=1000
    )

    IMAGE_PATH = "/user/davide.mattioli/u20330/SegEdge/sam2_experiments/Images/S2L2Ax10_T32UNC-6bd2a7faa-20250813_TCI.jpg"
    print("[INFO] Reading image from", IMAGE_PATH)
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image at {IMAGE_PATH}")
    print("[DEBUG] image_bgr shape:", image_bgr.shape)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Generate patches
    PATCH_SIZE = 1024
    OVERLAP = 128
    patches = generate_patches(image_rgb, patch_size=PATCH_SIZE, overlap=OVERLAP)

    # Process each patch
    all_masks = []
    for idx, (patch_rgb, offset_x, offset_y) in enumerate(patches):
        print(f"[INFO] Processing patch {idx + 1}/{len(patches)} at offset ({offset_x}, {offset_y})")
        t_patch = time.time()

        patch_masks = mask_generator_2.generate(patch_rgb)

        # Add offset information to each mask
        for mask in patch_masks:
            mask['offset_x'] = offset_x
            mask['offset_y'] = offset_y

        all_masks.extend(patch_masks)
        print(f"[DEBUG] Patch {idx + 1} generated {len(patch_masks)} masks in {time.time() - t_patch:.2f}s")

        # Clear cache periodically
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Merge masks
    merged_masks = merge_masks(all_masks, image_shape=image_rgb.shape[:2], overlap=OVERLAP)

    print("[INFO] Annotating image with merged masks")
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=merged_masks)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    print_vram_usage()

    print("[INFO] Calling plot_images_and_save ...")
    plot_images_and_save(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image'],
        save_path="my_grid.png"
    )

    print("[INFO] Total script time:", time.time() - t_start, "s")

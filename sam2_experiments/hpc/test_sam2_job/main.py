import time
from matplotlib import pyplot as plt
import cv2
import torch
import base64
import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



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
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    IMAGE_PATH = "/user/davide.mattioli/u20330/SegEdge/sam2_experiments/Images/super_small_sa.jpg"
    print("[INFO] Reading image from", IMAGE_PATH)
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image at {IMAGE_PATH}")
    print("[DEBUG] image_bgr shape:", image_bgr.shape)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("[INFO] Generating masks")
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=256,
        points_per_batch=128,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        min_mask_region_area=1000
    )

    t_mask = time.time()
    sam2_result_2 = mask_generator_2.generate(image_rgb)
    print("[DEBUG] Mask generation took", time.time() - t_mask, "s")

    print("[INFO] Annotating image")
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam2_result_2)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    print("[DEBUG] annotated_image shape:", annotated_image.shape if hasattr(annotated_image, "shape") else type(annotated_image))
    print_vram_usage()

    print("[INFO] Calling plot_images_and_save ...")
    plot_images_and_save(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image'],
        save_path="my_grid.png"
    )

    print("[INFO] Total script time:", time.time() - t_start, "s")

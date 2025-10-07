from matplotlib import pyplot as plt

import cv2
import torch
import base64

import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def plot_images_and_save(images, grid_size, titles, save_path):
    nrows, ncols = grid_size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]
            # assume BGR images, convert to RGB
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = f"/user/davide.mattioli/u20330/SegEdge/sam2_experiments/Models/sam2_hiera_large.pt"
    CONFIG = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    IMAGE_PATH = f"/user/davide.mattioli/u20330/SegEdge/sam2_experiments/Images/super_small_sa.jpg"

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=64,
        points_per_batch=32,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        min_mask_region_area=1000
    )
    sam2_result_2 = mask_generator_2.generate(image_rgb)


    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam2_result_2)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    def plot_images_and_save(images, grid_size, titles, save_path):
        nrows, ncols = grid_size
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
        axes = axes.flatten() if nrows*ncols > 1 else [axes]
        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx]
                # assume BGR images, convert to RGB
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if titles and idx < len(titles):
                    ax.set_title(titles[idx])
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    # Usage:
    plot_images_and_save(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image'],
        save_path="my_grid.png"
    )

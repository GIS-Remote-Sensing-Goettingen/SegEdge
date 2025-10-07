

if __name__ == '__main__':
    import cv2
    import torch
    import base64

    import numpy as np
    import supervision as sv

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("done")


# Default configuration for the DINO zero-shot pipeline.
# Adjust these paths/hyperparameters as needed; main.py will import them.

# Paths
IMG_PATH = "data/dop20_593000_5979000_1km_20cm.tif"          # Image A
IMG2_PATH = "data/dop20_592000_5982000_1km_20cm.tif"         # Image B
LAB_PATH = "data/planet_labels_2022.tif"                     # SH_2022 raster
GT_VECTOR_PATH = "data/labels_final.shp"                     # Ground truth for B
FEATURE_DIR = "data/dino_features"
BANK_CACHE_DIR = "data/dino_features/banks"
PLOT_DIR = "data/plots"
BEST_SETTINGS_PATH = "data/plots/best_settings.yml"

# Model / buffers
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
BUFFER_M = 8.0

# Grid search
K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100, 150, 200, 300, 500]
THRESHOLDS = [float(x) for x in __import__("numpy").linspace(0.01, 0.9, 50)]

# CRF search
PROB_SOFTNESS_VALUES = [0.03, 0.05, 0.08]
POS_W_VALUES = [3.0, 4.0]
POS_XY_STD_VALUES = [3.0]
BILATERAL_W_VALUES = [5.0, 7.0]
BILATERAL_XY_STD_VALUES = [25.0, 50.0]
BILATERAL_RGB_STD_VALUES = [3.0, 5.0]

# Shadow filtering (RGB weighted sum) after CRF
SHADOW_WEIGHT_SETS = [
    (1.0, 1.0, 1.0),
    (0.7, 1.0, 1.0),
    (0.5, 0.8, 1.0),
]

# Better initial guesses for 8-bit RGB in [0,255]
SHADOW_THRESHOLDS = [180, 210, 240, 270, 300, 330, 360, 450 ,500]

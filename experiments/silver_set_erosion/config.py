# Default configuration for the DINO zero-shot pipeline.
# Adjust these paths/hyperparameters as needed; main.py will import them.

# Paths
IMG_PATH = "data/tiles/dop20_596000_5974000_1km_20cm.tif"    # Image A (fallback)
IMG2_PATH = "data/dop20_592000_5982000_1km_20cm.tif"         # Image B
LAB_PATH = "data/lables/planet_labels_2022.tif"              # SH_2022 raster
# Ground truth for B (evaluation). If GT_VECTOR_PATHS is provided, they will be union-merged.
GT_VECTOR_PATH = "data/lables/labels_final.shp"
GT_VECTOR_PATHS = [
    "data/lables/lables_1.shp",
    "data/lables/lables_2.shp",
    "data/lables/lables_3.shp",
]
FEATURE_DIR = "data/dino_features"
BANK_CACHE_DIR = "data/dino_features/banks"
PLOT_DIR = "data/plots"
BEST_SETTINGS_PATH = "data/plots/best_settings.yml"
LOG_PATH = "data/plots/run.log"

# Optional: multiple labeled source images (Image A list) to build larger banks / XGB training data.
# If set, the pipeline will iterate over these paths and concatenate their banks.
# LAB_PATH is used for all sources unless you provide per-source label rasters via LAB_A_PATHS.
IMG_A_PATHS = [
    "data/tiles/dop20_596000_5974000_1km_20cm.tif",
    "data/tiles/dop20_596000_5975000_1km_20cm.tif",
    "data/tiles/dop20_596000_5976000_1km_20cm.tif",
    "data/tiles/dop20_596000_5977000_1km_20cm.tif",
    "data/tiles/dop20_596000_5983000_1km_20cm.tif",
]
LAB_A_PATHS = None  # e.g. ["data/a1_labels.tif", "data/a2_labels.tif"] (must match IMG_A_PATHS length)

# Model / buffers
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
BUFFER_M = 8.0
TILE_SIZE = 1024
STRIDE = 512
PATCH_SIZE = 16  # DINO patch
NEG_ALPHA = 1.0  # kNN negative bank weight
POS_FRAC_THRESH = 0.1  # fraction for positive patch labeling in A

# Optional: add local context to patch embeddings by averaging over a (2r+1)x(2r+1) patch neighborhood.
# This affects bank building, kNN scoring, and XGB training/scoring (features are still cached raw on disk).
FEAT_CONTEXT_RADIUS = 0  # 0 disables; try 1 or 2 for more context

# Grid search
#K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100 ,150, 200, 300, 500]
K_VALUES = [175,200,250]
THRESHOLDS = [float(x) for x in __import__("numpy").linspace(0.01, 0.9, 100)]

# CRF search
PROB_SOFTNESS_VALUES = [0.03, 0.05, 0.08]
POS_W_VALUES = [3.0, 4.0]
POS_XY_STD_VALUES = [3.0]
BILATERAL_W_VALUES = [5.0, 7.0]
BILATERAL_XY_STD_VALUES = [25.0, 50.0]
BILATERAL_RGB_STD_VALUES = [3.0, 5.0]
CRF_NUM_WORKERS = 32

# Shadow filtering (RGB weighted sum) after CRF
SHADOW_WEIGHT_SETS = [
    (1.0, 1.0, 1.0),
    (0.7, 1.0, 1.0),
    (0.5, 0.8, 1.0),
    (0.5, 1.0, 0.5),
    (0.5, 0.5, 1.0),
    (0.1, 0.5, 0.5),
]

# Better initial guesses for 8-bit RGB in [0,255]
SHADOW_THRESHOLDS = [20,40,60,80,100,120,160,180, 210, 240, 270, 300, 330, 360, 450 ,500]

# Evaluation options
CLIP_GT_TO_BUFFER = True  # if True, ignore GT outside the SH buffer (max IoU can reach 1.0)

# XGBoost options
XGB_USE_GPU = True
XGB_VAL_FRACTION = 0.2
XGB_NUM_BOOST_ROUND = 10
XGB_EARLY_STOP = 40
XGB_VERBOSE_EVAL = 20
# Optional search grid (list of partial param dicts that override the base defaults in xdboost.py)
XGB_PARAM_GRID = [
    # 1. The Current Champion (Baseline to beat)
    {"max_depth": 6, "eta": 0.05, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.05, "min_child_weight": 1},

    # 2. The "Deep & Slow" (Pushing for 0.67+)
    # Slower learning (0.03) + slightly deeper trees (7) often squeezes out the last 1-2% in segmentation.
    # REQUIRES: num_boost_round=800+
    {"max_depth": 7, "eta": 0.03, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.05, "min_child_weight": 1},

    # 3. The "Regularized" Champion
    # Sometimes slightly higher alpha (0.1) helps clean up noisy borders.
    {"max_depth": 6, "eta": 0.05, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.1,  "min_child_weight": 1},
]

# SegEdge zero-shot pipeline configuration.
# Organized by: IO paths, model/tiling, scoring/search, evaluation/runtime.

# -----------------------------------------------------------------------------
# 1) IO paths and datasets
# -----------------------------------------------------------------------------

# Single-tile fallbacks (used when *_TILES lists are None).
SOURCE_TILE = "data/tiles/dop20_596000_5974000_1km_20cm.tif"
TARGET_TILE = "data/dop20_592000_5982000_1km_20cm.tif"

# Label raster used to build banks on SOURCE_* tiles.
SOURCE_LABEL_RASTER = "data/lables/planet_labels_2022.tif"

# Evaluation GT vectors (union-merged). Use EVAL_GT_VECTORS when available,
EVAL_GT_VECTORS = [
    "data/lables/lables_1.shp",
    "data/lables/lables_2.shp",
    "data/lables/lables_3.shp",
    "data/lables/labels_final.shp",
]

# Multi-tile inputs (set to None to use SOURCE_TILE).
SOURCE_TILES = [
    "data/tiles/dop20_596000_5974000_1km_20cm.tif",
    "data/tiles/dop20_596000_5975000_1km_20cm.tif",
    "data/tiles/dop20_596000_5976000_1km_20cm.tif",
    "data/tiles/dop20_596000_5977000_1km_20cm.tif",
    "data/tiles/dop20_596000_5983000_1km_20cm.tif",
]

# Validation tiles (first is used for tuning; the rest are evaluated with fixed settings).
VAL_TILES = [
    "data/tiles/dop20_596000_5974000_1km_20cm.tif",
    "data/tiles/dop20_596000_5977000_1km_20cm.tif",
    "data/tiles/dop20_596000_5983000_1km_20cm.tif",
]
HOLDOUT_TILES = [
    "data/tiles/dop20_596000_5975000_1km_20cm.tif",
    "data/tiles/dop20_596000_5976000_1km_20cm.tif",
]


# Output locations (everything except DINO features).
FEATURE_DIR = "data/dino_features"
BANK_CACHE_DIR = "data/dino_features/banks"
OUTPUT_DIR = "output"
PLOT_DIR = "output/plots"
BEST_SETTINGS_PATH = "output/best_settings.yml"
LOG_PATH = "output/run.log"

# -----------------------------------------------------------------------------
# 2) Model + tiling
# -----------------------------------------------------------------------------
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
RESAMPLE_FACTOR = 3  # 3x downsample: 0.2 m/px -> 0.6 m/px
PATCH_SIZE = 16  # ViT patch size
TILE_SIZE = 1024
STRIDE = 512
BUFFER_M = 8.0  # spatial prior buffer (meters)

# Patch labeling for banks.
POS_FRAC_THRESH = 0.05  # patch is positive if FG fraction >= this
NEG_ALPHA = 1.0  # kNN negative bank weight

# Optional patch-context smoothing (applies to banks, kNN, XGB).
FEAT_CONTEXT_RADIUS = 0  # 0 disables; try 1 or 2 for more context

# Bank label erosion (in pixels on the resampled grid). Set to 0 to keep thin positives.
BANK_EROSION_RADIUS = 0

# -----------------------------------------------------------------------------
# 3) Scoring/search
# -----------------------------------------------------------------------------

# kNN grid search
#K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100, 150, 200, 300, 500]
K_VALUES = [175, 200, 250]
THRESHOLDS = [float(x) for x in __import__("numpy").linspace(0.01, 0.9, 100)]

# CRF grid search
PROB_SOFTNESS_VALUES = [0.03, 0.05, 0.08]
POS_W_VALUES = [3.0, 4.0]
POS_XY_STD_VALUES = [3.0]
BILATERAL_W_VALUES = [5.0, 7.0]
BILATERAL_XY_STD_VALUES = [25.0, 50.0]
BILATERAL_RGB_STD_VALUES = [3.0, 5.0]
CRF_NUM_WORKERS = 16

# Shadow filtering (RGB weighted sum) after CRF
SHADOW_WEIGHT_SETS = [
    (1.0, 1.0, 1.0),
    (0.7, 1.0, 1.0),
    (0.5, 0.8, 1.0),
    (0.5, 1.0, 0.5),
    (0.5, 0.5, 1.0),
    (0.1, 0.5, 0.5),
]
SHADOW_THRESHOLDS = [20, 40, 60, 80, 100, 120, 160, 180, 210, 240, 270, 300, 330, 360, 450, 500]

# XGBoost (patch classifier)
XGB_USE_GPU = True
XGB_VAL_FRACTION = 0.2
XGB_NUM_BOOST_ROUND = 10
XGB_EARLY_STOP = 40
XGB_VERBOSE_EVAL = 20
XGB_PARAM_GRID = [
    # 1) Baseline
    {"max_depth": 6, "eta": 0.05, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.05, "min_child_weight": 1},
    # 2) Deep & slow (needs more rounds)
    {"max_depth": 7, "eta": 0.03, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.05, "min_child_weight": 1},
    # 3) More regularized
    {"max_depth": 6, "eta": 0.05, "colsample_bytree": 0.3, "subsample": 0.9, "reg_alpha": 0.1,  "min_child_weight": 1},
]

# -----------------------------------------------------------------------------
# 4) Evaluation and runtime
# -----------------------------------------------------------------------------

# If True, evaluation ignores GT outside the SH buffer (upper bound can reach 1.0).
CLIP_GT_TO_BUFFER = True

# Split evaluation (used by split_eval.py).

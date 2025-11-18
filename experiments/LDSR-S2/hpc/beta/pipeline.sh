#!/usr/bin/env bash
# pipeline.sh — stage cutout here, then submit SR job from THIS directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/output_patches}"
mkdir -p "${OUTPUT_BASE}"

# --- defaults (all relative to per-patch working directory) ---
LATITUDE=${LATITUDE:-51.5413}
LONGITUDE=${LONGITUDE:-9.9158}
START_DATE=${START_DATE:-2025-08-18}
END_DATE=${END_DATE:-2025-08-22}
EDGE_SIZE=${EDGE_SIZE:-4096}  # pixels
RESOLUTION=${RESOLUTION:-10}
ENV_PATH="${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"
# -----------------------------------------------------

printf -v PATCH_STEM 'patch_lat_%0.6f_lon_%0.6f_edge_%d' "${LATITUDE}" "${LONGITUDE}" "${EDGE_SIZE}"
PATCH_ROOT="${OUTPUT_BASE}/${PATCH_STEM}"
DATA_DIR="${PATCH_ROOT}/data_sentinel2"
OUTPUT_DIR="${PATCH_ROOT}/outputs"
LOG_DIR="${PATCH_ROOT}/logs"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

# Copy SR runner assets into the patch workspace so jobs are self-contained.
cp "${SCRIPT_DIR}/run_sr.py" "${PATCH_ROOT}/run_sr.py"
cp "${SCRIPT_DIR}/config_10m.yaml" "${PATCH_ROOT}/config_10m.yaml"

# Activate environment (login node)
module load miniforge3 || true
source activate "${ENV_PATH}"

# Stage LR GeoTIFF via cubo into current directory tree
INPUT_TIF="${DATA_DIR}/input_LR_image_${LATITUDE}_${LONGITUDE}.tif"
python -u stage_s2_cutout.py \
  --lat "${LATITUDE}" \
  --lon "${LONGITUDE}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --edge-size "${EDGE_SIZE}" \
  --resolution "${RESOLUTION}" \
  --out-path "${INPUT_TIF}"


# Submit SR job, starting in THIS directory:
# - Each patch gets an isolated working directory under PATCH_ROOT.
# - We export INPUT_TIF and OUTPUT_DIR into the job env.
JOB_ID=$(sbatch --chdir "${PATCH_ROOT}" \
  --export=ALL,INPUT_TIF="$(realpath "${INPUT_TIF}")",OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")",LOG_DIR="$(realpath "${LOG_DIR}")" \
  "${SCRIPT_DIR}/sr_job.sh" | awk '{print $4}')
echo "Submitted job ${JOB_ID}. SLURM files will appear here."

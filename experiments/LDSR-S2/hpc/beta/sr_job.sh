#!/usr/bin/env bash
#SBATCH --job-name=SR_sentinel2
#SBATCH --output=SR_sentinel2_%j.out
#SBATCH --error=SR_sentinel2_%j.err
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1

set -euo pipefail

module load miniforge3 gcc cuda
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# INPUT_TIF and OUTPUT_DIR come from sbatch --export
: "${INPUT_TIF:?Must be set by sbatch --export INPUT_TIF=/path/to/input.tif}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Show GPU and env info (useful for debugging)
nvidia-smi || true
python --version
python -m torch.utils.collect_env

# Run the SR job on local files only (no internet needed)
python -u run_sr.py \
  --input-tif "${INPUT_TIF}" \
  --output-dir "${OUTPUT_DIR}" \
  --factor 4 \
  --window-size 128 128 \
  --overlap 12 \
  --eliminate-border-px 2 \
  --gpus 0 \
  --save-preview \
  | tee "${LOG_DIR}/job_${SLURM_JOB_ID}.log"

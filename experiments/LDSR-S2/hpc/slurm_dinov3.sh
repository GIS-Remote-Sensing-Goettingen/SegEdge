#!/bin/bash
#SBATCH --job-name=SR_sentinel2
#SBATCH --output=SR_sentinel2.out
#SBATCH --error=SR_sentinel2.err
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1

set -euo pipefail

module load miniforge3 gcc cuda
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOG_PATH="${LOG_PATH:-logs/job_${SLURM_JOB_ID}.log}"
mkdir -p "${OUTPUT_DIR}" "$(dirname -- "${LOG_PATH}")"

python -u LDSR-S2_hpc.py --output-dir "${OUTPUT_DIR}" > "${LOG_PATH}" 2>&1

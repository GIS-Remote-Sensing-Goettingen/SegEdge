#!/bin/bash

#SBATCH --job-name=SR_sentinel2 # Job name
#SBATCH --output=SR_sentinel2.out # Stdout
#SBATCH --error=SR_sentinel2.err  # Stderr
#SBATCH --mem=128G

#SBATCH -t 01:00:00                  # Estimated time, adapt to your needs

#SBATCH --partition=scc-gpu        # Partition (queue) name
#SBATCH --gres=gpu:1

module load miniforge3
module load gcc
module load cuda

# Activate the project environment (override with SEGEDGE_CONDA_ENV if needed).
if [[ -n "${SEGEDGE_CONDA_ENV:-}" ]]; then
  source activate "${SEGEDGE_CONDA_ENV}"
else
  source activate /mnt/vast-standard/home/davide.mattioli/u20330/all
fi

nvidia-smi

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Default to ./outputs if OUTPUT_DIR not set
  cxvaOUTPUT_DIR=${OUTPUT_DIR:-outputs}
LOG_PATH=${LOG_PATH:-logs/job_${SLURM_JOB_ID}.log}

# Create directories safely
mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_PATH}")"

# Run your Python script
python -u LDSR-S2_hpc.py \
  --output-dir "${OUTPUT_DIR}" \
  > "${LOG_PATH}" 2>&1

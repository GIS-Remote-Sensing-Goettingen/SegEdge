#!/bin/bash

#SBATCH --job-name=dinov3_tree_crowns  # Job name
#SBATCH --output=dinov3_tree_crowns.out # Stdout
#SBATCH --error=dinov3_tree_crowns.err  # Stderr

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

export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

INPUT_PATH=${INPUT_PATH:-data/samples/imagery/dinov3_smoll.tif}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/outputs/dinov3/hpc}
LOG_PATH=${LOG_PATH:-artifacts/logs/dinov3/hpc_run.log}
BAND_INDICES=${BAND_INDICES:-1,2,3}
DEVICE=${DEVICE:-cuda}

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_PATH}")"

python -u -m sege.pipelines.dinov3_tree_crowns \
  --input "${INPUT_PATH}" \
  --bands "${BAND_INDICES}" \
  --mode unsup \
  --device "${DEVICE}" \
  --log-file "${LOG_PATH}" \
  --out_dir "${OUTPUT_DIR}"

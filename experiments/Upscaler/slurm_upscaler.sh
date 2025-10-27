#!/bin/bash

#SBATCH --job-name=upscaler_dinov3-sat
#SBATCH --output=upscaler_dinov3-sat.out
#SBATCH --error=upscaler_dinov3-sat.err
#SBATCH --mem=128G
#SBATCH -t 01:00:00                  # Estimated time, adapt to your needs

#SBATCH --partition=scc-gpu        # Partition (queue) name
#SBATCH --gres=gpu:1

module load miniforge3
module load gcc
module load cuda

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



python -u upscaler_dinov3-sat.py

#!/usr/bin/env bash
#SBATCH --job-name=segmentation
#SBATCH --output=segmentation_%j.out
#SBATCH --error=segmentation_%j.err
#SBATCH --mem=128G
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

module load miniforge3 gcc cuda
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HUB_OFFLINE=1

# Show GPU and env info (useful for debugging)
nvidia-smi || true
python --version
python -m torch.utils.collect_env

# Run the SR job within the per-patch workspace
python -u ./main.py

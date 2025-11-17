#!/usr/bin/env bash
#SBATCH --job-name=SR_sentinel2
#SBATCH --output=SR_sentinel2_%j.out
#SBATCH --error=SR_sentinel2_%j.err
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=scc-gpu

set -euo pipefail

module load miniforge3
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"
python --version


# Run the SR job within the per-patch workspace
python -u "./coregister_with_arosics.py"
#!/usr/bin/env bash
#SBATCH --job-name=Coregister
#SBATCH --output=cor.out
#SBATCH --error=cor.err
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --partition=scc-gpu

set -euo pipefail

module load miniforge3
# Activate env (allow override)
source activate "${SEGEDGE_CONDA_ENV:-/mnt/vast-standard/home/davide.mattioli/u20330/all}"
python --version


# Run the SR job within the per-patch workspace
python -u "./coregister_with_arosics.py"
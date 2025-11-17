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

# Explicitly set PROJ data directory
export PROJ_DATA=/mnt/vast-standard/home/davide.mattioli/u20330/all/share/proj

# Verify it's set
echo "PROJ_DATA is set to: $PROJ_DATA"
ls -la $PROJ_DATA/proj.db

python coregister_with_arosics.py
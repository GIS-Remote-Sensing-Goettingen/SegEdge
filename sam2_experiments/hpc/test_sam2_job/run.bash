#!/bin/bash

#SBATCH --job-name=segmentation_job  # Job name
#SBATCH --output=segmentation_job.out # Output file
#SBATCH --error=segmentation_job.err  # Error file

#SBATCH -t 00:05:00                  # Estimated time, adapt to your needs
#SBATCH --mail-type=all              # Send mail when job begins and ends

#SBATCH -p scc-gpu            # The partition

module load miniforge3
module load gcc
module load cuda
source activate

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python -u main.py
#!/bin/bash

#SBATCH --job-name=sam2_mask_tiling
#SBATCH --output=sam2_mask_tiling.out
#SBATCH --error=sam2_mask_tiling.err

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

cd "${HOME}/SegEdge"


CHECKPOINT=${CHECKPOINT:-artifacts/checkpoints/sam2/models/sam2_hiera_large.pt}
MODEL_CONFIG=/mnt/vast-standard/home/davide.mattioli/u20330/segment-anything-2/sam2/configs/sam2/sam2_hiera_l.yaml
IMAGE_PATH=${IMAGE_PATH:-data/samples/imagery/1084-1393.tif}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/outputs/sam2/hpc}
PATCH_SIZE=${PATCH_SIZE:-2048}
OVERLAP=${OVERLAP:-128}

mkdir -p "${OUTPUT_DIR}"

python -u experiments/sam2/hpc/tiling_mask_generator.py \
  --image "${IMAGE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --model-config "${MODEL_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --patch-size "${PATCH_SIZE}" \
  --overlap "${OVERLAP}"

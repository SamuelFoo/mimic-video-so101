#!/bin/bash
#
# SLURM batch script for training an ACT policy with lerobot.

#SBATCH --job-name=train-act-single-arm
#SBATCH --output=/cluster/scratch/samfoo/slurm_runs/slurm-train-act-%j.out
#SBATCH --error=/cluster/scratch/samfoo/slurm_runs/slurm-train-act-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=200G
#SBATCH --time=24:00:00

set -euo pipefail

export CONDA_ENV="${CONDA_ENV:-lerobot}"
export OUTPUT_DIR="${OUTPUT_DIR:-/cluster/work/rsl/samfoo/robot_learning/lerobot_runs/act_single_arm_$(date +%Y%m%d_%H%M%S)}"
export DATASET_SRC="/cluster/home/samfoo/workspaces/robot_learning_project/data/20260428/record-test-merged"
export DATASET_DST="${TMPDIR}/record-test-merged"

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate "${CONDA_ENV}"

# Prepend the conda env's lib dir to LD_LIBRARY_PATH so it takes precedence
# For torchcodec
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export HF_USER=$(hf auth whoami | grep "user:" | awk '{print $2}')

if [[ -z "${HF_USER:-}" ]]; then
    echo "HF_USER is not set" >&2
    exit 1
fi

echo "=== ACT Training Job ==="
echo "Node: $(hostname)"
echo "HF_USER: ${HF_USER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Copying dataset to ${DATASET_DST} ..."
rsync -a "${DATASET_SRC}/" "${DATASET_DST}/"
echo "Dataset copy done."
echo ""

lerobot-train \
  --dataset.repo_id="${HF_USER}/record-test-merged" \
  --dataset.root="${DATASET_DST}" \
  --policy.type=act \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_single_arm \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id="${HF_USER}/act_policy" \
  --batch_size=16 \

echo "=== Training Complete ==="

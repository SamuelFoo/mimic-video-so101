#!/bin/bash
#
# Standalone training script for an ACT policy with lerobot.
# Can also be invoked by slurm_scripts/train_act.sbatch.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/act_single_arm_$(date +%Y%m%d_%H%M%S)}"
DATASET_SRC="${DATASET_SRC:-/cluster/home/samfoo/workspaces/robot_learning_project/data/20260428/record-test-merged}"
DATASET_DST="${DATASET_DST:-${TMPDIR:-/tmp}/record-test-merged}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"
source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export HF_USER=$(hf auth whoami | grep "user:" | awk '{print $2}')

if [[ -z "${HF_USER:-}" ]]; then
    echo "HF_USER is not set" >&2
    exit 1
fi

echo "=== ACT Training Job ==="
echo "Node:       $(hostname)"
echo "HF_USER:    ${HF_USER}"
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

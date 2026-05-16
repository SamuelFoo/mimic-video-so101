#!/bin/bash
#
# SLURM wrapper for scripts/train_act_overfit_student_cluster.sh.

#SBATCH --job-name=train-act-single-arm
#SBATCH --output=slurm-train-act-%j.out
#SBATCH --error=slurm-train-act-%j.err
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --tmp=200G
#SBATCH --time=24:00:00

set -euo pipefail

module load "${CUDA_MODULE:-cuda/12.6}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec "${REPO_ROOT}/scripts/train_act_overfit_student_cluster.sh"

#!/bin/bash
#
# SLURM wrapper for scripts/train_act.sh.

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

module load eth_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../scripts/train_act.sh"

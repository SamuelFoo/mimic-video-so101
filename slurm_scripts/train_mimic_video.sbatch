#!/bin/bash
#
# SLURM wrapper for scripts/train_mimic_video.sh.
# Loads cluster modules and translates the SLURM allocation into the env
# vars the standalone script expects (NUM_GPUS). Single-node only.
#
# To scale up: increase --gpus-per-node, then override EXPERIMENT to a bsz
# that matches the GPU count.

#SBATCH --job-name=train-mimic-video
#SBATCH --output=/cluster/scratch/samfoo/slurm_runs/slurm-mimic-video-%j.out
#SBATCH --error=/cluster/scratch/samfoo/slurm_runs/slurm-mimic-video-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_4090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=400G
#SBATCH --time=24:00:00

set -euo pipefail

module load stack/2024-06
module load cuda/12.4.1
module load eth_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Translate the SLURM allocation into the generic vars the standalone
# script reads.
export NUM_GPUS="$(echo "${SLURM_GPUS_PER_NODE:-1}" | tr -dc '0-9')"

"${SCRIPT_DIR}/../scripts/train_mimic_video.sh"

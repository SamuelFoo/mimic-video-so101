#!/bin/bash
#
# SLURM wrapper for scripts/train_mimic_video.sh.
# Loads cluster modules, resolves the multi-node rendezvous endpoint, then
# calls the standalone script via srun (one task per node). Each node stages
# the dataset and launches its torchrun workers independently.
#
# To scale up: increase --nodes and --gpus-per-node, then override EXPERIMENT
# to a bsz that matches total GPU count.

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

# Derive distributed config from SLURM environment and export so the
# standalone script sees them.
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE="$(echo "${SLURM_GPUS_PER_NODE:-1}" | tr -dc '0-9')"
export MASTER_PORT="${MASTER_PORT:-12341}"

# Resolve the head-node IP for the torchrun rendezvous endpoint.
nodes_array=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes_array[0]}"
export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)"
echo "Rendezvous endpoint: ${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes="${NNODES}" --ntasks-per-node=1 "${SCRIPT_DIR}/../scripts/train_mimic_video.sh"

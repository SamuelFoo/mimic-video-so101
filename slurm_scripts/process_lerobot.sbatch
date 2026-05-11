#!/bin/bash
#
# SLURM wrapper for scripts/process_lerobot.sh.

#SBATCH --job-name=process-lerobot
#SBATCH --output=/cluster/scratch/samfoo/slurm_runs/slurm-process-lerobot-%j.out
#SBATCH --error=/cluster/scratch/samfoo/slurm_runs/slurm-process-lerobot-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=200G
#SBATCH --time=4:00:00

set -euo pipefail

module load stack/2024-06
module load cuda/12.4.1
module load eth_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../scripts/process_lerobot.sh"

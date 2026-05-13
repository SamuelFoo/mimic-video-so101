#!/usr/bin/env bash
#
# One-shot setup for robot_learning_project. Idempotent — safe to re-run.
#
# Runs the env + auth + download steps from README.md:
#   1. uv sync the mimic-video venv
#   2. pin zarr<3 in the lerobot conda env (if present)
#   3. wandb + hf login (interactive, skipped if already authenticated)
#   4. download Cosmos checkpoints
#   5. download ex1_merged-cosmos-video + ex2_merged-cosmos-video datasets
#
# Stops before any pipeline run. After this finishes, activate the venv
# with `source mimic-video/model/.venv/bin/activate` and run
# scripts/process_lerobot.sh / scripts/train_cosmos_video.sh / etc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ---- 1. mimic-video venv ----------------------------------------------------
echo "==> [1/5] uv sync mimic-video venv"
if ! command -v uv >/dev/null; then
    echo "ERROR: uv not found. Install it from https://docs.astral.sh/uv/ first." >&2
    exit 1
fi
(cd mimic-video/model && uv sync --extra cu126)

# ---- 1b. Install Miniconda if conda is not available ------------------------
if ! command -v conda >/dev/null; then
    echo "==> conda not found — installing Miniconda"
    MINICONDA_INSTALLER="$(mktemp -d)/Miniconda3-latest-Linux-x86_64.sh"
    curl -fsSL -O --output-dir "$(dirname "${MINICONDA_INSTALLER}")" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm "${MINICONDA_INSTALLER}"
    # Make conda available in this shell session
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda init bash
    echo "  Miniconda installed at ${HOME}/miniconda3"
fi

# ---- 2. lerobot conda env: zarr<3 pin --------------------------------------
echo "==> [2/5] Pin zarr<3 in lerobot conda env"
if ! command -v conda >/dev/null; then
    echo "  conda not on PATH; skipping. After installing conda, run manually:"
    echo "    conda activate lerobot && pip install 'zarr<3' 'numcodecs<0.16'"
elif ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx lerobot; then
    echo "  conda env 'lerobot' does not exist; skipping."
    echo "  Create it via the lerobot HF setup guide, then re-run this script."
else
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lerobot
    pip install 'zarr<3' 'numcodecs<0.16'
    conda deactivate
fi

# ---- 3. Authenticate wandb + hf (interactive, idempotent) -------------------
echo "==> [3/5] wandb + hf auth"
# shellcheck source=/dev/null
source mimic-video/model/.venv/bin/activate

if grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  wandb already authenticated."
else
    wandb login
fi

if hf auth whoami 2>/dev/null | grep -q "Not logged in"; then
    hf auth login
else
    echo "  hf already authenticated."
fi

# ---- 4. Cosmos checkpoints --------------------------------------------------
echo "==> [4/5] Download Cosmos checkpoints"
(cd mimic-video/model && python scripts/download_checkpoints.py)

# ---- 5. Datasets ------------------------------------------------------------
echo "==> [5/5] Download datasets"
mkdir -p data
for name in ex1_merged-cosmos-video ex2_merged-cosmos-video; do
    target="data/${name}"
    if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
        echo "  ${target} already populated, skipping."
        continue
    fi
    hf download "robot-learning/${name}" \
        --repo-type dataset \
        --local-dir "${target}"
done

echo
echo "Setup complete. To use the mimic-video venv in your shell:"
echo "  source mimic-video/model/.venv/bin/activate"

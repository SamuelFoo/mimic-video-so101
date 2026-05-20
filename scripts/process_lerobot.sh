#!/bin/bash
#
# Standalone script for converting one or more LeRobot v3 datasets to
# mimic-video's per-episode .zarr layout, then precomputing T5 language
# embeddings. Can also be invoked by slurm_scripts/process_lerobot.sbatch.
#
# Edit DATASET_PAIRS below to choose which datasets to process.
#
# Parallelism:
#   - Phase 1 (zarr conversion) runs all datasets simultaneously (CPU/disk-bound).
#   - Phase 2 (T5 embeddings)  runs up to MAX_T5_JOBS at once (GPU-bound).
#     Override: MAX_T5_JOBS=4 ./process_lerobot.sh
#
# NOTE: The lerobot conda env needs zarr<3 / numcodecs<0.16 installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Datasets to process --------------------------------------------------
# Each entry is "<dataset_name>=<ex_type>". The dataset_name resolves to
# ${DATA_ROOT}/<dataset_name> (input) and ${DATA_ROOT}/<dataset_name>-zarr
# (output). The ex_type selects the language instruction from
# config/language_instructions.json.
DATASET_PAIRS=(
    "ex3-1-blue_all=ex3-1-blue"
    "ex3-1-orange_all=ex3-1-orange"
    "ex3-2-blue_all=ex3-2-blue"
    "ex3-2-orange_all=ex3-2-orange"
)

# Datasets whose zarr already exists — skip Phase 1 but still run T5 (Phase 2).
ZARR_SKIP=(
    "ex3-1-blue_all"
    "ex3-1-orange_all"
    "ex3-2-blue_all"
    "ex3-2-orange_all"
)
# ---------------------------------------------------------------------------

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MAX_T5_JOBS="${MAX_T5_JOBS:-1}"
OVERWRITE_ZARR="${OVERWRITE_ZARR:-}"  # set to any non-empty value to force full reprocess
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"

if (( ${#DATASET_PAIRS[@]} == 0 )); then
    echo "ERROR: DATASET_PAIRS is empty — edit the list at the top of $0" >&2
    exit 2
fi

declare -a DATASETS EX_TYPES IN_DIRS OUT_DIRS
for pair in "${DATASET_PAIRS[@]}"; do
    if [[ "${pair}" != *"="* ]]; then
        echo "ERROR: expected <dataset>=<ex_type>, got '${pair}'" >&2
        exit 2
    fi
    ds="${pair%%=*}"
    DATASETS+=("${ds}")
    EX_TYPES+=("${pair#*=}")
    IN_DIRS+=("${DATA_ROOT}/${ds}")
    OUT_DIRS+=("${DATA_ROOT}/${ds}-zarr")
done

# Resolve language instructions up front using whichever python is on PATH —
# avoids re-activating envs for each dataset.
declare -a LANGUAGE_INSTRUCTIONS
for ex in "${EX_TYPES[@]}"; do
    LANGUAGE_INSTRUCTIONS+=("$(python3 "${REPO_ROOT}/helpers/utils/language_instructions.py" "${ex}" --instructions "${INSTRUCTIONS_JSON}")")
done

# Precompute total episode counts (read once; used for zarr progress reporting).
declare -a TOTAL_EPISODES=()
for i in "${!DATASETS[@]}"; do
    _info="${IN_DIRS[$i]}/meta/info.json"
    if [[ -f "${_info}" ]]; then
        TOTAL_EPISODES+=("$(python3 -c "import json; print(json.load(open('${_info}')).get('total_episodes','?'))")")
    else
        TOTAL_EPISODES+=("?")
    fi
done

# source "${CONDA_BASE}/bin/activate"
# conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${_LOG_DIR}"' EXIT

# T5 env vars — set once here so subshells inherit them without needing
# conda deactivate / source activate in the main shell.
_T5_PYTHON="${MODEL_DIR}/.venv/bin/python"
_T5_CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
_T5_LD="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
_COSMOS_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

declare -a T5_PIDS=()
declare -a T5_IDX=()

# Returns true only if $1 is a running/sleeping process (not zombie, not gone).
# kill -0 succeeds on zombies, so we check /proc state instead.
_is_running() {
    local s
    s=$(ps -p "$1" -o state= 2>/dev/null)
    [[ "$s" == "R" || "$s" == "S" || "$s" == "D" ]]
}

# Reap any completed T5 jobs; print done/fail and shrink T5_PIDS.
_reap_t5() {
    local new_pids=() new_idx=() j ds
    for j in "${!T5_PIDS[@]}"; do
        if _is_running "${T5_PIDS[$j]}"; then
            new_pids+=("${T5_PIDS[$j]}")
            new_idx+=("${T5_IDX[$j]}")
        else
            ds="${DATASETS[${T5_IDX[$j]}]}"
            if wait "${T5_PIDS[$j]}"; then
                echo "  [done] T5 ${ds}"
            else
                echo "  [FAIL] T5 ${ds} — see log below" >&2
                cat "${_LOG_DIR}/t5_${ds}.log" >&2
                exit 1
            fi
        fi
    done
    T5_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    T5_IDX=("${new_idx[@]+"${new_idx[@]}"}")
}

# Returns true if every episode zarr in $1 already has language_embedding.
_t5_done() {
    local out_dir="$1"
    local ep_count emb_count
    ep_count=$(find "${out_dir}" -maxdepth 1 -name "episode_*.zarr" -type d 2>/dev/null | wc -l)
    (( ep_count == 0 )) && return 1
    emb_count=$(find "${out_dir}" -maxdepth 2 -name "language_embedding" -type d 2>/dev/null | wc -l)
    (( emb_count >= ep_count ))
}

# T5 queue: dataset indices waiting for a free slot.
declare -a T5_QUEUE=()

# Enqueue dataset index $1 for T5 (called from the launch loop — never blocks).
_enqueue_t5() {
    local i=$1 ds="${DATASETS[$1]}"
    if _t5_done "${OUT_DIRS[$i]}"; then
        echo "  [skip]  T5 ${ds} (already computed)"
    else
        T5_QUEUE+=("$i")
    fi
}

# Start queued T5 jobs up to MAX_T5_JOBS (non-blocking — called from poll loop).
_drain_t5_queue() {
    while (( ${#T5_QUEUE[@]} > 0 && ${#T5_PIDS[@]} < MAX_T5_JOBS )); do
        local i="${T5_QUEUE[0]}"
        T5_QUEUE=("${T5_QUEUE[@]:1}")
        local ds="${DATASETS[$i]}"
        echo "  [start] T5 ${ds}"
        (
            export PATH="/sbin:/usr/sbin:${PATH}"
            export CUDA_HOME="${_T5_CUDA_HOME}"
            export CUDA_PATH="${_T5_CUDA_HOME}"
            export LD_LIBRARY_PATH="${_T5_LD}:${LD_LIBRARY_PATH:-}"
            export COSMOS_PREDICT2_ARGS="${_COSMOS_ARGS}"
            "${_T5_PYTHON}" mimic-video/data_preprocessing/action/precompute_t5.py \
              --dataset-path "${OUT_DIRS[$i]}" \
              --prompt "${LANGUAGE_INSTRUCTIONS[$i]}"
        ) > "${_LOG_DIR}/t5_${ds}.log" 2>&1 &
        T5_PIDS+=($!)
        T5_IDX+=("$i")
    done
}

echo "=== LeRobot -> zarr + T5 (pipelined, max T5 jobs: ${MAX_T5_JOBS}) ==="
echo "Node:        $(hostname)"
echo "Conda env:   ${CONDA_ENV}"
echo "Datasets:    ${#DATASETS[@]}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo

# Launch zarr conversions; immediately start T5 for datasets with existing zarr.
declare -a ZARR_PIDS=()
declare -a ZARR_IDX=()
for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    _skip=0
    for _s in "${ZARR_SKIP[@]+"${ZARR_SKIP[@]}"}"; do
        [[ "$ds" == "$_s" ]] && _skip=1 && break
    done
    if (( _skip )); then
        echo "  [skip]  zarr ${ds} — queuing T5"
        _enqueue_t5 "$i"
        continue
    fi
    echo "  [start] zarr ${ds}"
    (
        python mimic-video/data_preprocessing/action/process_lerobot.py \
          --repo-id "${ds}" \
          --root "${IN_DIRS[$i]}" \
          --output-dir "${OUT_DIRS[$i]}" \
          --language-instruction "${LANGUAGE_INSTRUCTIONS[$i]}" \
          ${OVERWRITE_ZARR:+--overwrite}
    ) > "${_LOG_DIR}/zarr_${ds}.log" 2>&1 &
    ZARR_PIDS+=($!)
    ZARR_IDX+=("$i")
done

# Poll for zarr completions; drain T5 queue as slots open up.
_ZARR_FAILED=0
while (( ${#ZARR_PIDS[@]} > 0 || ${#T5_QUEUE[@]} > 0 || ${#T5_PIDS[@]} > 0 )); do
    _reap_t5
    # Only start T5 jobs once all zarr conversions are done to avoid RAM contention.
    (( ${#ZARR_PIDS[@]} == 0 )) && _drain_t5_queue
    new_pids=(); new_idx=()
    for j in "${!ZARR_PIDS[@]}"; do
        if _is_running "${ZARR_PIDS[$j]}"; then
            new_pids+=("${ZARR_PIDS[$j]}")
            new_idx+=("${ZARR_IDX[$j]}")
        else
            ds="${DATASETS[${ZARR_IDX[$j]}]}"
            if wait "${ZARR_PIDS[$j]}"; then
                echo "  [done] zarr ${ds} — queuing T5"
                _enqueue_t5 "${ZARR_IDX[$j]}"
            else
                echo "  [FAIL] zarr ${ds} — see log below" >&2
                cat "${_LOG_DIR}/zarr_${ds}.log" >&2
                _ZARR_FAILED=1
            fi
        fi
    done
    ZARR_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    ZARR_IDX=("${new_idx[@]+"${new_idx[@]}"}")
    for j in "${!ZARR_IDX[@]}"; do
        idx="${ZARR_IDX[$j]}"
        _done=$(find "${OUT_DIRS[$idx]}" -maxdepth 1 -name "episode_*.zarr" -type d 2>/dev/null | wc -l)
        (( _done < TOTAL_EPISODES[idx] )) && echo "  [prog]  zarr ${DATASETS[$idx]}: ${_done}/${TOTAL_EPISODES[$idx]}"
    done
    (( ${#ZARR_PIDS[@]} > 0 )) && sleep 10
done
(( _ZARR_FAILED )) && exit 1

echo "=== Conversion + T5 Precompute Complete ==="

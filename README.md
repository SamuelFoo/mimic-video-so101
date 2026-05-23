## Quickstart

### Cloning Repo

Clone repo and submodles.

```bash
git clone --recurse-submodules git@github.com:<project_url>
```

### Python Environments

Install `mimic-video` dependencies. Requires `uv`.

```bash
cd mimic-video/model
uv sync --extra cu126
source .venv/bin/activate
```

The `mimic-video` Python virtual environment can be sourced with

```bash
source mimic-video/model/.venv/bin/activate
```

Create a separate `lerobot` conda environment using the lerobot setup guide on HuggingFace.
Then pin zarr to the v2 line so the generated zarr episodes are readable by
the `mimic-video` environment:

```bash
conda activate lerobot
pip install 'zarr<3' 'numcodecs<0.16'
```

Use the `lerobot` conda environment for preprocessing datasets and the `mimic-video` Python virtual environment for training.

### Auth

In the `mimic-video` virtual environment, login to `wandb` and `hf`:

```bash
source mimic-video/model/.venv/bin/activate
wandb login
hf auth login
```

### Download

Download the checkpoints

```bash
cd mimic-video/model
source .venv/bin/activate
python scripts/download_checkpoints.py
```

Download the required dataset(s) to the `data/` folder, for example:

```bash
source mimic-video/model/.venv/bin/activate
cd ~/robot_learning_project/
hf download robot-learning/Ex1_merged \
  --repo-type dataset \
  --local-dir data/ex1_merged
```

Merge the dataset(s) into a single dataset (see `commands.md`).

### Video Model Inference

Run Video2World inference over the merged LeRobot dataset in three steps. All commands are run from the repo root.

**1. Convert LeRobot to zarr and precompute T5 embeddings.** Uses the `lerobot` conda environment for the conversion and `mimic-video/model/.venv` for the embeddings. For proces_lerobot.sh, make sure that only the lerobot conda environment is on.

```bash
./scripts/process_lerobot.sh
```

**2. Export zarr episodes to MP4 inputs and build the Video2World batch JSON.** Each run is written to a timestamped directory under `runs/video_inference/${DATASET_NAME}_<TIMESTAMP>/`, so successive runs do not overwrite each other. Optionally set `FRAME_FRACTION=0.5` to export only the first half of each episode.

```bash
./scripts/export_video_inputs.sh
```

**3. Run Video2World inference.** Uses `mimic-video/model/.venv`. Before running, set `RUN_DIR` (or `TIMESTAMP`) to point at the export you want to run inference on.

```bash
RUN_DIR=runs/video_inference/ex1_merged_2026-05-13_12-15-23 ./scripts/infer_video.sh
```

### Video Model Finetuning (ex1_merged)

End-to-end pipeline for finetuning the Cosmos-Predict2 2B video backbone on the merged LeRobot dataset. Uses LoRA (rank 256) by default. All commands run from the repo root.

**1. Convert LeRobot to zarr.** Same step as inference — skip if already done.

```bash
./scripts/process_lerobot.sh
```

**2. Build the Cosmos finetuning data layout.** Converts each `episode_*.zarr` into the `video/<ep>.mp4` + `metas/<ep>.txt` layout expected by `cosmos_predict2.data.dataset_video.Dataset`, then precomputes T5 embeddings into `t5_xxl/`. Writes to `data/${DATASET_NAME}-cosmos-video/`.

```bash
./scripts/prepare_video_finetune_data.sh
```

To re-run preprocessing after a config change: `OVERWRITE=true ./scripts/prepare_video_finetune_data.sh`. To skip T5 (e.g. iterating on the video extraction): `SKIP_T5=true ./scripts/prepare_video_finetune_data.sh`.

**3. Verify the dataset entry.** [data_video.py](mimic-video/model/cosmos_predict2/configs/defaults/data_video.py) registers `ex1_merged` with `dataset_dir=data/ex1_merged-cosmos-video`. The grid in [video2world.py](mimic-video/model/cosmos_predict2/configs/experiment/video2world.py) auto-creates the experiment `v2w_ex1_merged_lora_rank256_lr1.778e-04_bsz32`. If you change `DATASET_NAME` or `FINETUNE_DATA_DIR`, update the entry in `data_video.py` to match.

**4. Run training.** Defaults to the `ex1_merged` experiment with WandB logging enabled. Each run gets a timestamped output dir at `runs/cosmos_video/<EXPERIMENT>_<TIMESTAMP>/`.

```bash
./scripts/train_cosmos_video.sh
```

Useful env-var overrides:

- `EXPERIMENT=v2w_ex1_merged_lora_rank256_lr1.778e-04_bsz32`
- `MAX_ITER=20000` — max training iterations
- `WANDB_PROJECT=cosmos-video-finetune`, `WANDB_ENTITY=...`, `WANDB_MODE=offline`
- `VIDEO_DIT_PATH=...` — start from a different video checkpoint
- `GPUS_PER_NODE`, `NNODES`, `MASTER_PORT` for distributed training

**Gotchas**

- _Num_frames must be 1, 5, or 61._ Set in [data_video.py](mimic-video/model/cosmos_predict2/configs/defaults/data_video.py). Use 5 to speed up finetuning iteration.
- _Batch size cannot exceed the number of generated videos._ If `bsz` (set in [video2world.py](mimic-video/model/cosmos_predict2/configs/experiment/video2world.py)) is larger than the dataset size, training errors out — pick a smaller experiment or generate more videos.

### Mimic-Video Policy Training

Trains the action decoder on top of a frozen video backbone. The decoder cross-attends to layer-20 hidden states from a frozen Cosmos-Predict2 2B DiT and predicts low-dim actions. All commands run from the repo root.

**1. Convert LeRobot to zarr.** Same step as inference/finetuning — skip if already done.

```bash
./scripts/process_lerobot.sh
```

**2. Place a video backbone checkpoint.** The default experiment expects `mimic-video/model/checkpoints/video_backbone/iter_000000375_fused.pt`. Symlink or copy your fused video DiT there, or set `VIDEO_DIT_PATH=/abs/path.pt` when invoking the scripts below. The pretrained Cosmos checkpoint (`v2w_pretrained_cosmos.pt`) and the registered LoRA-fused variants in [world2action_model.py](mimic-video/model/cosmos_predict2/configs/defaults/world2action_model.py) also work — just pick a matching `EXPERIMENT`.

**3. Precompute VAE latents (one-time, recommended).** Encodes every train + val sample's 61-frame raw video to `[16, 16, 60, 80]` latents and stores them under `${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/` (default: `data/.latent_cache/`). The training step then loads latents via mmap and skips the per-step VAE encoder forward — the dominant memory hot-spot and a ~2-3× wall-clock saving. Cache is keyed by dataset config hash, so it auto-invalidates when the episode set or transforms change.

```bash
./scripts/precompute_video_latents.sh
```

With the default LeRobot config, samples are anchored at 5 Hz to match the author's effective video rate while avoiding one near-duplicate sample per raw 30 Hz camera frame.

**4. Run training.** Default experiment is `w2a_lerobot_iter_000000375_fused_lr1.000e-04_layer20_bsz128`, trained on the combined `ex1_merged + ex2_merged` zarr dirs.

```bash
./scripts/train_mimic_video.sh
```

Useful env-var overrides:

- `EXPERIMENT=w2a_lerobot_<video_ckpt>_lr<...>_layer20_bsz<...>` — pick a different registered experiment (see [world2action.py](mimic-video/model/cosmos_predict2/configs/experiment/world2action.py))
- `VIDEO_DIT_PATH=/abs/path.pt` — override the auto-resolved video backbone
- `MIMIC_VIDEO_DATASET_DIR=/path/to/dir` — dir to glob `**/*.zarr` under (default `${REPO_ROOT}/data`). Both `.statistics_cache/` and `.latent_cache/` land inside it.
- `TRAIN_LOCAL_BATCH_SIZE=16`, `GRAD_ACCUM_ITER=8` — per-GPU batch and accumulation; effective batch is product × world_size
- `ACTION_MODEL_CHANNELS`, `ACTION_MODEL_BLOCKS`, `ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK`, `ACTION_MODEL_ADALN_LORA_DIM` — shrink the action decoder for tighter GPU budgets (the defaults are ~10× smaller than the author's reference config; see comments in [train_mimic_video.sh](scripts/train_mimic_video.sh))
- `MAX_VAL_ITER=8` — cap the iter-0 validation pass; set to `null` for full validation
- `WANDB_PROJECT=mimic-video`, `WANDB_ENTITY=...`, `WANDB_MODE=offline`
- `GPUS_PER_NODE`, `NNODES`, `MASTER_PORT` for distributed training

Runs land in `runs/mimic_video/<EXPERIMENT>_<TIMESTAMP>/`.

### Video Model Finetuning — Cosmos-Predict2.5 (Video2World v2.5)

End-to-end pipeline for finetuning the Cosmos-Predict2.5 2B video backbone with Reason1 embeddings.
Uses LoRA and pre-projected conditioning embeddings (no Reason1-7B at training time).
All commands run from the repo root.

**Environment:** `cosmos-predict2.5/.venv` (Python 3.10, set up by `./setup.sh`).

---

**1. Convert raw recordings to zarr.** Same as inference — skip if already done.

```bash
./scripts/process_lerobot.sh
```

---

**2. Build the cosmos-video data layout** (`video/` + `metas/`).

```bash
EX_TYPE=ex1 SKIP_T5=true ./scripts/prepare_video_finetune_data.sh
```

`SKIP_T5=true` skips T5 embedding generation — we use Reason1 embeddings instead (next step).
The script writes `staging/mimic-video/<DATASET_NAME>-cosmos-video/video/` and `metas/`.

Repeat for each experiment type (`EX_TYPE=ex2`, `EX_TYPE=ex3`, …).

---

**3. Pre-compute Reason1 + crossattn\_proj embeddings** (`reason1_proj/`).

Runs Reason1-7B (Qwen2.5-VL-7B FULL\_CONCAT, 100 352-dim) and the frozen `crossattn_proj`
linear from the 2B checkpoint offline, storing `[n_tokens, 1024]` float16 pickles.
This removes ~200 MB of weights and ~98 MB of per-step conditioning tensors from GPU memory.

```bash
source cosmos-predict2.5/.venv/bin/activate
python scripts/precompute_reason1_embeddings.py \
    --dataset_dirs \
        staging/mimic-video/ex1_all_v4-cosmos-video \
        staging/mimic-video/ex2_all_v4-cosmos-video \
        staging/mimic-video/ex3-1-blue_all-cosmos-video \
        staging/mimic-video/ex3-2-blue_all-cosmos-video \
        staging/mimic-video/ex3-1-orange_all-cosmos-video \
        staging/mimic-video/ex3-2-orange_all-cosmos-video \
    --predict2_checkpoint ~/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt \
    --batch_size 4
```

The script deduplicates captions, so datasets where all episodes share one caption (common
for task-specific splits) complete in seconds. Output: `<dataset_dir>/reason1_proj/*.pickle`.

---

**4. Run finetuning.**

Experiment names follow `predict2_v2w_lora_rank<R>_<DATASET_NAME>` and are registered in
[`cosmos_predict2/experiments/base/mimic_video.py`](cosmos-predict2.5/cosmos_predict2/experiments/base/mimic_video.py).

```bash
DATASET_NAME=ex1_all_v4 LORA_RANK=32 ./scripts/train_cosmos_video_v25.sh
```

Useful env-var overrides:

- `DATASET_NAME` — key from `_DATASET_CFGS` in `mimic_video.py` (e.g. `ex1_all_v4`, `ex1_ex2_ex3_merged`)
- `LORA_RANK` — 32, 64, or 128
- `MAX_ITER=50000` — stop earlier for iteration testing
- `TRAIN_LOCAL_BATCH_SIZE=1` — per-GPU batch size (effective = × GPUs × grad_accum)
- `GRAD_ACCUM_ITER=4` — gradient accumulation steps
- `SAVE_ITER=500` — checkpoint cadence
- `LOAD_PATH=/path/to/checkpoint/iter_<iter_number>` — to pass a custom .distcp model weights
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE=offline`
- `GPUS_PER_NODE`, `NNODES`, `MASTER_PORT` for distributed training

Checkpoints land in `runs/cosmos_video_v25/<EXPERIMENT>_<TIMESTAMP>/`.

## If Triton Cache crashes
# 1. Clear the Triton inductor cache (stale compiled kernels)
rm -rf /tmp/torchinductor_shadeform/

# 2. Resume from the saved checkpoint
LOAD_PATH=/ephemeral/robot_learning_project/runs/cosmos_video_v25/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_2026-05-23_11-20-06/.../iter_000000700 \
bash scripts/train_cosmos_video_v25.sh


**Key design decisions:**

- `text_encoder_config=None` — Reason1-7B is not loaded at training time; embeddings are read from disk
- `use_crossattn_projection=False` — the `crossattn_proj` linear is not part of the training graph; its output is already stored in `reason1_proj/`
- `embedding_subdir="reason1_proj"` — `MimicVideoDataset` reads from `reason1_proj/` instead of `t5_xxl/`
- LoRA targets: `q_proj, k_proj, v_proj, output_proj, mlp.layer1, mlp.layer2`
- 5 conditioning frames (`obs_history=5`, `min/max_num_conditional_frames=5`)

---

### Mimic-Video Inference

```bash
ssh -L 8000:localhost:8000 infer-2

python deployment/run_so101_inference.py \
  --port /dev/ttyACM0 \
  --robot-id my_awesome_follower_arm \
  --server http://localhost:8000 \
  --prompt-key ex1 \
  --max-relative-target 0 \
  --camera-index 2 \
  --stop-after-step 0 \
  --meshcat
```

Prompts are loaded from [config/deployment_prompts.json](config/deployment_prompts.json); select one with `--prompt-key` or override with `--prompt "..."`.

A live camera window opens on the laptop by default — press **`r`** in that window to start/stop an MP4 recording (written to `recordings/so101_<timestamp>.mp4`).

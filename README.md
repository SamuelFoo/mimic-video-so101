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

- *Num_frames must be 1, 5, or 61.* Set in [data_video.py](mimic-video/model/cosmos_predict2/configs/defaults/data_video.py). Use 5 to speed up finetuning iteration.
- *Batch size cannot exceed the number of generated videos.* If `bsz` (set in [video2world.py](mimic-video/model/cosmos_predict2/configs/experiment/video2world.py)) is larger than the dataset size, training errors out — pick a smaller experiment or generate more videos.

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

# mimic-video Deployed on the SO-101

Please see the [original mimic-video project](https://mimic-video.github.io/).

Completed as part of ETH Zurich's [robot learning course](https://cvg.ethz.ch/lectures/Robot-Learning/).

The deployed weights are available on
[robot-learning/pushing_model](https://huggingface.co/robot-learning/pushing_model), and the
dataset is available at
[robot-learning/mimic-video-tar](https://huggingface.co/datasets/robot-learning/mimic-video-tar).

https://github.com/user-attachments/assets/56233521-9789-4c53-8d19-83b7cf8fe523

## Demos

The prompts below are the deployment prompts defined in
[config/deployment_prompts.json](config/deployment_prompts.json).

### White Ball With Obstacle (`ex2`)

<details><summary>Input prompt</summary>
A black robotic arm is shown in a clean, modern indoor setting, positioned behind a dark green mat on a tabletop. The base of the robotic arm is fixed to the tabletop. On the mat, a white ball is positioned at the start location on the right side, while the goal region is marked by the smaller white circle on the left side. Two white horizontal lines define a straight corridor. Both the white ball and black gripper are rigid and non-deformable. A red static obstacle is placed in the bigger circle near the center of the corridor. The black robot gripper guides the white ball from right to left toward the goal circle. The white ball moves around the obstacle along a path that curves toward the front. The white ball leaves the corridor when going around the obstacle. In general, neither the white ball nor the black gripper touches or moves the red obstacle and the red obstacle remains stationary. The white ball may occasionally touch the red obstacle, and the black gripper responds by making controlled corrective adjustments to guide the white ball around the obstacle and toward the goal. The final frame shows the white ball inside the left goal circle, while the obstacle remains in its original position.
</details>

<video src="demos/ex2.mp4" controls width="478"></video>

### Blue Cube (`ex3-1-blue`)

<details><summary>Input prompt</summary>
A black robotic arm is shown in a clean, modern indoor setting, positioned behind a dark green mat on a tabletop. The base of the robotic arm is fixed to the tabletop. On the mat, a blue cube is positioned at the start location on the right side, while the goal region is marked by the smaller white circle on the left side. Two white horizontal lines define a straight corridor. Both the blue cube and black gripper are rigid and non-deformable. The black robot gripper guides the blue cube from right to left toward the goal circle. The blue cube is generally guided within the corridor. The blue cube may occasionally drift outside the corridor after contact, and the black gripper responds with controlled corrective adjustments to guide the blue cube back onto the intended path toward the goal circle. The final frame shows the blue cube inside the left goal circle.
</details>

<video src="demos/ex3-1.mp4" controls width="478"></video>

### Blue Cube With Obstacle (`ex3-2-blue`)

<details><summary>Input prompt</summary>
A black robotic arm is shown in a clean, modern indoor setting, positioned behind a dark green mat on a tabletop. The base of the robotic arm is fixed to the tabletop. On the mat, a blue cube is positioned at the start location on the right side, while the goal region is marked by the smaller white circle on the left side. Two white horizontal lines define a straight corridor. Both the blue cube and black gripper are rigid and non-deformable. A red static obstacle is placed in the bigger circle near the center of the corridor. The black robot gripper guides the blue cube from right to left toward the goal circle. The blue cube moves around the obstacle along a path that curves toward the front. The blue cube leaves the corridor when going around the obstacle. In general, neither the blue cube nor the black gripper touches or moves the red obstacle and the red obstacle remains stationary. The blue cube may occasionally touch the red obstacle, and the black gripper responds by making controlled corrective adjustments to guide the blue cube around the obstacle and toward the goal. The final frame shows the blue cube inside the left goal circle, while the obstacle remains in its original position.
</details>

<video src="demos/ex3-2.mp4" controls width="478"></video>

## Quickstart

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
cd ~/mimic-video-so101/
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

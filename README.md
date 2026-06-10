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

<video src="https://github.com/user-attachments/assets/e8128459-3926-4116-89d4-f35b3907bef7" controls width="478"></video>

### Blue Cube (`ex3-1-blue`)

<details><summary>Input prompt</summary>
A black robotic arm is shown in a clean, modern indoor setting, positioned behind a dark green mat on a tabletop. The base of the robotic arm is fixed to the tabletop. On the mat, a blue cube is positioned at the start location on the right side, while the goal region is marked by the smaller white circle on the left side. Two white horizontal lines define a straight corridor. Both the blue cube and black gripper are rigid and non-deformable. The black robot gripper guides the blue cube from right to left toward the goal circle. The blue cube is generally guided within the corridor. The blue cube may occasionally drift outside the corridor after contact, and the black gripper responds with controlled corrective adjustments to guide the blue cube back onto the intended path toward the goal circle. The final frame shows the blue cube inside the left goal circle.
</details>

<video src="https://github.com/user-attachments/assets/e41f531a-ceb7-4ee0-960b-b4199e369b2f" controls width="478"></video>

### Orange Cube With Obstacle (`ex3-2-orange`)

<details><summary>Input prompt</summary>
A black robotic arm is shown in a clean, modern indoor setting, positioned behind a dark green mat on a tabletop. The base of the robotic arm is fixed to the tabletop. On the mat, a orange cube is positioned at the start location on the right side, while the goal region is marked by the smaller white circle on the left side. Two white horizontal lines define a straight corridor. Both the orange cube and black gripper are rigid and non-deformable. A red static obstacle is placed in the bigger circle near the center of the corridor. The black robot gripper guides the orange cube from right to left toward the goal circle. The orange cube moves around the obstacle along a path that curves toward the front. The orange cube leaves the corridor when going around the obstacle. In general, neither the orange cube nor the black gripper touches or moves the red obstacle and the red obstacle remains stationary. The orange cube may occasionally touch the red obstacle, and the black gripper responds by making controlled corrective adjustments to guide the orange cube around the obstacle and toward the goal. The final frame shows the orange cube inside the left goal circle, while the obstacle remains in its original position.
</details>

<video src="https://github.com/user-attachments/assets/c17d0562-9366-47ba-8ff9-17e9f67e68e4" controls width="478"></video>

## Quickstart

For automated setup, run the helper from the repository root:

```bash
./setup.sh
```

Alternatively, follow the manual setup instructions below.

### Python Environments

The Cosmos and mimic-video code requires Linux, Python 3.10, an NVIDIA GPU,
and CUDA 12.6-compatible drivers. Install the model environment with `uv`:

```bash
cd mimic-video/model
uv sync --extra cu126
uv pip install -r ../../deployment/requirements.txt
cd ../..
```

Activate it from the repository root with:

```bash
source mimic-video/model/.venv/bin/activate
```

Use a separate environment for LeRobot preprocessing and the robot client:

```bash
conda create -n lerobot python=3.12
conda activate lerobot
conda install -c conda-forge ffmpeg pinocchio
pip install lerobot 'zarr<3' 'numcodecs<0.16' opencv-python meshcat
```

The `zarr<3` and `numcodecs<0.16` pins are required by the current
`mimic-video` preprocessing pipeline.

### Auth

Authenticate with Hugging Face and Weights & Biases in the model environment:

```bash
source mimic-video/model/.venv/bin/activate
hf auth login
wandb login
```

### Checkpoints And Data

Download the upstream Cosmos tokenizer, text encoder, and pretrained
Video2World backbone:

```bash
cd mimic-video/model
source .venv/bin/activate
python scripts/download_checkpoints.py --models pretrained_cosmos_bridge
cd ../..
```

The deployed SO-101 video model, action model, action configuration, and
normalization statistics are published together. Download them from the
repository root to preserve the expected `checkpoints/` layout:

```bash
hf download robot-learning/pushing_model --local-dir .
```

The training data is available as archives:

```bash
hf download robot-learning/mimic-video-tar \
  --repo-type dataset \
  --local-dir downloads/mimic-video-tar
```

After extraction, place raw LeRobot datasets under `data/` using the names in
`DATASET_PAIRS` at the top of [process_lerobot.sh](scripts/process_lerobot.sh).
For action-model training, place the resulting `*-zarr` directories anywhere
under `staging/mimic-video/`; the action dataloader searches that directory
recursively.

### Video Model Inference

Run Video2World inference over a processed LeRobot dataset from the repository
root.

**1. Convert LeRobot datasets to zarr and precompute T5 embeddings.**
`process_lerobot.sh` processes the datasets listed in its `DATASET_PAIRS`
array. Run it from the `lerobot` environment:

```bash
conda activate lerobot
./scripts/process_lerobot.sh
```

**2. Export zarr episodes and create the inference batch.** The default
`FRAME_FRACTION=0.5` exports the first half of each episode; use
`FRAME_FRACTION=1` for complete episodes.

```bash
EX_TYPE=ex1 DATASET_NAME=ex1_all_v4 ./scripts/export_video_inputs.sh
```

The export command prints the timestamped `RUN_DIR`.

**3. Run Video2World inference.** Set `RUN_DIR` to the directory created above
and `VIDEO_DIT_PATH` to the desired video checkpoint:

```bash
RUN_DIR=runs/video_inference/ex1_all_v4_<TIMESTAMP> \
VIDEO_DIT_PATH="$(pwd)/checkpoints/video/<video-checkpoint>.pt" \
./scripts/infer_video.sh
```

Set `EPISODE_STRIDE=1` to infer every exported episode; the default is every
fifth episode.

### Video Model Finetuning

The current SO-101 Video2World configuration uses 21-frame clips, five
conditioning frames, and six VAE latent timesteps. See
[docs/parameters.md](docs/parameters.md) before changing temporal parameters.

**1. Prepare zarr datasets.** Run `process_lerobot.sh` as described above.
The current configuration produces:

- `ex1_all_v4-zarr`
- `ex2_all_v4-zarr`
- `ex3-1-blue_all-zarr`
- `ex3-1-orange_all-zarr`
- `ex3-2-blue_all-zarr`
- `ex3-2-orange_all-zarr`

**2. Build the Cosmos finetuning layout.** Run the preparation script for each
dataset to create `video/`, `metas/`, and `t5_xxl/` directories. For example:

```bash
EX_TYPE=ex1 DATASET_NAME=ex1_all_v4 ./scripts/prepare_video_finetune_data.sh
EX_TYPE=ex2 DATASET_NAME=ex2_all_v4 ./scripts/prepare_video_finetune_data.sh
```

Prepare the four Exercise 3 datasets in the same way, then merge them into
`ex3_all-cosmos-video`. The merge scripts contain cluster-specific paths and
must be adjusted for the local data root before use.

Use `OVERWRITE=true` to rebuild existing outputs or `SKIP_T5=true` while
iterating only on video extraction.

**3. Verify the dataset entries.**
[data_video.py](mimic-video/model/cosmos_predict2/configs/defaults/data_video.py)
registers each Video2World dataset name and maps it to its prepared
`*-cosmos-video` directory. The current SO-101 entries are `ex1_all_v4`,
`ex2_all_v4`, and `ex3_all`; the `ex1_ex2_ex3_merged` entry combines all three
with `MultiDataset`. Confirm that every `dataset_dir` points to the directories
created in the previous step.

The experiment grid in
[video2world.py](mimic-video/model/cosmos_predict2/configs/experiment/video2world.py)
automatically creates experiment names for every registered dataset and
supported LoRA rank, learning rate, and batch size. With the current training
defaults, the combined experiment is:

```text
v2w_ex1_ex2_ex3_merged_lora_rank32_lr5.623e-05_bsz64
```

If `DATASET_NAME`, `FINETUNE_DATA_DIR`, or the dataset mix changes, update
`data_video.py` first and ensure the resulting experiment name matches one of
the combinations generated by `video2world.py`.

**4. Run training.** The current defaults select LoRA rank 32, learning rate
`5.623e-05`, local batch size 4, and 16 gradient-accumulation steps. The
registered experiment carries the `bsz64` label:

```bash
./scripts/train_cosmos_video.sh
```

Useful env-var overrides:

- `EX_TYPE`, `DATASET_NAME`, `LORA_RANK`, `LR`, and `BSZ`
- `MAX_ITER`, `SAVE_ITER`, and `LOGGING_ITER`
- `TRAIN_LOCAL_BATCH_SIZE=4`, `GRAD_ACCUM_ITER=16`
- `WANDB_PROJECT=cosmos-video-finetune`, `WANDB_ENTITY=...`, `WANDB_MODE=offline`
- `GPUS_PER_NODE`, `NNODES`, `MASTER_PORT` for distributed training

The pretrained backbone is read from
`mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt`.
Replace or symlink that file to change the initialization checkpoint.

### Mimic-Video Policy Training

The action decoder cross-attends to layer-20 features from the frozen
Video2World backbone and predicts 15 joint targets.

**1. Place zarr datasets.** Put the processed `*-zarr` directories below
`staging/mimic-video/`, or set `MIMIC_VIDEO_DATASET_DIR` to another directory
containing them.

**2. Place a compatible video checkpoint.**

Override `VIDEO_DIT_PATH` and select the matching registered `EXPERIMENT` when
using another checkpoint.

**3. Precompute VAE latents.** The current 21-frame configuration produces
`[16, 6, 60, 80]` latent tensors. Caches are written below
`${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/` and are keyed by the dataset
configuration:

```bash
./scripts/precompute_video_latents.sh
```

**4. Run training.**
For a fresh run, clear the warm-start default:

```bash
LOAD_PATH=null ./scripts/train_mimic_video.sh
```

Useful env-var overrides:

- `EXPERIMENT=w2a_lerobot_<video_ckpt>_lr<...>_layer20_bsz<...>` — pick a different registered experiment (see [world2action.py](mimic-video/model/cosmos_predict2/configs/experiment/world2action.py))
- `VIDEO_DIT_PATH=/abs/path.pt` — override the auto-resolved video backbone
- `MIMIC_VIDEO_DATASET_DIR=/path/to/dir` — directory searched recursively for zarr episodes
- `TRAIN_LOCAL_BATCH_SIZE=32`, `GRAD_ACCUM_ITER=1`
- `NUM_GPUS` — number of GPUs used on the local machine
- `ACTION_MODEL_CHANNELS`, `ACTION_MODEL_BLOCKS`, `ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK`, `ACTION_MODEL_ADALN_LORA_DIM`
- `RUN_VALIDATION=True`, `MAX_VAL_ITER=8` — enable validation and cap its length
- `WANDB_PROJECT=mimic-video`, `WANDB_ENTITY=...`, `WANDB_MODE=offline`

Runs land in `runs/mimic_video/<EXPERIMENT>_<TIMESTAMP>/`.

### Mimic-Video Inference

Run the model server in the model environment on the GPU machine. Set paths
explicitly to the files downloaded from Hugging Face:

```bash
source mimic-video/model/.venv/bin/activate

VIDEO_MODEL_PATH="$(pwd)/checkpoints/video/<video-checkpoint>.pt" \
ACTION_MODEL_PATH="$(pwd)/checkpoints/action/<action-checkpoint>.pt" \
DATASET_STATS="$(pwd)/checkpoints/dataset_statistics.json" \
ACTION_CONFIG_PATH="$(pwd)/checkpoints/action/config.yaml" \
./deployment/serve_mimic_video.sh
```

On the machine connected to the SO-101, run the SSH tunnel and robot client in
**two separate shells**. Keep the SSH tunnel running while the client is in
use.

**Shell 1: open the SSH tunnel**

```bash
ssh -L 8000:localhost:8000 user@<gpu-host>
```

**Shell 2: run the robot client**

```bash
conda activate lerobot
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

The example sets `--max-relative-target 0`, which disables the per-step joint
delta safety limit. Use a positive value such as `5` to enable that limit.
`--stop-after-step 0` stops after the first video DiT forward pass, minimizing
latency. Generally, keep `stop-after-step` low between `0` and `10` for best action model performance.

Prompts are loaded from
[config/deployment_prompts.json](config/deployment_prompts.json). Select one
with `--prompt-key` or override it with `--prompt "..."`.

A live camera window opens on the robot machine by default. Press **`r`** in
that window to start or stop an MP4 recording in `recordings/`.

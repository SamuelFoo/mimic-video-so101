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
hf download robot-learning/Ex1_attempt_1 \
  --repo-type dataset \
  --local-dir ex1_attempt_1
```

Merge the dataset(s) into a single dataset (see `commands.md`).

### Video Model Inference

Run Video2World inference over the merged LeRobot dataset in three steps. All commands are run from the repo root.

**1. Convert LeRobot to zarr and precompute T5 embeddings.** Uses the `lerobot` conda environment for the conversion and `mimic-video/model/.venv` for the embeddings.

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

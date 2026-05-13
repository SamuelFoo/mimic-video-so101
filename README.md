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

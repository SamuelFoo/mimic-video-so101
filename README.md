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

Run Video2World inference over the merged LeRobot dataset in three steps:

```bash
# Run from the repo root. This script converts LeRobot -> zarr in the `lerobot`
# conda environment, then precomputes T5 embeddings in mimic-video/model/.venv.
./scripts/process_lerobot.sh

# Run from the repo root. This script exports existing zarr episodes to MP4
# inputs and builds the Video2World batch JSON.
./scripts/export_video_inputs.sh

# Run from the repo root. This script uses mimic-video/model/.venv.
./scripts/infer_video.sh
```

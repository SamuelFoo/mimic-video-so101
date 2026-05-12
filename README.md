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

Use the `lerobot` conda environment for preprocessing datasets and the `mimic-video` Python virtual environment for training.

### Auth

In the virtual environment, login to `wandb` and `hf`:

```bash
wandb login
hf auth login
```

### Download

Download the checkpoints

```bash
cd mimic-video/model
python scripts/download_checkpoints.py
```

Download the required dataset(s) to the `data/` folder, for example:

```bash
hf download robot-learning/Ex1_attempt_1 \
  --repo-type dataset \
  --local-dir ex1_attempt_1
```

Merge the dataset(s) into a single dataset (see `commands.md`).

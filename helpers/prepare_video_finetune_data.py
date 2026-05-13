#!/usr/bin/env python3
"""Convert per-episode zarrs into the Cosmos Video2World finetuning layout.

Output layout (matches mimic-video/data_preprocessing/video/get_t5_embeddings.py):
    <out_dir>/video/episode_XXXXXX.mp4
    <out_dir>/metas/episode_XXXXXX.txt
    <out_dir>/t5_xxl/                  # created empty; populated by get_t5_embeddings.py

Run get_t5_embeddings.py on <out_dir> after this script to fill t5_xxl/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import zarr
from zarr.errors import PathNotFoundError

from utils.language_instructions import DEFAULT_INSTRUCTIONS_PATH, get_language_instruction


def _read_language(root: zarr.Group, fallback: str) -> str:
    if "language_instruction" not in root:
        return fallback
    value = root["language_instruction"][0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _episode_index(path: Path) -> int:
    return int(path.name.removesuffix(".zarr").rsplit("_", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ex-type", default="ex1")
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    parser.add_argument("--prompt", default=None, help="Override prompt (otherwise read from zarr / instructions file).")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    default_prompt = args.prompt or get_language_instruction(args.ex_type, args.instructions)

    zarr_paths = sorted(args.zarr_dir.glob("episode_*.zarr"), key=_episode_index)
    if args.episodes:
        keep = set(args.episodes)
        zarr_paths = [p for p in zarr_paths if _episode_index(p) in keep]
    if args.max_episodes is not None:
        zarr_paths = zarr_paths[: args.max_episodes]
    if not zarr_paths:
        raise SystemExit(f"No episode_*.zarr directories found in {args.zarr_dir}")

    video_dir = args.out_dir / "video"
    metas_dir = args.out_dir / "metas"
    t5_dir = args.out_dir / "t5_xxl"
    video_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)
    t5_dir.mkdir(parents=True, exist_ok=True)

    for zarr_path in zarr_paths:
        episode = _episode_index(zarr_path)
        stem = f"episode_{episode:06d}"
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except PathNotFoundError:
            print(f"Skipping episode {episode:06d}: incomplete zarr store at {zarr_path}")
            continue
        if "workspace_rgb" not in root:
            print(f"Skipping episode {episode:06d}: no workspace_rgb in {zarr_path}")
            continue

        video_path = video_dir / f"{stem}.mp4"
        meta_path = metas_dir / f"{stem}.txt"
        prompt = _read_language(root, default_prompt)

        if args.overwrite or not video_path.exists():
            frames = root["workspace_rgb"][...]
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=1)

        if args.overwrite or not meta_path.exists():
            meta_path.write_text(prompt.strip() + "\n")

        print(f"episode {episode:06d}: {video_path} | {meta_path}")

    print(f"Done. Run get_t5_embeddings.py --dataset_path {args.out_dir} to fill {t5_dir}.")


if __name__ == "__main__":
    main()

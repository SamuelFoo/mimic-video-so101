#!/usr/bin/env python3
"""Export mimic-video per-episode zarrs to mp4s and a Video2World batch JSON."""

from __future__ import annotations

import argparse
import json
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
    stem = path.name.removesuffix(".zarr")
    return int(stem.rsplit("_", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--ex-type", default="ex1")
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=1.0,
        help="Keep only this leading fraction of each episode (e.g. 0.5 for first half).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.frame_fraction <= 1.0:
        raise SystemExit("--frame-fraction must be in (0, 1]")
    default_prompt = args.prompt or get_language_instruction(args.ex_type, args.instructions)

    zarr_paths = sorted(args.zarr_dir.glob("episode_*.zarr"), key=_episode_index)
    if args.episodes:
        keep = set(args.episodes)
        zarr_paths = [path for path in zarr_paths if _episode_index(path) in keep]
    if args.max_episodes is not None:
        zarr_paths = zarr_paths[: args.max_episodes]
    if not zarr_paths:
        raise SystemExit(f"No episode_*.zarr directories found in {args.zarr_dir}")

    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.batch_json.parent.mkdir(parents=True, exist_ok=True)

    batch: list[dict[str, str]] = []
    for zarr_path in zarr_paths:
        episode = _episode_index(zarr_path)
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except PathNotFoundError:
            print(f"Skipping episode {episode:06d}: incomplete zarr store at {zarr_path}")
            continue
        if "workspace_rgb" not in root:
            print(f"Skipping episode {episode:06d}: {zarr_path} does not contain workspace_rgb")
            continue

        input_video = args.video_dir / f"episode_{episode:06d}.mp4"
        output_video = args.output_dir / f"episode_{episode:06d}_generated.mp4"
        prompt = _read_language(root, default_prompt)

        if args.overwrite or not input_video.exists():
            frames = root["workspace_rgb"][...]
            if frames.dtype != np.uint8:
                frames = np.clip(frames, 0, 255).astype(np.uint8)
            if args.frame_fraction < 1.0:
                keep = max(5, int(round(len(frames) * args.frame_fraction)))
                frames = frames[:keep]
            imageio.mimsave(input_video, frames, fps=args.fps, macro_block_size=1)

        batch.append(
            {
                "input_video": str(input_video.resolve()),
                "prompt": prompt,
                "output_video": str(output_video.resolve()),
            }
        )
        print(f"episode {episode:06d}: {input_video} -> {output_video}")

    if not batch:
        raise SystemExit(
            f"No usable zarr episodes found in {args.zarr_dir}. "
            "If conversion failed earlier, rerun the export step with CONVERT=\"true\"."
        )

    args.batch_json.write_text(json.dumps(batch, indent=2) + "\n")
    print(f"Wrote batch JSON: {args.batch_json}")


if __name__ == "__main__":
    main()

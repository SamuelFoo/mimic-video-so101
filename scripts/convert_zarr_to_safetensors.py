#!/usr/bin/env python3
"""
Convert per-episode .zarr files (mimic-video format) to .safetensors files
compatible with the mimic-video-so101 ChunkReader.

Usage:
    python scripts/convert_zarr_to_safetensors.py \
        --input-dir staging/mimic-video \
        --output-dir staging/mimic-video-st \
        [--num-workers 8] \
        [--overwrite]

The output directory mirrors the input directory structure:
    <input-dir>/some/subdir/episode_000000.zarr
        -> <output-dir>/some/subdir/episode_000000.safetensors

Special handling:
  - language_instruction (dtype=object, bytes) is re-encoded as a uint8
    array of shape (1, N) — identical to how process_libero.py stores it.
    The so101 ChunkReader reads it via get_slice/get_tensor and the
    dataset treats it as LANGUAGE obs_type (no bytes-decoding needed).
  - All other numeric arrays are written verbatim (float32, float16,
    uint8, uint64 are all supported by safetensors).
  - A paths.pkl index cache is deleted from the output dir after the run
    so the so101 utils.get_paths() rescans fresh on the next training run.
"""

import argparse
import multiprocessing
import pathlib
import pickle
import sys
from functools import partial

import numpy as np
import tqdm

# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------

def _convert_episode(
    zarr_path: pathlib.Path,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool,
) -> str:
    """Convert a single .zarr episode to .safetensors. Returns a status string."""
    try:
        import zarr
        import safetensors.numpy as st
    except ImportError as e:
        return f"IMPORT ERROR ({e}): {zarr_path}"

    rel = zarr_path.relative_to(input_dir)
    out_path = (output_dir / rel).with_suffix(".safetensors")

    if out_path.exists() and not overwrite:
        return f"skip (exists): {out_path}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tensors: dict[str, np.ndarray] = {}
        with zarr.open(str(zarr_path), "r") as root:
            for key in root.keys():
                arr = root[key]

                if arr.dtype == object:
                    # language_instruction — stored as bytes objects.
                    # Re-encode each element as UTF-8 bytes and flatten into
                    # a uint8 array of shape (1, N) (N = byte length).
                    # If multiple elements exist they are concatenated with
                    # a null-byte separator so the shape stays (1, total_N).
                    raw: bytes = b"\x00".join(
                        v if isinstance(v, bytes) else str(v).encode("utf-8")
                        for v in arr[...]
                    )
                    tensors[key] = np.frombuffer(raw, dtype=np.uint8)[np.newaxis, :]
                else:
                    # All numeric arrays (float32, float16, uint8, uint64 …)
                    tensors[key] = arr[...]

        st.save_file(tensors, str(out_path))
        return f"ok: {out_path}"

    except Exception as e:
        # Remove partial output so the file isn't treated as valid on re-run
        if out_path.exists():
            out_path.unlink()
        return f"ERROR ({e}): {zarr_path}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert mimic-video .zarr episodes to .safetensors"
    )
    ap.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Root directory containing per-episode .zarr files (searched recursively)",
    )
    ap.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Root directory to write .safetensors files into (mirrors input structure)",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (multiprocessing.cpu_count() or 4) // 2),
        help="Number of parallel conversion workers (default: half of CPU count)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert episodes that already have a .safetensors file",
    )
    args = ap.parse_args()

    input_dir: pathlib.Path = args.input_dir.resolve()
    output_dir: pathlib.Path = args.output_dir.resolve()

    if not input_dir.exists():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    zarr_paths = sorted(input_dir.glob("**/*.zarr"))
    if not zarr_paths:
        print(f"No .zarr files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Episodes:    {len(zarr_paths)}")
    print(f"Workers:     {args.num_workers}")
    print(f"Overwrite:   {args.overwrite}")
    print()

    worker_fn = partial(
        _convert_episode,
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    errors: list[str] = []

    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            for msg in tqdm.tqdm(
                pool.imap_unordered(worker_fn, zarr_paths),
                total=len(zarr_paths),
                desc="zarr → safetensors",
            ):
                print(msg)
                if msg.startswith("ERROR"):
                    errors.append(msg)
    else:
        for path in tqdm.tqdm(zarr_paths, desc="zarr → safetensors"):
            msg = worker_fn(path)
            print(msg)
            if msg.startswith("ERROR"):
                errors.append(msg)

    # Invalidate the paths.pkl index so mimic-video-so101's utils.get_paths()
    # rescans the directory on the next training run.
    paths_cache = output_dir / "paths.pkl"
    if paths_cache.exists():
        paths_cache.unlink()
        print(f"\nRemoved stale paths cache: {paths_cache}")

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All episodes converted successfully.")


if __name__ == "__main__":
    main()

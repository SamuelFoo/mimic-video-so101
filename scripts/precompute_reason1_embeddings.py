#!/usr/bin/env python3
"""Pre-compute Reason1 + crossattn_proj embeddings for MimicVideoDataset.

Produces per-episode pickle files in <dataset_dir>/reason1_proj/ with shape
[n_tokens, 1024] — identical format to t5_xxl/ but with the correct
embedding distribution for the Cosmos-Predict2.5 cross-attention weights.

The pipeline is:
  caption text  →  Reason1-7B (FULL_CONCAT, 100352-dim)
                →  crossattn_proj (frozen Linear 100352→1024, from 2B checkpoint)
                →  stored as [n_tokens, 1024] numpy float16

This moves the crossattn_proj out of every training forward pass, saving
~200 MB of model weights + ~98 MB of conditioning tensor per training step.
During training set use_crossattn_projection=False in the model net config.

Usage:
    cd /ephemeral/robot_learning_project/cosmos-predict2.5
    source .venv/bin/activate
    python ../scripts/precompute_reason1_embeddings.py \\
        --dataset_dirs \\
            /ephemeral/robot_learning_project/staging/mimic-video/ex1_all_v4-cosmos-video \\
            /ephemeral/robot_learning_project/staging/mimic-video/ex2_all_v4-cosmos-video \\
        --predict2_checkpoint \\
            /home/shadeform/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/15a82a2ec231bc318692aa0456a36537c806e7d4/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Must be imported before any cosmos checkpoint URIs are resolved — this
# populates the checkpoint registry (S3 URI → HuggingFace mapping).
from cosmos_oss.checkpoints_predict2 import register_checkpoints as _register_checkpoints
_register_checkpoints()

# ---------------------------------------------------------------------------
# Constants — must match TextEncoder and MimicVideoDataset
# ---------------------------------------------------------------------------
_NUM_TOKENS = 512  # padding length
_REASON1_DIM = 100352  # 28 layers × 3584 (Qwen2.5-VL-7B FULL_CONCAT)
_PROJ_OUT_DIM = 1024  # crossattn_proj output = K/V proj input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_crossattn_proj(checkpoint_path: str) -> nn.Linear:
    """Extract the frozen crossattn_proj from the 2B checkpoint."""
    print(f"Loading crossattn_proj from {checkpoint_path} ...")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    proj = nn.Linear(_REASON1_DIM, _PROJ_OUT_DIM, bias=True)
    proj.weight = nn.Parameter(ck["net.crossattn_proj.0.weight"].bfloat16())
    proj.bias = nn.Parameter(ck["net.crossattn_proj.0.bias"].bfloat16())
    del ck
    proj.eval()
    proj.cuda()
    print("  crossattn_proj loaded.")
    return proj


def load_reason1_encoder():
    """Load the Reason1-7B text encoder (FULL_CONCAT strategy)."""
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
        TextEncoder,
        TextEncoderConfig,
    )
    from cosmos_predict2._src.imaginaire.utils.embedding_concat_strategy import EmbeddingConcatStrategy

    print("Loading Reason1-7B encoder (this may take ~1 min) ...")
    config = TextEncoderConfig(
        compute_online=True,
        embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
        # UUID cb3e3ffa resolves to nvidia/Cosmos-Reason1-7B (already cached)
        ckpt_path="cb3e3ffa-7b08-4c34-822d-61c7aa31a14f",
    )
    encoder = TextEncoder(config, device="cuda")
    print("  Reason1-7B loaded.")
    return encoder


def get_token_counts(encoder, captions: list[str]) -> list[int]:
    """Return actual (pre-padding) token count for each caption."""
    counts = []
    for cap in captions:
        conversations = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}],
            },
            {"role": "user", "content": [{"type": "text", "text": cap}]},
        ]
        tok_out = encoder.model.tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False, add_vision_id=False
        )
        counts.append(min(len(tok_out["input_ids"]), _NUM_TOKENS))
    return counts


@torch.no_grad()
def compute_projected_embeddings(
    encoder,
    crossattn_proj: nn.Linear,
    captions: list[str],
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Return {caption: np.float16 array [n_tokens, 1024]} for each unique caption."""
    unique_captions = list(dict.fromkeys(captions))  # deduplicated, order-preserving
    results: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(unique_captions), batch_size), desc="  embedding batches"):
        batch_caps = unique_captions[i : i + batch_size]

        # Get actual token counts before encoding
        n_tokens_list = get_token_counts(encoder, batch_caps)

        # Reason1 FULL_CONCAT: [B, 512, 100352]
        data_batch = {"ai_caption": batch_caps}
        emb_100k = encoder.compute_text_embeddings_online(data_batch, "ai_caption")  # [B, 512, 100352]

        # Project to 1024: [B, 512, 1024]
        emb_1024 = crossattn_proj(emb_100k)  # [B, 512, 1024]

        for j, cap in enumerate(batch_caps):
            n = n_tokens_list[j]
            arr = emb_1024[j, :n, :].cpu().to(torch.float16).numpy()  # [n, 1024]
            results[cap] = arr

        torch.cuda.empty_cache()

    return results


def process_dataset(
    dataset_dir: str,
    encoder,
    crossattn_proj: nn.Linear,
    batch_size: int,
    overwrite: bool,
) -> None:
    metas_dir = Path(dataset_dir) / "metas"
    video_dir = Path(dataset_dir) / "video"
    out_dir = Path(dataset_dir) / "reason1_proj"

    if not metas_dir.exists():
        print(f"  WARN: no metas/ in {dataset_dir}, skipping.")
        return
    if not video_dir.exists():
        print(f"  WARN: no video/ in {dataset_dir}, skipping.")
        return

    out_dir.mkdir(exist_ok=True)

    # Collect episodes
    txt_files = sorted(metas_dir.glob("*.txt"))
    print(f"  {len(txt_files)} episodes found.")

    # Skip already done
    if not overwrite:
        txt_files = [f for f in txt_files if not (out_dir / f.with_suffix(".pickle").name).exists()]
        print(f"  {len(txt_files)} episodes need computation.")

    if not txt_files:
        print("  Nothing to do.")
        return

    # Read captions
    captions = []
    for f in txt_files:
        captions.append(f.read_text(encoding="utf-8").strip())

    # Compute all unique captions
    print(f"  Computing embeddings for {len(set(captions))} unique captions ...")
    cap_to_emb = compute_projected_embeddings(encoder, crossattn_proj, captions, batch_size)

    # Write per-episode pickle files
    print("  Saving pickle files ...")
    for txt_file, cap in zip(txt_files, captions):
        arr = cap_to_emb[cap]  # [n_tokens, 1024]
        out_path = out_dir / txt_file.with_suffix(".pickle").name
        with open(out_path, "wb") as f:
            pickle.dump([arr], f)  # list-of-one-array, same format as t5_xxl

    print(f"  Done. Saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        help="One or more dataset root dirs, each containing metas/ and video/",
    )
    p.add_argument(
        "--predict2_checkpoint",
        required=True,
        help="Path to the Cosmos-Predict2.5-2B .pt checkpoint (for crossattn_proj weights)",
    )
    p.add_argument("--batch_size", type=int, default=4, help="Reason1 batch size (reduce if OOM)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output pickle exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Add cosmos-predict2.5 to path if needed
    repo_root = Path(__file__).resolve().parent.parent / "cosmos-predict2.5"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    crossattn_proj = load_crossattn_proj(args.predict2_checkpoint)
    encoder = load_reason1_encoder()

    for dataset_dir in args.dataset_dirs:
        print(f"\n=== {dataset_dir} ===")
        process_dataset(dataset_dir, encoder, crossattn_proj, args.batch_size, args.overwrite)

    print("\nAll done. Set use_crossattn_projection=False in model net config for training.")


if __name__ == "__main__":
    main()

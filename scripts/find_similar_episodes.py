"""
Find similar ball trajectories across episodes using pre-computed metadata.

Algorithm
---------
1. Load centroids (N, 2) from every ep*.h5 in <dataset>/metadata/.
2. Drop leading / trailing NaN frames (untracked region).
3. Normalise each episode's frame index to t ∈ [0, 1].
4. Subsample at N_SAMPLES equidistant points (default 11 → every 10 % of
   episode duration) by snapping each target t to the nearest valid frame.
5. Compute all-pairs mean-L2 distance over the N_SAMPLES points.
6. Cluster episodes whose distance < THRESHOLD with union-find.

Distance metric: mean L2 (Euclidean) per sample — natural for 2-D pixel coords.
Threshold guess: 20 px (based on prior DTW analysis where similar pairs scored
12–15 px and dissimilar pairs scored 30–125 px).

Outputs
-------
  stdout                         — pairwise table (* = below threshold) + clusters
  <metadata>/similarity_heatmap.png — colour-coded distance matrix

Usage
-----
    python scripts/find_similar_episodes.py
    python scripts/find_similar_episodes.py --threshold 25
    python scripts/find_similar_episodes.py --n-samples 21   # every 5 %
    python scripts/find_similar_episodes.py --dataset /other/path
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATASET_ID        = "/home/nielsen/codes/Ex1_attempt_1"
THRESHOLD         = 20.0   # px — mean-L2 below this → episodes are "similar"
SAMPLE_INTERVAL_S = 0.1    # seconds between samples; step ∝ episode length


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_centroids(h5_path: Path) -> np.ndarray:
    """Return (N, 2) float32 array from h5 file; NaN where detection was missing."""
    with h5py.File(h5_path, "r") as f:
        return f["centroids"][:]   # (N, 2)


def strip_nan_ends(centroids: np.ndarray) -> np.ndarray:
    """Remove leading and trailing all-NaN rows."""
    valid = ~np.isnan(centroids[:, 0])
    if not valid.any():
        return centroids  # all NaN — caller will skip
    first, last = valid.argmax(), len(valid) - valid[::-1].argmax()
    return centroids[first:last]


# ---------------------------------------------------------------------------
# Normalisation + subsampling
# ---------------------------------------------------------------------------

def subsample(centroids: np.ndarray, n_samples: int) -> np.ndarray | None:
    """
    Normalise frame index to [0, 1] then snap n_samples equidistant target
    times to the nearest non-NaN frame.

    Returns (n_samples, 2) float32, or None if too few valid frames.
    """
    centroids = strip_nan_ends(centroids)
    valid_mask = ~np.isnan(centroids[:, 0])
    n_valid = valid_mask.sum()

    if n_valid < n_samples:
        return None  # not enough data

    n = len(centroids)
    t = np.linspace(0.0, 1.0, n)          # normalised time per frame
    t_valid = t[valid_mask]
    pts_valid = centroids[valid_mask]      # (n_valid, 2)

    targets = np.linspace(0.0, 1.0, n_samples)
    result  = np.empty((n_samples, 2), dtype=np.float32)

    for i, target in enumerate(targets):
        nearest = np.argmin(np.abs(t_valid - target))
        result[i] = pts_valid[nearest]

    return result


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-sample Euclidean distance between two (N, 2) arrays."""
    return float(np.linalg.norm(a - b, axis=1).mean())


# ---------------------------------------------------------------------------
# Clustering (union-find)
# ---------------------------------------------------------------------------

def cluster_episodes(episodes: list[int], similar_pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = {ep: ep for ep in episodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in similar_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for ep in episodes:
        root = find(ep)
        groups.setdefault(root, []).append(ep)

    return [sorted(g) for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Cluster trajectory plots
# ---------------------------------------------------------------------------

def save_cluster_plots(clusters: list[list[int]], trajectories: dict[int, np.ndarray], out_dir: Path):
    """
    For each cluster, save a 2-panel PNG:
      left  — 2D spatial path (x vs y, image coords → y flipped)
      right — x and y vs normalised time [0, 1]
    Each episode is a distinct colour.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ci, group in enumerate(clusters):
        fig, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Cluster {ci + 1}: {[f'ep{e:03d}' for e in group]}", fontsize=11)

        for k, ep in enumerate(group):
            traj = trajectories[ep]          # (n_samples, 2)
            n    = len(traj)
            t    = np.linspace(0.0, 1.0, n)
            x, y = traj[:, 0], traj[:, 1]
            color = colors[k % len(colors)]
            label = f"ep{ep:03d}"

            # 2D spatial
            ax_xy.plot(x, y, color=color, linewidth=1.5, label=label)
            ax_xy.scatter(x[[0]], y[[0]], color=color, marker="o", s=40, zorder=5)
            ax_xy.scatter(x[[-1]], y[[-1]], color=color, marker="s", s=40, zorder=5)

            # x / y vs time
            ax_t.plot(t, x, color=color, linewidth=1.5, linestyle="-",  label=f"{label} x")
            ax_t.plot(t, y, color=color, linewidth=1.5, linestyle="--", label=f"{label} y")

        ax_xy.set_xlabel("x (px)")
        ax_xy.set_ylabel("y (px)")
        ax_xy.invert_yaxis()   # image coords: y increases downward
        ax_xy.set_title("2D path  (○=start  □=end)")
        ax_xy.legend(fontsize=8)
        ax_xy.set_aspect("equal", adjustable="datalim")

        ax_t.set_xlabel("normalised time  t ∈ [0, 1]")
        ax_t.set_ylabel("position (px)")
        ax_t.set_title("x (solid) and y (dashed) vs time")
        ax_t.legend(fontsize=7, ncol=2)

        plt.tight_layout()
        path = out_dir / f"cluster_{ci + 1:02d}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved cluster plot → {path}")


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def save_heatmap(dist_matrix: np.ndarray, episode_ids: list[int], threshold: float, path: str):
    n = len(episode_ids)
    labels = [f"ep{e:03d}" for e in episode_ids]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.5)))
    im = ax.imshow(dist_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=dist_matrix[~np.isnan(dist_matrix)].max())

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = dist_matrix[i, j]
            if np.isnan(v):
                continue
            text  = f"{v:.1f}"
            color = "white" if v > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Mean L2 distance (px) — threshold = {threshold} px", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("mean L2 (px)")

    # Highlight similar pairs with a green border
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(dist_matrix[i, j]) and dist_matrix[i, j] < threshold:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find similar ball trajectories from metadata h5 files.")
    parser.add_argument("--dataset",    default=DATASET_ID,  help=f"LeRobot dataset path (default: {DATASET_ID})")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD, help=f"Similarity threshold in px (default: {THRESHOLD})")
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_S,
                        help=f"Seconds between samples (default: {SAMPLE_INTERVAL_S})")
    parser.add_argument("--from-episode", type=int, default=0,
                        help="Only process episodes with index >= this value (default: 0)")
    args = parser.parse_args()

    metadata_dir = Path(args.dataset) / "metadata"
    h5_files     = [p for p in sorted(metadata_dir.glob("ep*.h5"))
                    if int(p.stem[2:]) >= args.from_episode]

    if not h5_files:
        print(f"No ep*.h5 files found in {metadata_dir} (from episode {args.from_episode})")
        return

    # Read FPS from dataset info
    info_path = Path(args.dataset) / "meta" / "info.json"
    with open(info_path) as f:
        fps = float(json.load(f)["fps"])

    print(f"Found {len(h5_files)} h5 files in {metadata_dir}")
    print(f"FPS: {fps}  |  Sample interval: {args.sample_interval}s  |  Threshold: {args.threshold} px\n")

    # --- First pass: load centroids, find shortest valid episode ---
    raw: dict[int, np.ndarray] = {}
    for h5_path in h5_files:
        ep = int(h5_path.stem[2:])
        raw[ep] = load_centroids(h5_path)

    valid_lengths = []
    for ep, centroids in raw.items():
        stripped = strip_nan_ends(centroids)
        n_valid = int((~np.isnan(stripped[:, 0])).sum())
        if n_valid > 0:
            valid_lengths.append(n_valid)

    if not valid_lengths:
        print("No valid episodes found.")
        return

    n_samples = max(2, int(min(valid_lengths) / (fps * args.sample_interval)))
    print(f"N_samples = {n_samples}  (shortest valid episode: {min(valid_lengths)} frames)\n")

    # --- Second pass: subsample ---
    trajectories: dict[int, np.ndarray] = {}
    for h5_path in h5_files:
        ep = int(h5_path.stem[2:])
        centroids = raw[ep]
        sampled   = subsample(centroids, n_samples)
        if sampled is None:
            print(f"  ep{ep:03d}: too few valid frames — skipped")
        else:
            trajectories[ep] = sampled
            print(f"  ep{ep:03d}: {len(centroids)} frames → {n_samples} samples")

    episodes = sorted(trajectories)
    n = len(episodes)
    if n < 2:
        print("Need at least 2 valid episodes.")
        return

    # --- Pairwise distances ---
    dist_matrix = np.full((n, n), np.nan)
    for i in range(n):
        dist_matrix[i, i] = 0.0

    similar_pairs: list[tuple[int, int]] = []

    print(f"\nPairwise mean-L2 distances (px)  [* = below {args.threshold} px threshold]  ({n_samples} samples/ep)")
    print("-" * 55)

    for i, j in combinations(range(n), 2):
        ep_i, ep_j = episodes[i], episodes[j]
        d = mean_l2(trajectories[ep_i], trajectories[ep_j])
        dist_matrix[i, j] = dist_matrix[j, i] = d
        flag = " *" if d < args.threshold else ""
        print(f"  ep{ep_i:03d} vs ep{ep_j:03d}  {d:7.2f} px{flag}")
        if d < args.threshold:
            similar_pairs.append((ep_i, ep_j))

    # --- Clusters ---
    clusters = cluster_episodes(episodes, similar_pairs)
    print(f"\nSimilar clusters ({len(clusters)} found):")
    if clusters:
        for group in clusters:
            print(f"  {[f'ep{e:03d}' for e in group]}")
    else:
        print("  None — no pairs below threshold")

    # --- Cluster plots ---
    if clusters:
        save_cluster_plots(clusters, trajectories, metadata_dir)

    # --- Heatmap ---
    save_heatmap(dist_matrix, episodes, args.threshold,
                 str(metadata_dir / "similarity_heatmap.png"))


if __name__ == "__main__":
    main()

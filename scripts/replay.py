"""
Kinematic replay of SO-101 joint states from a LeRobot dataset using Pinocchio + Meshcat,
with synchronized front camera video displayed via matplotlib.

Install dependencies (activate your lerobot conda env first):
    conda install pinocchio -c conda-forge
    pip install meshcat datasets av pandas pyarrow matplotlib
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import av
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datasets import load_dataset
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

def get_dataset_fps(dataset_id):
    info_path = Path(dataset_id) / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)["fps"]

def get_episode_video_info(dataset_id, episode):
    """Return (video_path, from_timestamp, to_timestamp) for the front camera of the given episode."""
    episodes_dir = Path(dataset_id) / "meta" / "episodes"
    episode_dfs = [pd.read_parquet(p) for p in sorted(episodes_dir.rglob("*.parquet"))]
    episodes = pd.concat(episode_dfs, ignore_index=True)
    row = episodes[episodes["episode_index"] == episode].iloc[0]
    chunk_idx = int(row["videos/observation.images.front/chunk_index"])
    file_idx = int(row["videos/observation.images.front/file_index"])
    from_ts = float(row["videos/observation.images.front/from_timestamp"])
    to_ts = float(row["videos/observation.images.front/to_timestamp"])
    video_path = (
        Path(dataset_id)
        / "videos"
        / "observation.images.front"
        / f"chunk-{chunk_idx:03d}"
        / f"file-{file_idx:03d}.mp4"
    )
    return str(video_path), from_ts, to_ts

def episode_video_frames(video_path, from_ts, to_ts):
    """Generator yielding episode frames as RGB numpy arrays using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO

    # Seek to just before the episode start
    seek_ts = int(from_ts / stream.time_base)
    container.seek(seek_ts, stream=stream, any_frame=False, backward=True)

    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t < from_ts - 0.001:
            continue
        if t >= to_ts + 0.001:
            break
        yield frame.to_ndarray(format="rgb24")

    container.close()

def replay_kinematics_meshcat(dataset_id, urdf_path, mesh_dir, episode=0, fps=None):
    if fps is None:
        fps = get_dataset_fps(dataset_id)
        print(f"FPS from dataset metadata: {fps}")

    print(f"Loading dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")
    episode_frames = dataset.filter(lambda x: x["episode_index"] == episode)
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode} not found in dataset.")
    print(f"Episode {episode}: {len(episode_frames)} frames")

    print(f"Loading Pinocchio model from URDF: {urdf_path}...")
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. Make sure you have meshcat installed.")
        raise err

    viz.loadViewerModel()

    print("Meshcat Viewer initialized.")
    print("If your browser didn't open automatically, look for the local URL above (usually http://127.0.0.1:7000)")

    # Set up front camera video
    video_path, from_ts, to_ts = get_episode_video_info(dataset_id, episode)
    print(f"Opening front camera video: {video_path} (start={from_ts:.2f}s, end={to_ts:.2f}s)")
    frames_gen = episode_video_frames(video_path, from_ts, to_ts)

    # Set up matplotlib window
    plt.ion()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.canvas.manager.set_window_title(f"Front Camera — Episode {episode}")
    ax.axis("off")
    im_handle = None

    print("Starting replay... Close the camera window to stop early.")
    time.sleep(1)

    sleep_time = 1.0 / fps

    for i, frame in enumerate(episode_frames):
        t_start = time.time()

        joint_angles = np.deg2rad(np.array(frame["observation.state"]))
        if len(joint_angles) > model.nq:
            joint_angles = joint_angles[:model.nq]
        viz.display(joint_angles)

        try:
            rgb = next(frames_gen)
            if im_handle is None:
                im_handle = ax.imshow(rgb)
            else:
                im_handle.set_data(rgb)
            fig.canvas.draw_idle()
            plt.pause(0.001)
        except StopIteration:
            pass

        if not plt.fignum_exists(fig.number):
            print("Camera window closed. Stopping replay.")
            break

        if i % fps == 0:
            print(f"Frame {i}/{len(episode_frames)}...")

        elapsed = time.time() - t_start
        remaining = sleep_time - elapsed
        if remaining > 0:
            time.sleep(remaining)

    plt.close("all")
    print(f"Episode {episode} replay finished.")

if __name__ == "__main__":
    DATASET_ID = "/home/nielsen/codes/ex2_all"
    URDF_PATH  = "/home/nielsen/codes/robot_learning_project/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
    MESH_DIR   = "/home/nielsen/codes/robot_learning_project/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf"

    parser = argparse.ArgumentParser(description="Replay SO-101 episode kinematics in Meshcat.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    args = parser.parse_args()

    replay_kinematics_meshcat(DATASET_ID, URDF_PATH, mesh_dir=MESH_DIR, episode=args.episode)

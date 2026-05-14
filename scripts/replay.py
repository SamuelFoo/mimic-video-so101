"""
Kinematic replay of SO-101 joint states from a LeRobot dataset using Pinocchio + Meshcat.

Install dependencies (activate your lerobot conda env first):
    conda install pinocchio -c conda-forge
    pip install meshcat datasets
"""
import json
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

def get_dataset_fps(dataset_id):
    info_path = Path(dataset_id) / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)["fps"]

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
    print("Starting replay...")
    time.sleep(1)

    sleep_time = 1.0 / fps

    for i, frame in enumerate(episode_frames):
        joint_angles = np.deg2rad(np.array(frame['observation.state']))
        if len(joint_angles) > model.nq:
            joint_angles = joint_angles[:model.nq]
        viz.display(joint_angles)
        if i % fps == 0:
            print(f"Frame {i}/{len(episode_frames)}...")
        time.sleep(sleep_time)

    print(f"Episode {episode} replay finished.")

if __name__ == "__main__":
    # --- Configuration ---
    
    # Replace with your actual Hugging Face repo ID or local path
    DATASET_ID = "/home/nielsen/codes/Ex1_attempt_1" 
    
    # Path to the SO-101 URDF file
    URDF_PATH = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"

    # Meshes are referenced as "assets/..." relative to the URDF, so point to the urdf/ folder
    MESH_DIR = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf"
    
    EPISODE = 0  # Change to replay a different episode

    replay_kinematics_meshcat(DATASET_ID, URDF_PATH, mesh_dir=MESH_DIR, episode=EPISODE)
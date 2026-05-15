"""Real-time SO-101 inference loop driven by the mimic-video server.

This script runs in the **lerobot conda env** on the same laptop that has the
SO-101 connected over USB. It talks to a GPU box that's running
`deployment/serve_mimic_video.sh`. Per-step compute on the laptop is just
JPEG encoding + an HTTP POST.

Loop:
    1. Read joint state + camera frame from the SO-101.
    2. POST them to the server, get back a 15-step action chunk (joint angles
       in degrees, in canonical SO-101 motor order).
    3. Replay the chunk to the robot at the policy's target frequency (5 Hz).
    4. Repeat. While the chunk plays, the server is idle — the next /infer
       blocks until inference returns (a few seconds), then the next chunk
       plays. See the `num-execute` flag below if you want to re-plan more
       often than every full chunk (at the cost of more idle waits).

Run (in the lerobot env, on the laptop):
    conda activate lerobot
    python deployment/run_so101_inference.py \\
        --port /dev/ttyACM0 \\
        --robot-id my_awesome_follower_arm \\
        --server http://gpu-box:8000 \\
        --prompt-key ex1

Ctrl-C disconnects cleanly.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import signal
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

# lerobot
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# Our HTTP client lives alongside this file.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from client_mimic_video import MimicVideoClient  # noqa: E402

# Canonical SO-101 motor order — matches `observation.state` / `action` in the
# LeRobot dataset features (data/ex*_merged/meta/info.json) and the order
# MimicDataset trained on.
MOTOR_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_DIM = len(MOTOR_ORDER)


def load_prompt(args: argparse.Namespace, repo_root: pathlib.Path) -> str:
    if args.prompt:
        return args.prompt
    instructions_path = args.instructions_json or (repo_root / "config" / "language_instructions.json")
    with open(instructions_path) as f:
        instructions = json.load(f)
    if args.prompt_key not in instructions:
        raise SystemExit(
            f"Prompt key {args.prompt_key!r} not in {instructions_path}. "
            f"Available: {sorted(instructions)}"
        )
    return instructions[args.prompt_key]


def encode_frame_jpeg(frame_rgb: np.ndarray, quality: int) -> bytes:
    """Encode an HxWx3 uint8 RGB frame as JPEG bytes. Matches the server's
    `PIL.Image.open(...).convert('RGB')` decode path exactly."""
    pil = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def state_from_obs(obs: dict[str, Any]) -> list[float]:
    return [float(obs[f"{m}.pos"]) for m in MOTOR_ORDER]


def action_dict_from_vector(action: list[float] | np.ndarray) -> dict[str, float]:
    if len(action) != STATE_DIM:
        raise RuntimeError(f"Server returned {len(action)}-D action; expected {STATE_DIM} for SO-101.")
    return {f"{m}.pos": float(v) for m, v in zip(MOTOR_ORDER, action)}


def make_robot(args: argparse.Namespace) -> SO101Follower:
    cam_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.camera_fps,
        width=args.camera_width,
        height=args.camera_height,
    )
    robot_cfg = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.port,
        cameras={args.camera_key: cam_cfg},
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(robot_cfg)


def run_loop(args: argparse.Namespace, repo_root: pathlib.Path) -> None:
    prompt = load_prompt(args, repo_root)
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", file=sys.stderr)

    client = MimicVideoClient(server=args.server or os.environ.get("MIMIC_VIDEO_SERVER", "http://localhost:8000"),
                              timeout=args.timeout)
    print(f"Server: {client.server}", file=sys.stderr)
    health = client.health()
    if not health.get("ok"):
        raise SystemExit(f"Server not ready: {health}")

    robot = make_robot(args)
    robot.connect()
    print(f"Connected to SO-101 on {args.port}", file=sys.stderr)

    # Tell the server we're starting a fresh episode.
    client.reset(prompt)

    step_dt = 1.0 / args.chunk_rate_hz
    total_steps = 0
    stop_flag = {"value": False}

    def _handle_sigint(signum, frame):  # noqa: ARG001
        stop_flag["value"] = True
        print("\nCtrl-C received, finishing current step then disconnecting...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not stop_flag["value"]:
            if args.max_steps and total_steps >= args.max_steps:
                break

            # ---- Plan: query the server with the current observation -------
            obs = robot.get_observation()
            state = state_from_obs(obs)
            frame = obs[args.camera_key]
            if frame is None:
                raise RuntimeError(f"Camera {args.camera_key!r} returned None — is the device free?")

            jpeg = encode_frame_jpeg(frame, quality=args.jpeg_quality)

            t0 = time.perf_counter()
            resp = client.infer(
                prompt=prompt,
                state=state,
                image_bytes=jpeg,
                return_full_chunk=True,
                num_sampling_step=args.num_sampling_step,
                stop_after_step=args.stop_after_step,
            )
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            actions = resp["actions"]
            n_to_execute = min(args.num_execute, len(actions))
            print(
                f"[step {total_steps}] state={['%.1f' % s for s in state]} "
                f"-> {len(actions)} actions ({rtt_ms:.0f} ms round-trip, "
                f"server {resp['infer_ms']:.0f} ms, ran_model={resp['ran_model']}); "
                f"executing {n_to_execute}",
                file=sys.stderr,
            )

            # ---- Execute: replay the first N actions at the policy rate ----
            chunk_start = time.perf_counter()
            for i in range(n_to_execute):
                if stop_flag["value"]:
                    break
                action = actions[i]
                action_dict = action_dict_from_vector(action)
                if args.dry_run:
                    print(f"  dry-run action[{i}] = {action_dict}", file=sys.stderr)
                else:
                    robot.send_action(action_dict)

                total_steps += 1
                if args.max_steps and total_steps >= args.max_steps:
                    break

                # Pace to step_dt from the start of the chunk so jitter in
                # individual send_action calls doesn't accumulate.
                target_t = chunk_start + (i + 1) * step_dt
                now = time.perf_counter()
                if target_t > now:
                    time.sleep(target_t - now)
    finally:
        try:
            robot.disconnect()
        except Exception as exc:  # pragma: no cover
            print(f"Warning: robot.disconnect() raised: {exc}", file=sys.stderr)
        print(f"Done. Executed {total_steps} actions.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Robot
    p.add_argument("--port", required=True, help="USB port for the SO-101 follower (e.g. /dev/ttyACM0)")
    p.add_argument("--robot-id", required=True, help="Calibration id (must match the id you teleoperated with)")
    p.add_argument("--max-relative-target", type=float, default=5.0,
                   help="Per-step max joint delta in degrees (safety). Set None to disable.")

    # Camera
    p.add_argument("--camera-key", default="front",
                   help="Camera dict key — must match the model's training camera ('front' for this repo)")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30,
                   help="Camera capture fps. The model conditions at 5 Hz; capture fps just needs to be high enough.")

    # Prompt
    p.add_argument("--prompt", default=None, help="Override prompt text (else loaded from --instructions-json by key)")
    p.add_argument("--prompt-key", default="ex1",
                   help="Key into the instructions JSON (e.g. 'ex1', 'ex2')")
    p.add_argument("--instructions-json", type=pathlib.Path, default=None,
                   help="Path to language instructions JSON (default: <repo>/config/language_instructions.json)")

    # Server
    p.add_argument("--server", default=None, help="Full server URL (else env MIMIC_VIDEO_SERVER or http://localhost:8000)")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds")
    p.add_argument("--num-sampling-step", type=int, default=35)
    p.add_argument("--stop-after-step", type=int, default=None,
                   help="Stop video denoising after this step (trade quality for latency)")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for the frame upload (1-100)")

    # Loop control
    p.add_argument("--chunk-rate-hz", type=float, default=5.0,
                   help="Replay rate within each action chunk (matches the 5 Hz training target_frequency)")
    p.add_argument("--num-execute", type=int, default=15,
                   help="How many of the 15-step chunk to execute before re-planning. Lower = more re-plans, "
                        "more idle waits during inference; higher = less reactive.")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many total actions sent. 0 = unlimited (Ctrl-C to stop).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions but don't send them to the robot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_relative_target is not None and args.max_relative_target <= 0:
        args.max_relative_target = None
    repo_root = _HERE.parent
    run_loop(args, repo_root)


if __name__ == "__main__":
    main()

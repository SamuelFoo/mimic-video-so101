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

Dependencies (in the lerobot conda env):
    conda install pinocchio -c conda-forge
    pip install meshcat
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import pathlib
import signal
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    _PINOCCHIO_AVAILABLE = True
except ImportError:
    _PINOCCHIO_AVAILABLE = False

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


_DEFAULT_URDF = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
_DEFAULT_MESH_DIR = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf"


class DisplayRecorder:
    """Live camera preview with a single-key toggle for MP4 recording.

    `show(frame_rgb)` is called every time the laptop reads a frame; pressing
    `record_key` (default `r`) starts/stops writing the displayed frames to
    `record_dir/so101_<timestamp>.mp4` at the camera's capture fps.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        window_name: str,
        record_dir: pathlib.Path,
        fps: float,
        record_key: str,
    ) -> None:
        self.enabled = enabled
        self.window_name = window_name
        self.history_window_name = f"{window_name} — sent frames"
        self.record_dir = record_dir
        self.fps = max(1.0, float(fps))
        self.record_key = (record_key or "r")[:1].lower()
        self._writer: cv2.VideoWriter | None = None
        self._record_path: pathlib.Path | None = None
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.history_window_name, cv2.WINDOW_NORMAL)
            print(
                f"Display on. Press '{self.record_key}' in the camera window to "
                f"start/stop recording (saved to {self.record_dir}/).",
                file=sys.stderr,
            )

    def show(self, frame_rgb: np.ndarray) -> None:
        if not self.enabled or frame_rgb is None:
            return
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._writer is not None:
            self._writer.write(bgr)
            cv2.circle(bgr, (18, 18), 8, (0, 0, 255), -1)
            cv2.putText(bgr, "REC", (32, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.window_name, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(self.record_key):
            self._toggle(bgr.shape[:2])

    def show_history(self, jpeg_list: list[bytes]) -> None:
        """Decode the JPEG history that's about to be sent to the server and
        display it as a horizontal strip (oldest → newest, left → right)."""
        if not self.enabled or not jpeg_list:
            return
        target_h = 180
        tiles = []
        for idx, b in enumerate(jpeg_list):
            arr = np.frombuffer(b, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * target_h / h), target_h))
            label = f"t-{len(jpeg_list) - 1 - idx}" if idx < len(jpeg_list) - 1 else "t (now)"
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
        if not tiles:
            return
        sep = np.full((target_h, 4, 3), 255, dtype=np.uint8)
        pieces: list[np.ndarray] = []
        for i, tile in enumerate(tiles):
            if i > 0:
                pieces.append(sep)
            pieces.append(tile)
        strip = np.concatenate(pieces, axis=1)
        cv2.imshow(self.history_window_name, strip)
        cv2.waitKey(1)

    def _toggle(self, frame_hw: tuple[int, int]) -> None:
        if self._writer is None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._record_path = self.record_dir / f"so101_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            h, w = frame_hw
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._record_path), fourcc, self.fps, (w, h))
            print(f"Recording → {self._record_path}", file=sys.stderr)
        else:
            self._writer.release()
            print(f"Saved recording → {self._record_path}", file=sys.stderr)
            self._writer = None
            self._record_path = None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"Saved recording → {self._record_path}", file=sys.stderr)
            self._writer = None
        if self.enabled:
            cv2.destroyAllWindows()


def setup_meshcat(urdf_path: str, mesh_dir: str, repo_root: pathlib.Path):
    if not _PINOCCHIO_AVAILABLE:
        raise SystemExit("pinocchio not installed. Run: conda install pinocchio -c conda-forge && pip install meshcat")
    urdf = pathlib.Path(urdf_path) if pathlib.Path(urdf_path).is_absolute() else repo_root / urdf_path
    mesh = pathlib.Path(mesh_dir) if pathlib.Path(mesh_dir).is_absolute() else repo_root / mesh_dir
    if not urdf.exists():
        raise SystemExit(f"URDF not found: {urdf}\nHint: git submodule update --init --recursive")
    model, col_model, vis_model = pin.buildModelsFromUrdf(str(urdf), str(mesh))
    viz = MeshcatVisualizer(model, col_model, vis_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    print("MeshCat viewer ready — open the printed URL in a browser if it didn't open automatically.", file=sys.stderr)
    return viz, model


def load_prompt(args: argparse.Namespace, repo_root: pathlib.Path) -> str:
    if args.prompt:
        return args.prompt
    instructions_path = args.instructions_json
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


def action_dict_from_vector(action: list[float] | np.ndarray, *, reverse_model_joint_order: bool = False) -> dict[str, float]:
    if len(action) != STATE_DIM:
        raise RuntimeError(f"Server returned {len(action)}-D action; expected {STATE_DIM} for SO-101.")
    if reverse_model_joint_order:
        action = list(reversed(action))
    return {f"{m}.pos": float(v) for m, v in zip(MOTOR_ORDER, action)}


def fmt_vec(values: list[float] | np.ndarray, precision: int = 1) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def interpolate_actions(actions: np.ndarray, duration_s: float, rate_hz: float) -> np.ndarray:
    """Linearly interpolate an action chunk over a fixed playback duration."""
    if len(actions) <= 1 or rate_hz <= 0:
        return actions
    n_commands = max(2, int(round(duration_s * rate_hz)))
    source_t = np.linspace(0.0, duration_s, len(actions), dtype=np.float32)
    target_t = np.linspace(0.0, duration_s, n_commands, endpoint=False, dtype=np.float32)
    return np.stack(
        [np.interp(target_t, source_t, actions[:, dim]) for dim in range(actions.shape[1])],
        axis=1,
    ).astype(np.float32)


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

    viz = None
    viz_model = None
    if args.meshcat:
        viz, viz_model = setup_meshcat(args.urdf_path, args.mesh_dir, repo_root)

    robot = make_robot(args)
    robot.connect()
    print(f"Connected to SO-101 on {args.port}", file=sys.stderr)

    record_dir = args.record_dir if args.record_dir.is_absolute() else repo_root / args.record_dir
    display = DisplayRecorder(
        enabled=args.display,
        window_name=f"SO-101 [{args.camera_key}]",
        record_dir=record_dir,
        fps=args.camera_fps,
        record_key=args.record_key,
    )

    step_dt = 1.0 / args.chunk_rate_hz
    total_steps = 0
    plan_idx = 0
    prev_plan_state: np.ndarray | None = None
    prev_plan_actions: np.ndarray | None = None
    stop_flag = {"value": False}

    frame_history: collections.deque[bytes] = collections.deque(maxlen=args.img_horizon)
    last_frame_t: float = 0.0
    frame_interval: float = 1.0 / args.frame_capture_hz

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
            robot_state = state_from_obs(obs)
            state = list(reversed(robot_state)) if args.reverse_joint_order else robot_state

            # Force a fresh camera read (not the background-thread cached frame
            # from get_observation) so the last frame we send is captured as
            # close to the request as possible.
            cam = robot.cameras[args.camera_key]
            frame = cam.read()
            if frame is None:
                raise RuntimeError(f"Camera {args.camera_key!r} returned None — is the device free?")

            display.show(frame)
            jpeg = encode_frame_jpeg(frame, quality=args.jpeg_quality)

            # Record the freshly-captured frame into history just before sending
            # the request so the server always sees the most up-to-date observation.
            frame_history.append(jpeg)
            last_frame_t = time.perf_counter()
            images_bytes_list = list(frame_history) if len(frame_history) == args.img_horizon else None
            display.show_history(list(frame_history))

            t0 = time.perf_counter()
            seed = args.seed if args.fixed_seed else args.seed + plan_idx
            resp = client.infer(
                prompt=prompt,
                state=state,
                image_bytes=jpeg,
                images_bytes=images_bytes_list,
                return_full_chunk=True,
                num_sampling_step=args.num_sampling_step,
                stop_after_step=args.stop_after_step,
                seed=seed,
            )
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            actions = resp["actions"]
            actions_np = np.asarray(actions, dtype=np.float32)
            robot_actions_np = actions_np[:, ::-1] if args.reverse_joint_order else actions_np
            n_to_execute = min(args.num_execute, len(actions))
            state_np = np.asarray(robot_state, dtype=np.float32)
            state_delta = (
                float(np.max(np.abs(state_np - prev_plan_state)))
                if prev_plan_state is not None
                else None
            )
            action_delta = (
                float(np.mean(np.abs(robot_actions_np - prev_plan_actions)))
                if prev_plan_actions is not None and prev_plan_actions.shape == robot_actions_np.shape
                else None
            )
            action_step_delta = (
                float(np.mean(np.abs(np.diff(robot_actions_np, axis=0))))
                if len(robot_actions_np) > 1
                else 0.0
            )
            print(
                f"[step {total_steps}] seed={seed} state={fmt_vec(robot_state)} "
                f"-> {len(actions)} actions ({rtt_ms:.0f} ms round-trip, "
                f"server {resp['infer_ms']:.0f} ms, ran_model={resp['ran_model']}); "
                f"executing {n_to_execute}",
                file=sys.stderr,
            )
            if args.reverse_joint_order:
                print(f"  reverse-joint-order: sent model_state={fmt_vec(state)}", file=sys.stderr)
            print(
                f"  chunk first={fmt_vec(robot_actions_np[0])} last={fmt_vec(robot_actions_np[-1])} "
                f"mean_step_delta={action_step_delta:.2f}"
                + ("" if state_delta is None else f" state_delta_since_plan={state_delta:.2f}")
                + ("" if action_delta is None else f" action_delta_since_plan={action_delta:.2f}"),
                file=sys.stderr,
            )
            prev_plan_state = state_np
            prev_plan_actions = robot_actions_np
            plan_idx += 1

            # ---- Execute: replay the first N actions over their trained duration ----
            execution_actions = robot_actions_np[:n_to_execute]
            playback_duration_s = n_to_execute * step_dt
            command_rate_hz = args.interpolate_actions_hz or args.chunk_rate_hz
            if command_rate_hz > args.chunk_rate_hz and n_to_execute > 1:
                execution_actions = interpolate_actions(
                    execution_actions,
                    duration_s=playback_duration_s,
                    rate_hz=command_rate_hz,
                )
                print(
                    f"  interpolated {n_to_execute} anchors over {playback_duration_s:.1f}s "
                    f"into {len(execution_actions)} commands at {command_rate_hz:.1f} Hz",
                    file=sys.stderr,
                )
            else:
                command_rate_hz = args.chunk_rate_hz

            command_dt = 1.0 / command_rate_hz
            chunk_start = time.perf_counter()
            for i, robot_action in enumerate(execution_actions):
                if stop_flag["value"]:
                    break
                action_dict = action_dict_from_vector(robot_action)
                if args.dry_run:
                    print(f"  dry-run action[{i}] = {action_dict}", file=sys.stderr)
                else:
                    robot.send_action(action_dict)
                if viz is not None and viz_model is not None:
                    q = np.deg2rad(np.array(robot_action[:viz_model.nq]))
                    viz.display(q)

                total_steps += 1
                if args.max_steps and total_steps >= args.max_steps:
                    break

                # Pace from the start of the chunk so jitter in
                # individual send_action calls doesn't accumulate.
                target_t = chunk_start + (i + 1) * command_dt
                now = time.perf_counter()

                # Capture a frame at frame_capture_hz into the rolling history.
                if now - last_frame_t >= frame_interval:
                    try:
                        cap_obs = robot.get_observation()
                        cap_frame = cap_obs.get(args.camera_key)
                        if cap_frame is not None:
                            display.show(cap_frame)
                            frame_history.append(encode_frame_jpeg(cap_frame, quality=args.jpeg_quality))
                            last_frame_t = now
                    except Exception as exc:
                        print(f"  warn: frame capture failed: {exc}", file=sys.stderr)

                if target_t > now:
                    time.sleep(target_t - now)
    finally:
        try:
            display.close()
        except Exception as exc:  # pragma: no cover
            print(f"Warning: display.close() raised: {exc}", file=sys.stderr)
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
    p.add_argument("--reverse-joint-order", action="store_true",
                   help="Experimental: send joint state to the model in reverse order and reverse model actions back before execution.")

    # Camera
    p.add_argument("--camera-key", default="front",
                   help="Camera dict key — must match the model's training camera ('front' for this repo)")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30,
                   help="Camera capture fps. The model conditions at 5 Hz; capture fps just needs to be high enough.")
    p.add_argument("--display", action=argparse.BooleanOptionalAction, default=True,
                   help="Show the live camera feed in an OpenCV window. "
                        "Press the record key (default 'r') in the window to toggle MP4 recording.")
    p.add_argument("--record-key", default="r",
                   help="Single character that toggles recording when the camera window is focused.")
    p.add_argument("--record-dir", type=pathlib.Path, default=pathlib.Path("recordings"),
                   help="Where MP4 recordings are written (absolute, or relative to the repo root).")

    # Prompt
    p.add_argument("--prompt", default=None, help="Override prompt text (else loaded from --instructions-json by key)")
    p.add_argument("--prompt-key", default="ex1",
                   help="Key into the instructions JSON (e.g. 'ex1', 'ex2')")
    p.add_argument("--instructions-json", type=pathlib.Path,
                   default=_HERE.parent / "config" / "deployment_prompts.json",
                   help="Path to language instructions JSON")

    # Server
    p.add_argument("--server", default=None, help="Full server URL (else env MIMIC_VIDEO_SERVER or http://localhost:8000)")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds")
    p.add_argument("--num-sampling-step", type=int, default=35)
    p.add_argument("--stop-after-step", type=int, default=1,
                   help="Stop video denoising after this step (trade quality for latency). "
                        "Sent on every /infer call so you can change it without restarting the server.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base diffusion seed. By default each new chunk uses seed + chunk_index.")
    p.add_argument("--fixed-seed", action="store_true",
                   help="Reuse --seed for every chunk for reproducible debugging.")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for the frame upload (1-100)")
    p.add_argument("--img-horizon", type=int, default=5,
                   help="Number of historical frames to send per inference call (must match server's img_horizon)")
    p.add_argument("--frame-capture-hz", type=float, default=5.0,
                   help="Rate at which frames are captured into the rolling history during action execution")

    # Loop control
    p.add_argument("--chunk-rate-hz", type=float, default=5.0,
                   help="Replay rate within each action chunk (matches the 5 Hz training target_frequency)")
    p.add_argument("--interpolate-actions-hz", type=float, default=30.0,
                   help="Command rate for linear interpolation between chunk actions. "
                        "Set 0 or <= --chunk-rate-hz to send only the raw 5 Hz actions.")
    p.add_argument("--num-execute", type=int, default=15,
                   help="How many of the 15-step chunk to execute before re-planning. Lower = more re-plans, "
                        "more idle waits during inference; higher = less reactive.")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after this many total actions sent. 0 = unlimited (Ctrl-C to stop).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions but don't send them to the robot.")

    # Visualization
    p.add_argument("--meshcat", action="store_true",
                   help="Open a MeshCat browser viewer and display each action as it executes.")
    p.add_argument("--urdf-path", default=_DEFAULT_URDF,
                   help="Path to SO-101 URDF (absolute or relative to repo root).")
    p.add_argument("--mesh-dir", default=_DEFAULT_MESH_DIR,
                   help="Mesh directory for the URDF (absolute or relative to repo root).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_relative_target is not None and args.max_relative_target <= 0:
        args.max_relative_target = None
    repo_root = _HERE.parent
    run_loop(args, repo_root)


if __name__ == "__main__":
    main()

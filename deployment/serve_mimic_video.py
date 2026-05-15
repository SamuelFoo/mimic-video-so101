"""HTTP/WebSocket inference server for the full mimic-video pipeline.

This server is wired for the LeRobot action space used by this project:
both state and action are 6-D absolute joint angles (degrees) for SO-ARM-101,
sampled at 5 Hz. The world2action pipeline's normalizer denormalizes
internally using the dataset statistics JSON, so the floats returned by the
server are directly executable joint targets — no per-environment rotation/
delta conversion is needed (that logic only applies to the libero/bridge
checkpoints).

Loads the Video2World2ActionPipeline once at startup and serves it over a
FastAPI app. Laptops can talk to it from anywhere reachable on the network:
they ship a JPEG/PNG-compressed frame plus the current 6-D joint state plus
a task prompt, and get back the next 15-step action chunk.

Run via deployment/serve_mimic_video.sh (which sets the model venv + CUDA libs).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import io
import json
import logging
import pathlib
import threading
import time
import uuid
from typing import Any

import einops
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

log = logging.getLogger("mimic-video-server")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )
    action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)
    with dataset_statistics_path.open("rb") as f:
        stats = json.load(f)
    action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=dtype,
    )

    return Video2World2ActionPipeline(video_pipe, action_pipe).cuda()


# ---------------------------------------------------------------------------
# Inference session (rolling image + state history, action buffer)
# ---------------------------------------------------------------------------

class InferenceSession:
    """Holds rolling image/state history and a cached action buffer for one episode."""

    def __init__(
        self,
        img_horizon: int,
        lowdim_horizon: int,
        frame_stride: int,
        resize_hw: tuple[int, int],
    ):
        self.img_horizon = img_horizon
        self.lowdim_horizon = lowdim_horizon
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw
        # When stride > 1 we need stride*(img_horizon-1)+1 raw frames in flight.
        self._raw_image_maxlen = frame_stride * (img_horizon - 1) + 1
        self._image_history: collections.deque[np.ndarray] = collections.deque(maxlen=self._raw_image_maxlen)
        self._lowdim_history: collections.deque[np.ndarray] = collections.deque(maxlen=lowdim_horizon)
        self.prompt: str = ""
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx: int = 0

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self._image_history.clear()
        self._lowdim_history.clear()
        self.action_buffer = None
        self.action_buffer_idx = 0

    def add_image(self, frame_chw_1hw: np.ndarray) -> None:
        self._image_history.append(frame_chw_1hw)

    def add_state(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self.lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())

    def build_input_video(self) -> np.ndarray:
        """Return (C, img_horizon, H, W) by sampling history with frame_stride."""
        if not self._image_history:
            raise ValueError("No frames in history; call /reset and send at least one frame.")
        if len(self._image_history) >= self._raw_image_maxlen:
            picked = list(self._image_history)[:: self.frame_stride][-self.img_horizon :]
        else:
            # Not enough history yet: repeat the most recent frame.
            picked = [self._image_history[-1]] * self.img_horizon
        return np.concatenate(picked, axis=1)

    def build_input_state(self) -> np.ndarray:
        if not self._lowdim_history:
            raise ValueError("No state in history; send a state vector first.")
        return np.stack(list(self._lowdim_history), axis=0)


# ---------------------------------------------------------------------------
# Image preprocessing (matches eval/bridge VAMInference._process_image)
# ---------------------------------------------------------------------------

def decode_and_preprocess(image_bytes: bytes, resize_hw: tuple[int, int]) -> np.ndarray:
    """JPEG/PNG bytes → (C, 1, H, W) float32 in [-1, 1]."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = transforms.Resize(resize_hw)(pil)
    arr = np.asarray(pil, dtype=np.uint8)
    chw = einops.rearrange(arr, "h w c -> c h w")[:, None, :, :]
    return 2.0 * (chw.astype(np.float32) / 255.0 - 0.5)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    prompt: str


class InferRequest(BaseModel):
    """JSON body for /infer when sending base64-encoded frames."""

    prompt: str
    state: list[float]
    image_b64: str  # base64-encoded JPEG or PNG
    return_full_chunk: bool = True
    num_sampling_step: int = 35
    stop_after_step: int | None = None
    seed: int = 0


def make_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="mimic-video inference", version="0.1.0")

    # The pipeline is loaded lazily on first request so the server can boot
    # quickly enough for healthchecks; in practice we load it in startup.
    state: dict[str, Any] = {
        "pipeline": None,
        "session": InferenceSession(
            img_horizon=args.img_horizon,
            lowdim_horizon=args.lowdim_horizon,
            frame_stride=args.frame_stride,
            resize_hw=(args.resize_h, args.resize_w),
        ),
        # Serialize GPU calls — one 2B-DiT forward at a time per process.
        "gpu_lock": threading.Lock(),
    }

    @app.on_event("startup")
    def _load_model() -> None:
        log.info("Loading pipeline (this can take ~1 min)...")
        t0 = time.time()
        state["pipeline"] = load_pipeline(
            experiment_name=args.experiment_name,
            video_model_path=args.video_model_path,
            action_model_path=args.action_model_path,
            dataset_statistics_path=pathlib.Path(args.dataset_statistics_path),
        )
        log.info(f"Pipeline ready in {time.time() - t0:.1f}s on cuda:0.")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML.format(
            img_h=args.img_horizon,
            lowdim_h=args.lowdim_horizon,
            stride=args.frame_stride,
            exp=args.experiment_name,
        )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": state["pipeline"] is not None,
            "experiment": args.experiment_name,
            "img_horizon": args.img_horizon,
            "lowdim_horizon": args.lowdim_horizon,
            "frame_stride": args.frame_stride,
            "resize_hw": [args.resize_h, args.resize_w],
            "session_frames": len(state["session"]._image_history),
            "session_states": len(state["session"]._lowdim_history),
            "buffered_actions": (
                0 if state["session"].action_buffer is None
                else int(state["session"].action_buffer.shape[0] - state["session"].action_buffer_idx)
            ),
        }

    @app.post("/reset")
    def reset(req: ResetRequest) -> dict[str, Any]:
        state["session"].reset(req.prompt)
        log.info(f"Reset session with prompt: {req.prompt!r}")
        return {"ok": True}

    @app.post("/infer")
    def infer_json(req: InferRequest) -> JSONResponse:
        if state["pipeline"] is None:
            raise HTTPException(503, "Pipeline still loading")
        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception as exc:
            raise HTTPException(400, f"Bad image_b64: {exc}") from exc
        return JSONResponse(_run_step(
            state, args, image_bytes, np.asarray(req.state, dtype=np.float32),
            prompt=req.prompt,
            return_full_chunk=req.return_full_chunk,
            num_sampling_step=req.num_sampling_step,
            stop_after_step=req.stop_after_step,
            seed=req.seed,
        ))

    @app.post("/infer_multipart")
    async def infer_multipart(
        prompt: str,
        state_json: str,
        frame: UploadFile,
        return_full_chunk: bool = True,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
    ) -> JSONResponse:
        """Alternative endpoint: upload the frame as multipart instead of base64."""
        if state["pipeline"] is None:
            raise HTTPException(503, "Pipeline still loading")
        image_bytes = await frame.read()
        state_vec = np.asarray(json.loads(state_json), dtype=np.float32)
        return JSONResponse(_run_step(
            state, args, image_bytes, state_vec,
            prompt=prompt,
            return_full_chunk=return_full_chunk,
            num_sampling_step=num_sampling_step,
            stop_after_step=stop_after_step,
            seed=seed,
        ))

    @app.websocket("/ws")
    async def ws_infer(ws: WebSocket) -> None:
        await ws.accept()
        peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
        log.info(f"WS client connected: {peer}")
        try:
            while True:
                msg = await ws.receive_text()
                req = json.loads(msg)
                op = req.get("op", "infer")
                if op == "reset":
                    state["session"].reset(req["prompt"])
                    await ws.send_text(json.dumps({"ok": True, "op": "reset"}))
                    continue
                if op != "infer":
                    await ws.send_text(json.dumps({"ok": False, "error": f"unknown op {op!r}"}))
                    continue
                if state["pipeline"] is None:
                    await ws.send_text(json.dumps({"ok": False, "error": "pipeline still loading"}))
                    continue
                image_bytes = base64.b64decode(req["image_b64"])
                state_vec = np.asarray(req["state"], dtype=np.float32)
                # Run blocking inference in a thread so we don't stall the event loop.
                result = await asyncio.to_thread(
                    _run_step,
                    state, args, image_bytes, state_vec,
                    prompt=req["prompt"],
                    return_full_chunk=req.get("return_full_chunk", True),
                    num_sampling_step=req.get("num_sampling_step", 35),
                    stop_after_step=req.get("stop_after_step"),
                    seed=req.get("seed", 0),
                )
                await ws.send_text(json.dumps(result))
        except WebSocketDisconnect:
            log.info(f"WS client disconnected: {peer}")

    return app


# ---------------------------------------------------------------------------
# Core inference step (shared between REST and WebSocket)
# ---------------------------------------------------------------------------

def _run_step(
    state: dict[str, Any],
    args: argparse.Namespace,
    image_bytes: bytes,
    state_vec: np.ndarray,
    *,
    prompt: str,
    return_full_chunk: bool,
    num_sampling_step: int,
    stop_after_step: int | None,
    seed: int,
) -> dict[str, Any]:
    session: InferenceSession = state["session"]
    pipeline: Video2World2ActionPipeline = state["pipeline"]

    if state_vec.ndim != 1:
        raise HTTPException(400, f"state must be a 1-D vector, got shape {state_vec.shape}")
    if args.expected_state_dim is not None and state_vec.shape[0] != args.expected_state_dim:
        raise HTTPException(
            400,
            f"state must be {args.expected_state_dim}-D (SO-ARM-101 joint angles in degrees), "
            f"got {state_vec.shape[0]}",
        )

    if prompt != session.prompt:
        log.info(f"Prompt changed → resetting session: {prompt!r}")
        session.reset(prompt)

    frame_chw_1hw = decode_and_preprocess(image_bytes, session.resize_hw)
    session.add_image(frame_chw_1hw)
    session.add_state(state_vec)

    request_id = uuid.uuid4().hex[:8]

    # If we still have unconsumed actions, return the next one without re-running the model.
    if session.action_buffer is not None and session.action_buffer_idx < session.action_buffer.shape[0]:
        return _serialize_action_response(session, request_id, ran_model=False, infer_ms=0.0, return_full_chunk=return_full_chunk)

    # Run the model (heavy: ~seconds of GPU work).
    images = session.build_input_video()  # (C, img_horizon, H, W)
    lowdims = session.build_input_state()  # (lowdim_horizon, state_dim)

    input_vid = torch.from_numpy(images[None]).cuda().bfloat16()
    state_tensor = torch.from_numpy(lowdims[None]).cuda().bfloat16()

    t0 = time.time()
    with state["gpu_lock"]:
        with torch.no_grad():
            pred_actions = pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=prompt,
                num_sampling_step=num_sampling_step,
                stop_after_step=stop_after_step if stop_after_step is not None else args.stop_after_step,
                seed=seed,
                use_cuda_graphs=args.use_cuda_graphs,
            )
    infer_ms = (time.time() - t0) * 1000.0

    session.action_buffer = pred_actions[0].float().cpu().numpy()
    session.action_buffer_idx = 0
    log.info(f"[{request_id}] inference {infer_ms:.0f}ms → chunk {session.action_buffer.shape}")

    return _serialize_action_response(session, request_id, ran_model=True, infer_ms=infer_ms, return_full_chunk=return_full_chunk)


def _serialize_action_response(
    session: InferenceSession,
    request_id: str,
    *,
    ran_model: bool,
    infer_ms: float,
    return_full_chunk: bool,
) -> dict[str, Any]:
    assert session.action_buffer is not None
    if return_full_chunk:
        actions = session.action_buffer[session.action_buffer_idx :].tolist()
        # Mark the whole chunk as consumed so the next /infer triggers a re-plan.
        session.action_buffer_idx = session.action_buffer.shape[0]
    else:
        actions = [session.action_buffer[session.action_buffer_idx].tolist()]
        session.action_buffer_idx += 1
    return {
        "ok": True,
        "request_id": request_id,
        "ran_model": ran_model,
        "infer_ms": infer_ms,
        "actions": actions,
        "action_dim": session.action_buffer.shape[1],
        "remaining_in_buffer": int(session.action_buffer.shape[0] - session.action_buffer_idx),
    }


# ---------------------------------------------------------------------------
# A tiny landing page so you know the server is up
# ---------------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<html><head><title>mimic-video inference</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:820px;margin:32px auto;padding:0 16px;color:#222}}
code,pre{{background:#f4f4f4;padding:2px 6px;border-radius:4px}}
pre{{padding:12px;overflow-x:auto}}
</style></head>
<body>
<h1>mimic-video inference server (LeRobot)</h1>
<p>Experiment: <code>{exp}</code> &middot; img_horizon=<code>{img_h}</code>
   &middot; lowdim_horizon=<code>{lowdim_h}</code> &middot; frame_stride=<code>{stride}</code></p>

<p><strong>Action space.</strong> State and actions are 6-D absolute joint angles in <em>degrees</em>
(SO-ARM-101: 5 arm joints + gripper). The world2action pipeline applies the dataset-statistics
normalizer internally, so the floats returned here are directly executable joint targets — feed them
straight to your robot. The chunk has 15 timesteps at the policy's target frequency (5 Hz).</p>

<h2>POST /reset</h2>
<pre>{{ "prompt": "pick up the red block" }}</pre>
<p>Clears image/state history and pins the task prompt. Also auto-triggered if <code>prompt</code> changes mid-session.</p>

<h2>POST /infer  (application/json)</h2>
<pre>{{
  "prompt": "pick up the red block",
  "state": [j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, gripper_deg],
  "image_b64": "...base64 JPEG/PNG of the workspace_rgb camera, will be resized to 480x640...",
  "return_full_chunk": true
}}</pre>
<p>Returns:</p>
<pre>{{
  "ok": true,
  "ran_model": true,
  "infer_ms": 5400.0,
  "actions": [[j1, j2, j3, j4, j5, gripper], ...],   // 15 rows of 6 joint-angle (deg) targets
  "action_dim": 6,
  "remaining_in_buffer": 0
}}</pre>
<p>The model only re-runs when the action buffer is exhausted; intermediate calls return cached
actions instantly. Set <code>return_full_chunk=false</code> if you'd rather pull one action at a time.</p>

<h2>POST /infer_multipart</h2>
<p>Same as <code>/infer</code> but uploads the frame as multipart (<code>frame</code>) and the state as a JSON
string (<code>state_json</code>). Useful from <code>curl -F</code>:</p>
<pre>curl -F prompt='pick up the red block' \\
     -F state_json='[0,-90,90,0,0,30]' \\
     -F frame=@frame.jpg \\
     http://&lt;host&gt;:8000/infer_multipart</pre>

<h2>WebSocket /ws</h2>
<pre>{{ "op": "reset", "prompt": "..." }}
{{ "op": "infer", "prompt": "...", "state": [...6 floats...], "image_b64": "..." }}</pre>

<h2>GET /healthz</h2>
<p>Returns load status + current session counters.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mimic-video inference server")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--video-model-path", required=True)
    p.add_argument("--action-model-path", required=True)
    p.add_argument("--dataset-statistics-path", required=True)
    p.add_argument("--img-horizon", type=int, default=5)
    p.add_argument("--lowdim-horizon", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Keep every Nth raw frame for conditioning. Use 1 if the client already streams at "
                        "the policy's conditioning fps (5 Hz for the LeRobot run). Raise it if your camera "
                        "runs faster and you want server-side downsampling (e.g. 6 for a 30 Hz feed).")
    p.add_argument("--resize-h", type=int, default=480)
    p.add_argument("--resize-w", type=int, default=640)
    p.add_argument("--stop-after-step", type=int, default=None,
                   help="Stop video denoising after this step (saves time at small quality cost).")
    p.add_argument("--expected-state-dim", type=int, default=6,
                   help="Validate inbound state vectors have this dim. 6 for SO-ARM-101. "
                        "Pass 0 (or any non-positive value) to disable the check.")
    p.add_argument("--use-cuda-graphs", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.expected_state_dim is not None and args.expected_state_dim <= 0:
        args.expected_state_dim = None
    app = make_app(args)
    log.info(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

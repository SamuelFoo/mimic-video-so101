"""HTTP inference server for the full mimic-video pipeline.

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
import base64
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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

_NET_OVERRIDE_PREFIXES = ("world2action_pipe.net", "model.config.pipe_config.net")
_SKIP_NET_KEYS = {"_target_", "sac_config"}  # _target_ is a class import; sac_config is a struct


def _action_net_overrides(action_config_path: pathlib.Path) -> list[str]:
    """Read `world2action_pipe.net.*` from the checkpoint's config.yaml and
    return hydra-style override strings so the instantiated action decoder
    matches the saved checkpoint's architecture (model_channels, num_blocks,
    pair_timestep_feature_rank, etc.). Future arch tweaks land in the yaml
    automatically — no need to touch this server."""
    import yaml

    # The training-time dump embeds `!!python/object/...` tags (e.g.
    # TextEncoderClass) that SafeLoader rejects. We only need the scalar values
    # under world2action_pipe.net, so silently ignore any Python-tagged node.
    class _IgnorePyTagsLoader(yaml.SafeLoader):
        pass

    _IgnorePyTagsLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda loader, suffix, node: None,
    )

    with action_config_path.open() as f:
        saved = yaml.load(f, Loader=_IgnorePyTagsLoader)
    net_cfg = saved.get("world2action_pipe", {}).get("net", {})
    overrides: list[str] = []
    for key, value in net_cfg.items():
        if key in _SKIP_NET_KEYS or not isinstance(value, (str, int, float, bool)):
            continue
        for prefix in _NET_OVERRIDE_PREFIXES:
            overrides.append(f"{prefix}.{key}={value}")
    return overrides


def load_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    action_config_path: pathlib.Path | None,
    pipeline_state_t: int,
    xattn_video_prefix_length: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    config = make_config()
    overrides = ["--", f"experiment={experiment_name}"]
    if action_config_path is not None and action_config_path.exists():
        net_overrides = _action_net_overrides(action_config_path)
        log.info(f"Applying {len(net_overrides) // len(_NET_OVERRIDE_PREFIXES)} "
                 f"action-net overrides from {action_config_path}")
        overrides.extend(net_overrides)
    else:
        raise FileNotFoundError(
            f"No action config.yaml found at {action_config_path}. "
            "Copy the config.yaml from your training run next to the action checkpoint, "
            "or set ACTION_CONFIG_PATH=/path/to/config.yaml."
        )
    # The action ckpt was trained with a specific xattn_video_prefix_length;
    # if it doesn't match here, cross-attn sees a KV set the decoder wasn't
    # trained on. Same paths the train script overrides.
    overrides.extend([
        f"world2action_pipe.xattn_video_prefix_length={xattn_video_prefix_length}",
        f"model.config.pipe_config.xattn_video_prefix_length={xattn_video_prefix_length}",
    ])
    config = override(config, overrides)
    config.model.config.video_pipe_config.guardrail_config.enabled = False
    # The video DiT was trained for a specific state_t; the pipeline config
    # default (config_video2world.py) may have moved since. Pin it here to
    # match the checkpoint, otherwise denoising state_shape and DiT positional
    # embeddings disagree.
    config.model.config.video_pipe_config.state_t = pipeline_state_t

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
# Inference session (cached action buffer for receding-horizon re-plan)
# ---------------------------------------------------------------------------

class InferenceSession:
    """Caches the most recent action chunk so callers can drain it across
    successive /infer calls without re-running the diffusion model.

    State history is NOT cached here — it's sent by the client on every call
    (matching the per-call `images_b64` pattern). This avoids the pitfall where
    a client restart with the same prompt leaves stale per-step state in the
    server's deque (the prompt-change branch is the only reset trigger)."""

    def __init__(self, resize_hw: tuple[int, int]):
        self.resize_hw = resize_hw
        self.prompt: str = ""
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx: int = 0

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self.action_buffer = None
        self.action_buffer_idx = 0


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

class InferRequest(BaseModel):
    """JSON body for /infer when sending base64-encoded frames."""

    prompt: str
    state: list[float]
    image_b64: str  # base64-encoded JPEG or PNG (most recent frame)
    images_b64: list[str] | None = None  # optional: full frame history oldest→newest; bypasses single-frame fallback
    states: list[list[float]] | None = None  # optional: full state history oldest→newest; bypasses single-state fallback
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
        "session": InferenceSession(resize_hw=(args.resize_h, args.resize_w)),
        # Serialize GPU calls — one 2B-DiT forward at a time per process.
        "gpu_lock": threading.Lock(),
    }

    @app.on_event("startup")
    def _load_model() -> None:
        log.info("Loading pipeline (this can take ~1 min)...")
        t0 = time.time()
        action_config_path = (
            pathlib.Path(args.action_config_path) if args.action_config_path
            else pathlib.Path(args.action_model_path).parent / "config.yaml"
        )
        state["pipeline"] = load_pipeline(
            experiment_name=args.experiment_name,
            video_model_path=args.video_model_path,
            action_model_path=args.action_model_path,
            dataset_statistics_path=pathlib.Path(args.dataset_statistics_path),
            action_config_path=action_config_path,
            pipeline_state_t=args.pipeline_state_t,
            xattn_video_prefix_length=args.xattn_video_prefix_length,
        )
        log.info(f"Pipeline ready in {time.time() - t0:.1f}s on cuda:0.")

    @app.post("/infer")
    def infer_json(req: InferRequest) -> JSONResponse:
        if state["pipeline"] is None:
            raise HTTPException(503, "Pipeline still loading")
        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception as exc:
            raise HTTPException(400, f"Bad image_b64: {exc}") from exc
        all_images_bytes = None
        if req.images_b64 is not None:
            try:
                all_images_bytes = [base64.b64decode(b) for b in req.images_b64]
            except Exception as exc:
                raise HTTPException(400, f"Bad images_b64: {exc}") from exc
        return JSONResponse(_run_step(
            state, args, image_bytes, np.asarray(req.state, dtype=np.float32),
            prompt=req.prompt,
            return_full_chunk=req.return_full_chunk,
            num_sampling_step=req.num_sampling_step,
            stop_after_step=req.stop_after_step,
            seed=req.seed,
            all_images_bytes=all_images_bytes,
            all_states=req.states,
        ))

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
    all_images_bytes: list[bytes] | None = None,
    all_states: list[list[float]] | None = None,
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

    request_id = uuid.uuid4().hex[:8]

    # If we still have unconsumed actions, return the next one without re-running the model.
    if session.action_buffer is not None and session.action_buffer_idx < session.action_buffer.shape[0]:
        return _serialize_action_response(
            session, request_id, ran_model=False, infer_ms=0.0,
            return_full_chunk=return_full_chunk,
            num_execute_actions=args.num_execute_actions,
        )

    # Run the model (heavy: ~seconds of GPU work).
    if all_images_bytes is not None:
        if len(all_images_bytes) != args.img_horizon:
            raise HTTPException(
                400,
                f"images_b64 must have exactly {args.img_horizon} frames, got {len(all_images_bytes)}",
            )
        frames = [decode_and_preprocess(b, session.resize_hw) for b in all_images_bytes]
    else:
        # Single-frame fallback (e.g. CLI client): repeat to fill the horizon.
        single = decode_and_preprocess(image_bytes, session.resize_hw)
        frames = [single] * args.img_horizon
    images = np.concatenate(frames, axis=1)  # (C, img_horizon, H, W)

    # State history: client-supplied (oldest→newest) when present, otherwise
    # repeat the single most-recent state to fill the horizon. Mirrors the
    # images_b64 fallback so a one-shot CLI request still works.
    if all_states is not None:
        if len(all_states) != args.lowdim_horizon:
            raise HTTPException(
                400,
                f"states must have exactly {args.lowdim_horizon} entries, got {len(all_states)}",
            )
        lowdims_list = []
        for i, s in enumerate(all_states):
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim != 1:
                raise HTTPException(400, f"states[{i}] must be 1-D, got shape {arr.shape}")
            if args.expected_state_dim is not None and arr.shape[0] != args.expected_state_dim:
                raise HTTPException(
                    400,
                    f"states[{i}] must be {args.expected_state_dim}-D, got {arr.shape[0]}",
                )
            lowdims_list.append(arr)
        lowdims = np.stack(lowdims_list, axis=0)
    else:
        lowdims = np.stack([state_vec] * args.lowdim_horizon, axis=0)

    input_vid = torch.from_numpy(images[None]).cuda().bfloat16()
    state_tensor = torch.from_numpy(lowdims[None]).cuda().bfloat16()

    # Video2World2ActionPipeline requires stop_after_step to be set — the video
    # backbone only returns the (crossattn_emb, video_sigma) tuple at that step.
    # Falls back to num_sampling_step - 1 (near-fully-denoised, last in-loop step):
    # the post-loop "clean pass" branch in video2world.generate_video returns a
    # 0-D sigma_min that crashes the caller's `.unsqueeze(1)`. The in-loop
    # branch returns a 1-D sigma, which works. Lower it to trade quality for
    # latency.
    effective_stop = stop_after_step if stop_after_step is not None else max(0, num_sampling_step - 1)

    t0 = time.time()
    with state["gpu_lock"]:
        with torch.no_grad():
            pred_actions = pipeline(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=prompt,
                num_sampling_step=num_sampling_step,
                stop_after_step=effective_stop,
                seed=seed,
                use_cuda_graphs=args.use_cuda_graphs,
            )
    infer_ms = (time.time() - t0) * 1000.0

    session.action_buffer = pred_actions[0].float().cpu().numpy()
    session.action_buffer_idx = 0
    log.info(f"[{request_id}] inference {infer_ms:.0f}ms → chunk {session.action_buffer.shape}")

    if args.save_videos:
        with state["gpu_lock"]:
            video_path = _save_future_video(
                pipeline, input_vid,
                prompt=prompt,
                num_sampling_step=num_sampling_step,
                stop_after_step=effective_stop,
                seed=seed,
                use_cuda_graphs=args.use_cuda_graphs,
                out_dir=pathlib.Path(args.video_dir),
                fps=args.video_fps,
                request_id=request_id,
            )
        log.info(f"[{request_id}] saved future video → {video_path}")

    return _serialize_action_response(
        session, request_id, ran_model=True, infer_ms=infer_ms,
        return_full_chunk=return_full_chunk,
        num_execute_actions=args.num_execute_actions,
    )


def _decode_video_at_step(
    video_pipe,
    input_vid: torch.Tensor,
    *,
    num_latent_conditional_frames: int,
    prompt: str,
    num_sampling_step: int,
    stop_after_step: int,
    seed: int,
    use_cuda_graphs: bool,
) -> torch.Tensor:
    """Replicate `Video2WorldPipeline.generate_video` but return the decoded
    video derived from `x0_pred` at iteration `stop_after_step` — i.e. the same
    partially-denoised state the action decoder consumed. With `stop_after_step
    == num_sampling_step - 1` this is very close to the fully denoised output;
    with smaller values you get progressively noisier predictions."""
    from imaginaire.utils import misc

    data_batch = video_pipe._get_data_batch_input(
        input_vid, prompt, None, "", num_latent_conditional_frames=num_latent_conditional_frames,
    )
    video_pipe._normalize_video_databatch_inplace(data_batch)
    video_pipe._augment_image_dim_inplace(data_batch)
    input_key = video_pipe.input_image_key if video_pipe.is_image_batch(data_batch) else video_pipe.input_video_key
    n_sample = data_batch[input_key].shape[0]
    _T, _H, _W = data_batch[input_key].shape[-3:]
    state_shape = [
        video_pipe.config.state_ch,
        video_pipe.config.state_t,
        _H // video_pipe.tokenizer.spatial_compression_factor,
        _W // video_pipe.tokenizer.spatial_compression_factor,
    ]
    x0_fn = video_pipe.get_x0_fn_from_batch(
        data_batch, guidance=0.0, is_negative_prompt=True, use_cuda_graphs=use_cuda_graphs,
    )
    sample = (
        misc.arch_invariant_rand(
            (n_sample, *tuple(state_shape)), torch.float32, video_pipe.tensor_kwargs["device"], seed,
        )
        * video_pipe.scheduler.config.sigma_max
    ).to(dtype=torch.float32)
    video_pipe.scheduler.set_timesteps(num_sampling_step, device=sample.device)
    x0_prev = None
    last_x0_pred = None
    for i, _ in enumerate(video_pipe.scheduler.timesteps):
        sigma_t = video_pipe.scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_t.repeat(sample.shape[0])
        x0_pred = x0_fn(sample, sigma_in, return_only_hidden_states_up_to=None)
        last_x0_pred = x0_pred
        if i == stop_after_step:
            break
        sample, x0_prev = video_pipe.scheduler.step(
            x0_pred=x0_pred, i=i, sample=sample, x0_prev=x0_prev,
        )
    return video_pipe.decode(last_x0_pred)


def _save_future_video(
    pipeline: Video2World2ActionPipeline,
    input_vid: torch.Tensor,
    *,
    prompt: str,
    num_sampling_step: int,
    stop_after_step: int,
    seed: int,
    use_cuda_graphs: bool,
    out_dir: pathlib.Path,
    fps: int,
    request_id: str,
) -> pathlib.Path:
    """Decode the video at the same `stop_after_step` the action decoder used,
    so the MP4 reflects what the action decoder actually conditioned on.
    Roughly doubles per-call latency vs. action-only inference."""
    import imageio.v2 as imageio

    T = input_vid.shape[2]
    with torch.no_grad():
        video = _decode_video_at_step(
            pipeline.video2world_pipeline,
            input_vid,
            num_latent_conditional_frames=1 if T == 1 else 2,
            prompt=prompt,
            num_sampling_step=num_sampling_step,
            stop_after_step=stop_after_step,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )

    frames = video[0].float().clamp(-1.0, 1.0).add(1.0).mul(127.5).byte().cpu().numpy()
    frames = einops.rearrange(frames, "c t h w -> t h w c")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{request_id}.mp4"
    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return out_path


def _serialize_action_response(
    session: InferenceSession,
    request_id: str,
    *,
    ran_model: bool,
    infer_ms: float,
    return_full_chunk: bool,
    num_execute_actions: int,
) -> dict[str, Any]:
    """Slice the cached action buffer for the response and trigger a re-plan
    after `num_execute_actions` have been consumed — receding-horizon, matches
    eval/libero/run.py and eval/bridge/.../video_action_model.py
    (`num_execute_actions=5`).
    """
    assert session.action_buffer is not None
    chunk_len = session.action_buffer.shape[0]
    execute_cap = min(num_execute_actions, chunk_len)
    if return_full_chunk:
        actions = session.action_buffer[session.action_buffer_idx : execute_cap].tolist()
        # Force a re-plan on the next call.
        session.action_buffer_idx = chunk_len
    else:
        actions = [session.action_buffer[session.action_buffer_idx].tolist()]
        session.action_buffer_idx += 1
        if session.action_buffer_idx >= execute_cap:
            # Mark buffer fully consumed so the next /infer re-plans, even
            # though chunk has more steps. This is the receding-horizon trick.
            session.action_buffer_idx = chunk_len
    return {
        "ok": True,
        "request_id": request_id,
        "ran_model": ran_model,
        "infer_ms": infer_ms,
        "actions": actions,
        "action_dim": session.action_buffer.shape[1],
        "remaining_in_buffer": int(chunk_len - session.action_buffer_idx),
    }


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
    p.add_argument("--lowdim-horizon", type=int, default=1,
                   help="Number of low-dim state entries the action decoder expects per call. "
                        "Clients can either send a 'states' list of exactly this length "
                        "(oldest→newest), or send a single 'state' that the server repeats "
                        "to fill the horizon.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Keep every Nth raw frame for conditioning. Use 1 if the client already streams at "
                        "the policy's conditioning fps (5 Hz for the LeRobot run). Raise it if your camera "
                        "runs faster and you want server-side downsampling (e.g. 6 for a 30 Hz feed).")
    p.add_argument("--resize-h", type=int, default=480)
    p.add_argument("--resize-w", type=int, default=640)
    p.add_argument("--expected-state-dim", type=int, default=6,
                   help="Validate inbound state vectors have this dim. 6 for SO-ARM-101. "
                        "Pass 0 (or any non-positive value) to disable the check.")
    p.add_argument("--use-cuda-graphs", action="store_true")
    p.add_argument("--num-execute-actions", type=int, default=5,
                   help="Receding horizon: only this many actions per 15-step chunk are exposed "
                        "to the client before forcing a re-plan. Matches eval/libero/run.py and "
                        "eval/bridge/.../video_action_model.py defaults.")
    p.add_argument("--save-videos", action="store_true",
                   help="On every /infer call, ALSO run the video2world backbone with full "
                        "decoding and write an MP4 of the model's predicted future to --video-dir. "
                        "Doubles per-call latency; intended for debugging the world model.")
    p.add_argument("--video-dir", default="./visualizations",
                   help="Where --save-videos writes MP4s.")
    p.add_argument("--video-fps", type=int, default=5,
                   help="MP4 framerate. Matches the policy's 5 Hz conditioning by default.")
    p.add_argument("--action-config-path", default=None,
                   help="Path to the training-time config.yaml saved alongside the action checkpoint. "
                        "world2action_pipe.net.* values from this file override the experiment-default "
                        "architecture so any future arch change (model_channels, num_blocks, ...) is picked "
                        "up automatically. Defaults to <action-model-path's dir>/config.yaml.")
    p.add_argument("--pipeline-state-t", type=int, required=True,
                   help="Number of latent timesteps the video DiT emits. Must equal the state_t the "
                        "action checkpoint was trained against — the config_video2world.py default may "
                        "have moved since the checkpoint was trained.")
    p.add_argument("--xattn-video-prefix-length", default="null",
                   help="Slice the video DiT's hidden states to the first N latent timesteps before "
                        "cross-attn into the action decoder. Must match training. Use 'null' for no "
                        "slicing (action decoder sees all state_t positions).")
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

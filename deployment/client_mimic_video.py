"""Client for the mimic-video inference server.

Usable two ways:

1. **Library** — import `MimicVideoClient` from a robot control loop and call
   `client.infer(prompt, state, image_bytes)` once per chunk.
2. **CLI** — run this script directly to send a single frame + state to the
   server and print the returned 15-step action chunk. Handy for sanity-checking
   that the SSH tunnel / VPN to your GPU box is wired up correctly.

## Configuring the server URL

Three ways, in priority order (most specific wins):
1. `--server http://host:port` on the CLI
2. `--host` + `--port` flags
3. `MIMIC_VIDEO_SERVER=http://host:port` env var
4. Fallback to `http://localhost:8000` (matches the server default)

The typical laptop setup is an SSH tunnel:

    ssh -L 8000:localhost:8000 user@gpu-box
    # then on the laptop:
    python deployment/client_mimic_video.py \\
        --image frame.jpg \\
        --state '[0,-90,90,0,0,30]' \\
        --prompt 'pick up the red block'

Dependencies on the laptop: just `requests` (`pip install requests`). No torch,
no cosmos, no model env needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
from typing import Any

import requests

DEFAULT_SERVER = os.environ.get("MIMIC_VIDEO_SERVER", "http://localhost:8000")
DEFAULT_TIMEOUT_S = 600.0  # the 2B DiT can take a few seconds per chunk


class MimicVideoClient:
    """Thin HTTP client wrapping /healthz, /reset, /infer."""

    def __init__(self, server: str = DEFAULT_SERVER, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.server = server.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        r = self._session.get(f"{self.server}/healthz", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, prompt: str) -> None:
        r = self._session.post(
            f"{self.server}/reset",
            json={"prompt": prompt},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def infer(
        self,
        prompt: str,
        state: list[float] | tuple[float, ...],
        image_bytes: bytes,
        *,
        return_full_chunk: bool = True,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """POST a single frame + state to the server; return the action response.

        `state` should be a 6-D list of SO-ARM-101 joint angles in degrees.
        `image_bytes` should be raw JPEG/PNG bytes (the server will decode and
        resize to 480x640 itself).
        """
        body = {
            "prompt": prompt,
            "state": list(state),
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "return_full_chunk": return_full_chunk,
            "num_sampling_step": num_sampling_step,
            "stop_after_step": stop_after_step,
            "seed": seed,
        }
        r = self._session.post(
            f"{self.server}/infer",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


def _resolve_server(args: argparse.Namespace) -> str:
    if args.server:
        return args.server
    if args.host or args.port:
        host = args.host or "localhost"
        port = args.port or 8000
        return f"http://{host}:{port}"
    return DEFAULT_SERVER


def _parse_state(raw: str) -> list[float]:
    parsed = json.loads(raw)
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError(f"--state must be a JSON list, got {type(parsed).__name__}")
    return [float(x) for x in parsed]


def _read_image_bytes(path: pathlib.Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage notes.",
    )
    p.add_argument("--server", help=f"Full server URL (default: env MIMIC_VIDEO_SERVER or {DEFAULT_SERVER})")
    p.add_argument("--host", help="Server hostname/IP (overridden by --server)")
    p.add_argument("--port", type=int, help="Server port (overridden by --server)")
    p.add_argument("--image", type=pathlib.Path, required=True, help="JPEG/PNG file to send as the workspace frame")
    p.add_argument("--state", type=_parse_state, required=True,
                   help="JSON list of 6 joint angles in degrees, e.g. '[0,-90,90,0,0,30]'")
    p.add_argument("--prompt", required=True, help="Task description")
    p.add_argument("--reset", action="store_true", help="Send /reset before /infer (clears server-side history)")
    p.add_argument("--return-next-only", action="store_true",
                   help="Ask for only the next action instead of the full 15-step chunk")
    p.add_argument("--num-sampling-step", type=int, default=35, help="Denoising steps (default 35)")
    p.add_argument("--stop-after-step", type=int, default=None, help="Stop video denoising after this step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-actions", type=pathlib.Path, default=None,
                   help="Write the returned action chunk as JSON to this path (else printed to stdout)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="HTTP timeout in seconds")
    args = p.parse_args()

    server = _resolve_server(args)
    client = MimicVideoClient(server=server, timeout=args.timeout)

    print(f"Server: {server}", file=sys.stderr)

    # Quick health check so misconfigured URLs fail loudly with a clear message.
    try:
        health = client.health()
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: cannot reach {server}/healthz: {exc}", file=sys.stderr)
        print("Hint: is the SSH tunnel up? Did you `./deployment/serve_mimic_video.sh` on the GPU box?", file=sys.stderr)
        return 1
    if not health.get("ok"):
        print(f"Server reports not-ready: {health}", file=sys.stderr)
        return 1

    image_bytes = _read_image_bytes(args.image)
    print(f"Sending {args.image.name} ({len(image_bytes) / 1024:.1f} KiB)", file=sys.stderr)

    if args.reset:
        client.reset(args.prompt)

    t0 = time.perf_counter()
    resp = client.infer(
        prompt=args.prompt,
        state=args.state,
        image_bytes=image_bytes,
        return_full_chunk=not args.return_next_only,
        num_sampling_step=args.num_sampling_step,
        stop_after_step=args.stop_after_step,
        seed=args.seed,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    actions = resp["actions"]
    print(
        f"Got {len(actions)} action(s) × {resp['action_dim']} dims "
        f"(server {resp['infer_ms']:.0f} ms, round-trip {dt_ms:.0f} ms, "
        f"ran_model={resp['ran_model']})",
        file=sys.stderr,
    )

    payload = json.dumps(actions, indent=2)
    if args.save_actions:
        args.save_actions.write_text(payload)
        print(f"Wrote {args.save_actions}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

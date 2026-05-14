"""
Track ball trajectory for every episode in a LeRobot dataset using
Grounding DINO (frame-0 localisation) + SAM2 (video propagation).

Outputs per episode (written to --output-dir, default: ./ball_tracking/):
  ep<NNN>.h5           — masks (N,H,W uint8) + centroids (N,2 float32, NaN=missing)
  ep<NNN>_clean.mp4    — raw video frames, no annotations
  ep<NNN>_annotated.mp4 — light-blue mask overlay + (x,y) centroid text

Usage:
    python scripts/extract_ball_trajectory.py
    python scripts/extract_ball_trajectory.py --episode 3
    python scripts/extract_ball_trajectory.py --text "white ball" --output-dir out/

Dependencies (lerobot env):
    pip install -e /home/nielsen/codes/sam2   (+ hydra-core, iopath)
    pip install h5py
"""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset / model configuration
# ---------------------------------------------------------------------------

DATASET_ID          = "/home/nielsen/codes/Ex1_attempt_1"
VIDEO_KEY           = "observation.images.front"
DEFAULT_SAM2_MODEL  = "facebook/sam2-hiera-small"
DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_TEXT        = "white ball"
GDINO_THRESHOLD     = 0.3


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_index_map(dataset_id: str) -> tuple[dict, dict]:
    """Return (global_idx→episode, episode→(start, end)) from parquet metadata."""
    parquet_files = sorted(Path(dataset_id).rglob("data/**/*.parquet"))
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "index"]) for f in parquet_files]
    )
    index_to_episode = dict(zip(df["index"].tolist(), df["episode_index"].tolist()))
    episode_ranges = {
        int(ep): (int(sub["index"].min()), int(sub["index"].max()))
        for ep, sub in df.groupby("episode_index")
    }
    return index_to_episode, episode_ranges


def get_video_files(dataset_id: str) -> list[Path]:
    return sorted((Path(dataset_id) / "videos" / VIDEO_KEY).rglob("*.mp4"))


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        n = c.streams.video[0].frames
        return n if n else sum(1 for _ in c.decode(video=0))


def _load_frames_worker(args: tuple) -> tuple[dict[int, np.ndarray], float]:
    """Decode one video file, returning {global_idx: frame_rgb} for the target episode."""
    vf_str, offset, episode, index_to_episode = args
    frames: dict[int, np.ndarray] = {}
    fps = 30.0
    with av.open(vf_str) as container:
        fps = float(container.streams.video[0].average_rate) or 30.0
        for local_idx, frame in enumerate(container.decode(video=0)):
            global_idx = offset + local_idx
            if index_to_episode.get(global_idx) == episode:
                frames[global_idx] = frame.to_ndarray(format="rgb24")
    return frames, fps


def load_episode_frames(
    dataset_id: str,
    episode: int,
    index_to_episode: dict,
    video_files: list[Path],
    offsets: list[int],
) -> tuple[list[np.ndarray], float]:
    """Return (list of RGB uint8 frames sorted by global index, fps) for one episode."""
    args_list = [
        (str(vf), offset, episode, index_to_episode)
        for vf, offset in zip(video_files, offsets)
    ]
    all_frames: dict[int, np.ndarray] = {}
    fps = 30.0
    with ProcessPoolExecutor(max_workers=len(video_files)) as executor:
        for frames_dict, file_fps in executor.map(_load_frames_worker, args_list):
            all_frames.update(frames_dict)
            fps = file_fps
    return [all_frames[i] for i in sorted(all_frames)], fps


# ---------------------------------------------------------------------------
# Frame-0 detection: Grounding DINO → bounding box
# ---------------------------------------------------------------------------

def detect_box_with_gdino(
    frame_rgb: np.ndarray,
    text: str,
    gdino_processor,
    gdino_model,
    device: str,
) -> np.ndarray:
    """Return highest-score box [x0,y0,x1,y1] for text in frame_rgb. Raises if none found."""
    pil_img    = Image.fromarray(frame_rgb)
    text_input = text.rstrip(".") + "."
    inputs     = gdino_processor(images=pil_img, text=text_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    H, W    = frame_rgb.shape[:2]
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=GDINO_THRESHOLD,
        text_threshold=GDINO_THRESHOLD,
        target_sizes=[(H, W)],
    )[0]

    if len(results["boxes"]) == 0:
        raise RuntimeError(
            f"Grounding DINO found no '{text}' in frame 0 (threshold={GDINO_THRESHOLD}). "
            "Try a different --text."
        )

    best  = results["scores"].argmax().item()
    box   = results["boxes"][best].cpu().numpy().astype(np.float32)
    score = results["scores"][best].item()
    print(f"  GDino '{text}': box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) score={score:.2f}")
    return box


# ---------------------------------------------------------------------------
# SAM2 tracking
# ---------------------------------------------------------------------------

def frames_to_jpeg_dir(frames: list[np.ndarray], directory: str):
    for i, frame_rgb in enumerate(frames):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(directory) / f"{i:05d}.jpg"), bgr)


def track_episode_sam2(
    frames: list[np.ndarray],
    predictor,
    boxes: dict[int, np.ndarray],
) -> dict[int, tuple[np.ndarray, list[np.ndarray | None]]]:
    """
    Run SAM2 with box prompts for multiple objects on frame 0 and propagate
    in a single pass.

    Args:
        boxes: {obj_id: box [x0,y0,x1,y1]}

    Returns:
        {obj_id: (masks (N,H,W) uint8, positions list of [x,y]|None)}
    """
    H, W = frames[0].shape[:2]
    n    = len(frames)
    results: dict[int, tuple[np.ndarray, list]] = {
        obj_id: (np.zeros((n, H, W), dtype=np.uint8), [None] * n)
        for obj_id in boxes
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_to_jpeg_dir(frames, tmpdir)
        inference_state = predictor.init_state(
            video_path=tmpdir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        for obj_id, box in boxes.items():
            predictor.add_new_points_or_box(
                inference_state, frame_idx=0, obj_id=obj_id, box=box
            )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            for i, obj_id in enumerate(obj_ids):
                if obj_id not in results:
                    continue
                mask = (mask_logits[i, 0].cpu().numpy() > 0).astype(np.uint8)
                results[obj_id][0][frame_idx] = mask
                ys, xs = np.where(mask)
                if len(xs):
                    results[obj_id][1][frame_idx] = np.array([xs.mean(), ys.mean()])

        predictor.reset_state(inference_state)

    return results


# ---------------------------------------------------------------------------
# Post-processing: interpolate None gaps
# ---------------------------------------------------------------------------

def interpolate_positions(positions: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Linearly interpolate across None gaps. Leading/trailing Nones stay None."""
    result = list(positions)
    n, i = len(result), 0
    while i < n:
        if result[i] is None:
            j = i
            while j < n and result[j] is None:
                j += 1
            if i > 0 and j < n:
                p0, p1, gap = result[i - 1], result[j], j - (i - 1)
                for k in range(i, j):
                    t = (k - (i - 1)) / gap
                    result[k] = (1 - t) * p0 + t * p1
            i = j
        else:
            i += 1
    return result


CircleData = tuple[np.ndarray, np.ndarray]   # (filled_mask H×W uint8, centroid [x,y] float32)

# ---------------------------------------------------------------------------
# Table circle detection via GDINO + SAM2 image predictor (static, frame-0 only)
# ---------------------------------------------------------------------------

def detect_table_circles_gdino_sam2(
    frame_rgb: np.ndarray,
    gdino_processor,
    gdino_model,
    sam2_image_predictor,
    device: str,
) -> tuple[CircleData, CircleData] | None:
    """
    Detect the two white mat circles using Grounding DINO + SAM2 image predictor.

    Goal circle = leftmost (lowest x-center). Fills each ring mask via fitEllipse.
    For the obstacle, tries both with and without a scaled goal-logit template as
    mask_input, and keeps whichever SAM2 scores higher.

    Returns ((goal_mask, goal_centroid), (obstacle_mask, obstacle_centroid)) or None.
    """
    H, W = frame_rgb.shape[:2]
    pil_img = Image.fromarray(frame_rgb)

    inputs = gdino_processor(
        images=pil_img, text="circle marking on mat.", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = gdino_model(**inputs)
    res = gdino_processor.post_process_grounded_object_detection(
        out, inputs["input_ids"],
        threshold=0.15, text_threshold=0.15,
        target_sizes=[(H, W)],
    )[0]

    raw = sorted(
        [(res["boxes"][i].cpu().numpy(), res["scores"][i].item())
         for i, lbl in enumerate(res["labels"]) if lbl == "circle"],
        key=lambda x: -x[1],
    )

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / (ua + 1e-6)

    # NMS: suppress overlapping detections (IoU > 0.3), keep highest score
    circles: list[tuple[np.ndarray, float]] = []
    for box, score in raw:
        if all(_iou(box, k[0]) < 0.3 for k in circles):
            circles.append((box, score))

    if len(circles) < 2:
        print(f"  WARNING: GDINO detected only {len(circles)} circle(s) — circle data omitted")
        return None

    # Goal = leftmost circle (lowest x-center)
    circles.sort(key=lambda x: (x[0][0] + x[0][2]) / 2)
    goal_box, goal_score = circles[0]
    obs_box,  obs_score  = circles[1]
    goal_cx = (goal_box[0] + goal_box[2]) / 2
    goal_cy = (goal_box[1] + goal_box[3]) / 2
    obs_cx  = (obs_box[0]  + obs_box[2])  / 2
    obs_cy  = (obs_box[1]  + obs_box[3])  / 2

    print(f"  goal_circle     GDINO score={goal_score:.2f} "
          f"cx={goal_cx:.0f} cy={goal_cy:.0f}")
    print(f"  obstacle_circle GDINO score={obs_score:.2f} "
          f"cx={obs_cx:.0f} cy={obs_cy:.0f}")

    # SAM2 image predictor: goal circle
    sam2_image_predictor.set_image(frame_rgb)
    goal_masks, _, goal_logits = sam2_image_predictor.predict(
        box=goal_box, multimask_output=False
    )
    goal_ring  = goal_masks[0].astype(np.uint8)
    goal_logit = goal_logits[0]   # (256, 256)

    # Build scaled+translated goal logit as template for the obstacle
    _new_sz = int(256 * 1.2)
    _log_s  = cv2.resize(goal_logit, (_new_sz, _new_sz), interpolation=cv2.INTER_LINEAR)
    _c2     = (_new_sz - 256) // 2
    _log_s  = _log_s[_c2:_c2 + 256, _c2:_c2 + 256]
    _dx256  = int((obs_cx - goal_cx) / W * 256)
    _dy256  = int((obs_cy - goal_cy) / H * 256)
    _log_t  = np.roll(np.roll(_log_s, _dy256, axis=0), _dx256, axis=1)
    mask_input = _log_t[None]

    # SAM2 obstacle: try template vs no-template, keep higher confidence
    obs_masks_t, obs_scores_t, _ = sam2_image_predictor.predict(
        box=obs_box, mask_input=mask_input, multimask_output=True
    )
    obs_masks_n, obs_scores_n, _ = sam2_image_predictor.predict(
        box=obs_box, multimask_output=True
    )
    best_t, best_n = int(obs_scores_t.argmax()), int(obs_scores_n.argmax())
    if obs_scores_t[best_t] >= obs_scores_n[best_n]:
        obs_ring = obs_masks_t[best_t].astype(np.uint8)
        print(f"  obstacle_circle SAM2 template   score={obs_scores_t[best_t]:.3f}")
    else:
        obs_ring = obs_masks_n[best_n].astype(np.uint8)
        print(f"  obstacle_circle SAM2 no-template score={obs_scores_n[best_n]:.3f}")

    def _ring_to_filled(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        ys, xs = np.where(ring > 0)
        if len(xs) < 5:
            return None, None
        pts     = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.float32)
        ellipse = cv2.fitEllipse(pts)
        mask    = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)
        cx_e, cy_e = ellipse[0]
        return mask, np.array([cx_e, cy_e], dtype=np.float32)

    goal_mask, goal_centroid = _ring_to_filled(goal_ring)
    obs_mask,  obs_centroid  = _ring_to_filled(obs_ring)

    if goal_mask is None or obs_mask is None:
        print("  WARNING: ellipse fit failed for one or both circles — circle data omitted")
        return None

    print(f"  goal_circle     centroid: ({goal_centroid[0]:.1f}, {goal_centroid[1]:.1f})")
    print(f"  obstacle_circle centroid: ({obs_centroid[0]:.1f}, {obs_centroid[1]:.1f})")
    return (goal_mask, goal_centroid), (obs_mask, obs_centroid)


# ---------------------------------------------------------------------------
# Table circle detection (Canny+GHT fallback — kept for debug_circles.py)
# ---------------------------------------------------------------------------



def _generalized_hough_ellipse(
    edges: np.ndarray,
    goal_ellipse: tuple,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    step: int = 5,
    inlier_dist: float = 8.0,
    min_votes: int = 5,
) -> tuple[tuple | None, list, np.ndarray | None]:
    """
    Generalised Hough Transform using the goal ellipse as a template.

    For every candidate center on a (step × step) grid inside the search window,
    counts how many Canny edge pixels lie within inlier_dist of the template
    ellipse placed at that center.  The peak of the vote accumulator is the
    obstacle circle center.

    Returns (obs_ellipse, inlier_indices, accumulator).
      obs_ellipse   — cv2 ellipse tuple at the winning center, or None.
      inlier_indices — indices into np.column_stack([xs, ys]) of all edge pixels.
      accumulator   — 2-D vote array shaped (n_cy, n_cx); None if no edge pixels.
    """
    _, (goal_MA, goal_ma), goal_angle = goal_ellipse
    cos_a = np.cos(np.radians(goal_angle))
    sin_a = np.sin(np.radians(goal_angle))

    ys, xs  = np.where(edges > 0)
    if len(xs) == 0:
        return None, [], None
    all_pts = np.column_stack([xs, ys]).astype(np.float32)   # (N, 2)

    cx_vals = np.arange(x_range[0], x_range[1], step, dtype=np.float32)
    cy_vals = np.arange(y_range[0], y_range[1], step, dtype=np.float32)
    if len(cx_vals) == 0 or len(cy_vals) == 0:
        return None, [], None

    # Vectorised: compute votes for all centers at once.
    # dx/dy: (N, n_cx) after broadcasting over cy in a loop to keep memory bounded.
    accumulator = np.zeros((len(cy_vals), len(cx_vals)), dtype=np.int32)

    for i, cy in enumerate(cy_vals):
        dy = all_pts[:, 1] - cy                             # (N,)
        dx = all_pts[:, 0:1] - cx_vals[np.newaxis, :]      # (N, n_cx)
        u  = ( dx * cos_a + dy[:, np.newaxis] * sin_a) / (goal_MA / 2 + 1e-6)
        v  = (-dx * sin_a + dy[:, np.newaxis] * cos_a) / (goal_ma / 2 + 1e-6)
        dist = np.abs(np.sqrt(u ** 2 + v ** 2) - 1) * (goal_ma / 2)
        accumulator[i] = (dist < inlier_dist).sum(axis=0)

    best_flat = accumulator.argmax()
    best_votes = accumulator.flat[best_flat]
    if best_votes < min_votes:
        return None, [], accumulator

    best_iy, best_ix = np.unravel_index(best_flat, accumulator.shape)
    best_cx, best_cy = float(cx_vals[best_ix]), float(cy_vals[best_iy])

    # Refine center at step=1px in a ±step window around the coarse peak.
    refine_r = step + 1
    ref_cx   = np.arange(best_cx - refine_r, best_cx + refine_r + 1, 1.0, dtype=np.float32)
    ref_cy   = np.arange(best_cy - refine_r, best_cy + refine_r + 1, 1.0, dtype=np.float32)
    ref_acc  = np.zeros((len(ref_cy), len(ref_cx)), dtype=np.int32)
    for i, cy in enumerate(ref_cy):
        dy = all_pts[:, 1] - cy
        dx = all_pts[:, 0:1] - ref_cx[np.newaxis, :]
        u  = ( dx * cos_a + dy[:, np.newaxis] * sin_a) / (goal_MA / 2 + 1e-6)
        v  = (-dx * sin_a + dy[:, np.newaxis] * cos_a) / (goal_ma / 2 + 1e-6)
        dist = np.abs(np.sqrt(u ** 2 + v ** 2) - 1) * (goal_ma / 2)
        ref_acc[i] = (dist < inlier_dist).sum(axis=0)
    ri, rj = np.unravel_index(ref_acc.argmax(), ref_acc.shape)
    best_cx, best_cy = float(ref_cx[rj]), float(ref_cy[ri])

    dx_b = all_pts[:, 0] - best_cx
    dy_b = all_pts[:, 1] - best_cy
    u_b  = ( dx_b * cos_a + dy_b * sin_a) / (goal_MA / 2 + 1e-6)
    v_b  = (-dx_b * sin_a + dy_b * cos_a) / (goal_ma / 2 + 1e-6)
    dist_b     = np.abs(np.sqrt(u_b ** 2 + v_b ** 2) - 1) * (goal_ma / 2)
    inlier_idx = np.where(dist_b < inlier_dist)[0].tolist()

    obs_ellipse = ((best_cx, best_cy), (goal_MA, goal_ma), goal_angle)

    return obs_ellipse, inlier_idx, accumulator


def detect_table_circles(
    frame_rgb: np.ndarray,
    min_area: int = 2000,
    max_area: int = 80000,
    canny_low: int = 30,
    canny_high: int = 100,
    min_circularity: float = 0.5,
    max_aspect_ratio: float = 3.0,
    dedup_radius: float = 30.0,
    return_intermediates: bool = False,
) -> tuple[CircleData, CircleData] | None:
    """
    Detect the two white circles drawn on the mat using Canny edge detection.

    Primary path: contour-based detection with circularity + aspect-ratio filters.
    Fallback (when only 1 circle found): RANSAC ellipse fit on the remaining edge
    points, constrained to the goal circle's size/shape, to recover the partially
    occluded obstacle circle.

    Candidates sorted left → right:
      index 0 → goal_circle     (leftmost, fully visible)
      index 1 → obstacle_circle (middle, may be partially occluded)

    Returns ((goal_mask, goal_centroid), (obstacle_mask, obstacle_centroid)),
    or None if fewer than 2 circles are found.
    """
    gray    = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_pass, filter_pass = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area or len(cnt) < 5:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        ar = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), _ = ar
        aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-6)
        area_pass.append((cx, cy, ar, area, circularity, aspect_ratio, cnt))
        if circularity < min_circularity or aspect_ratio > max_aspect_ratio:
            continue
        filter_pass.append((cx, cy, ar, area, circularity, aspect_ratio, cnt))

    # Deduplicate: keep best-circularity within dedup_radius
    deduped: list = []
    for item in sorted(filter_pass, key=lambda c: -c[4]):
        cx, cy = item[0], item[1]
        if all(np.hypot(cx - d[0], cy - d[1]) > dedup_radius for d in deduped):
            deduped.append(item)
    candidates = [(cx, cy, ellipse, area, circ)
                  for cx, cy, ellipse, area, circ, *_ in deduped]

    # --- Fallback: Generalised Hough Transform for the occluded obstacle circle ---
    ght_accumulator = ght_inlier_pts = None
    x_range = y_range = (0, 0)
    if len(candidates) == 1:
        goal_cx, goal_cy, goal_ellipse, *_ = candidates[0]
        _, (goal_MA, goal_ma), goal_angle = goal_ellipse
        goal_r = max(goal_MA, goal_ma) / 2
        H, W   = edges.shape

        x_range = (goal_cx + goal_r * 0.3, min(W, goal_cx + goal_r * 6))
        y_range = (max(0, goal_cy - goal_r * 2), min(H, goal_cy + goal_r * 2))

        # Filter edges to only those adjacent to a bright (white circle) pixel.
        # This suppresses robot-arm and mat-boundary edges, leaving only the
        # white circle arc for GHT voting.
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        bright_mask = (gray > 160).astype(np.uint8)
        bright_dilated = cv2.dilate(bright_mask, np.ones((3, 3), np.uint8))
        edges_white = (edges & bright_dilated).astype(np.uint8)

        # Obstacle white circle is ~1.2x the goal/target circle in pixel space.
        _OBS_SCALE = 1.2
        obs_template = (goal_ellipse[0],
                        (goal_MA * _OBS_SCALE, goal_ma * _OBS_SCALE),
                        goal_angle)
        obs_ellipse, inlier_idx, ght_accumulator = _generalized_hough_ellipse(
            edges_white, obs_template, x_range=x_range, y_range=y_range,
            step=5, inlier_dist=8.0, min_votes=5,
        )
        if obs_ellipse is not None:
            ys, xs = np.where(edges_white > 0)
            all_pts = np.column_stack([xs, ys]).astype(np.float32)
            ght_inlier_pts = all_pts[inlier_idx]
            obs_cx, obs_cy = obs_ellipse[0]
            candidates.append((obs_cx, obs_cy, obs_ellipse, 0, 0))
            print("  obstacle_circle: recovered via Generalised Hough Transform")

    if len(candidates) < 2:
        if return_intermediates:
            return None, {"edges": edges, "area_pass": area_pass,
                          "filter_pass": filter_pass, "candidates": candidates,
                          "ght_accumulator": ght_accumulator,
                          "ght_x_range": x_range, "ght_y_range": y_range,
                          "ght_inlier_pts": ght_inlier_pts}
        return None

    candidates.sort(key=lambda c: c[0])  # sort left → right by x centroid
    goal_cx, goal_cy, goal_ellipse, *_        = candidates[0]
    obstacle_cx, obstacle_cy, obs_ellipse, *_ = candidates[1]

    H, W = frame_rgb.shape[:2]

    goal_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(goal_mask, goal_ellipse, 255, -1)

    obstacle_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(obstacle_mask, obs_ellipse, 255, -1)

    print(f"  goal_circle     centroid: ({goal_cx:.1f}, {goal_cy:.1f})")
    print(f"  obstacle_circle centroid: ({obstacle_cx:.1f}, {obstacle_cy:.1f})")

    result = (
        (goal_mask,     np.array([goal_cx,     goal_cy],     dtype=np.float32)),
        (obstacle_mask, np.array([obstacle_cx, obstacle_cy], dtype=np.float32)),
    )
    if return_intermediates:
        return result, {"edges": edges, "area_pass": area_pass,
                        "filter_pass": filter_pass, "candidates": candidates,
                        "ght_accumulator": ght_accumulator,
                        "ght_x_range": x_range, "ght_y_range": y_range,
                        "ght_inlier_pts": ght_inlier_pts}
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_hdf5(
    masks: np.ndarray,
    positions: list[np.ndarray | None],
    path: str,
    goal_mask: np.ndarray | None = None,
    goal_centroid: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
    obstacle_centroid: np.ndarray | None = None,
    robot_masks: np.ndarray | None = None,
):
    centroids = np.full((len(positions), 2), np.nan, dtype=np.float32)
    for i, p in enumerate(positions):
        if p is not None:
            centroids[i] = p
    with h5py.File(path, "w") as f:
        f.create_dataset("masks",     data=masks,     compression="gzip", compression_opts=4)
        f.create_dataset("centroids", data=centroids)
        if goal_mask is not None:
            f.create_dataset("goal_circle_mask",      data=goal_mask,      compression="gzip", compression_opts=4)
        if goal_centroid is not None:
            f.create_dataset("goal_circle_centroid",  data=goal_centroid)
        if obstacle_mask is not None:
            f.create_dataset("obstacle_circle_mask",     data=obstacle_mask,     compression="gzip", compression_opts=4)
        if obstacle_centroid is not None:
            f.create_dataset("obstacle_circle_centroid", data=obstacle_centroid)
        if robot_masks is not None:
            f.create_dataset("robot_masks", data=robot_masks, compression="gzip", compression_opts=4)
    print(f"  Saved HDF5  → {path}")


def save_clean_video(frames: list[np.ndarray], fps: float, path: str):
    h, w   = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame_rgb in frames:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved clean → {path}")


# Ball mask: light-blue (173, 216, 230) RGB → (230, 216, 173) BGR
_MASK_COLOR_BGR    = np.array([230, 216, 173], dtype=np.float32)
_MASK_ALPHA        = 0.6

# Goal circle: light-green (144, 238, 144) RGB → (144, 238, 144) BGR
_GOAL_COLOR_BGR     = np.array([144, 238, 144], dtype=np.float32)
# Obstacle circle: light-red (255, 160, 160) RGB → (160, 160, 255) BGR
_OBSTACLE_COLOR_BGR = np.array([160, 160, 255], dtype=np.float32)
_CIRCLE_ALPHA       = 0.45
# Robot arm: light-purple / plum (221, 160, 221) RGB = (221, 160, 221) BGR
_ROBOT_COLOR_BGR    = np.array([221, 160, 221], dtype=np.float32)
_ROBOT_ALPHA        = 0.5

_TRAIL_COLOR_BGR = (0, 255, 255)   # yellow in BGR
_TRAIL_MAX_LEN   = 60              # max number of past points to draw


def _apply_mask_overlay(vis: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float):
    if mask.any():
        vis[mask > 0] = (
            vis[mask > 0].astype(np.float32) * (1 - alpha) + color * alpha
        ).astype(np.uint8)


def _draw_legend(vis: np.ndarray, entries: list[tuple[tuple[int, int, int], str]]):
    """Draw a small color-swatch legend in the top-right corner of vis (in-place).

    entries: [(bgr_color, label), ...]
    """
    swatch_h, swatch_w = 14, 14
    pad, gap = 6, 4
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.42, 1

    # Measure text widths to size the background box
    text_widths = [cv2.getTextSize(lbl, font, font_scale, thickness)[0][0] for _, lbl in entries]
    box_w = pad + swatch_w + gap + max(text_widths) + pad
    row_h = max(swatch_h, 14) + 4
    box_h = pad + len(entries) * row_h + pad

    H, W = vis.shape[:2]
    x0   = W - box_w - 8
    y0   = 8

    # Semi-transparent dark background
    roi = vis[y0:y0 + box_h, x0:x0 + box_w]
    roi[:] = (roi.astype(np.float32) * 0.4 + np.array([20, 20, 20], np.float32) * 0.6).astype(np.uint8)

    for idx, (color, label) in enumerate(entries):
        sy = y0 + pad + idx * row_h
        sx = x0 + pad
        # Filled color swatch
        cv2.rectangle(vis, (sx, sy), (sx + swatch_w, sy + swatch_h), color, -1)
        cv2.rectangle(vis, (sx, sy), (sx + swatch_w, sy + swatch_h), (200, 200, 200), 1)
        # Label text
        cv2.putText(vis, label, (sx + swatch_w + gap, sy + swatch_h - 1),
                    font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)


def save_annotated_video(
    frames: list[np.ndarray],
    masks: np.ndarray,
    positions: list[np.ndarray | None],
    fps: float,
    path: str,
    goal_mask: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
    robot_masks: np.ndarray | None = None,
):
    h, w   = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Build legend entries for the masks that are actually present
    legend_entries: list[tuple[tuple[int, int, int], str]] = []
    if goal_mask is not None:
        legend_entries.append((tuple(int(v) for v in _GOAL_COLOR_BGR),     "goal circle"))
    if obstacle_mask is not None:
        legend_entries.append((tuple(int(v) for v in _OBSTACLE_COLOR_BGR), "obstacle circle"))
    legend_entries.append((tuple(int(v) for v in _MASK_COLOR_BGR),         "ball"))
    if robot_masks is not None:
        legend_entries.append((tuple(int(v) for v in _ROBOT_COLOR_BGR),    "robot arm"))

    trail: list[tuple[int, int]] = []   # running history of valid centroids

    for i, (frame_rgb, mask, pos) in enumerate(zip(frames, masks, positions)):
        vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if goal_mask is not None:
            _apply_mask_overlay(vis, goal_mask, _GOAL_COLOR_BGR, _CIRCLE_ALPHA)
        if obstacle_mask is not None:
            _apply_mask_overlay(vis, obstacle_mask, _OBSTACLE_COLOR_BGR, _CIRCLE_ALPHA)

        # Light-purple robot mask overlay
        if robot_masks is not None and robot_masks[i].any():
            _apply_mask_overlay(vis, robot_masks[i], _ROBOT_COLOR_BGR, _ROBOT_ALPHA)

        # Light-blue ball mask overlay
        if mask.any():
            _apply_mask_overlay(vis, mask, _MASK_COLOR_BGR, _MASK_ALPHA)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 191, 0), 2)

        if pos is not None:
            cx, cy = int(pos[0]), int(pos[1])
            trail.append((cx, cy))
            if len(trail) > _TRAIL_MAX_LEN:
                trail.pop(0)

        # Yellow trajectory trail
        if len(trail) >= 2:
            pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=_TRAIL_COLOR_BGR, thickness=2)

        # Centroid dot + text
        if pos is not None:
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(vis, f"({cx}, {cy})", (cx + 8, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        _draw_legend(vis, legend_entries)

        writer.write(vis)

    writer.release()
    print(f"  Saved annot → {path}")


def save_trajectory_plot(positions: list[np.ndarray | None], episode: int, path: str):
    n       = len(positions)
    valid   = [(i, p) for i, p in enumerate(positions) if p is not None]
    missing = [i for i, p in enumerate(positions) if p is None]

    if not valid:
        print(f"  No positions — skipping plot for ep {episode:03d}")
        return

    pts  = np.array([p for _, p in valid])
    t    = np.arange(n)
    # Full-length arrays with NaN where position is missing
    x_full = np.array([p[0] if p is not None else np.nan for p in positions])
    y_full = np.array([p[1] if p is not None else np.nan for p in positions])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: 2D path (valid positions only) ---
    ax = axes[0]
    ax.plot(pts[:, 0], pts[:, 1], color="tab:green", linewidth=1.5)
    ax.scatter(*pts[0],  color="tab:green", marker="o", s=60, zorder=5, label="start")
    ax.scatter(*pts[-1], color="tab:green", marker="x", s=80, zorder=5, label="end")
    ax.set_title(f"Episode {episode:03d} — ball trajectory")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # --- Right: x/y over full frame range ---
    ax = axes[1]
    ax.plot(t, x_full, color="tab:red",  linewidth=1.5, label="x")
    ax.plot(t, y_full, color="tab:blue", linewidth=1.5, label="y")

    if missing:
        # Place red X markers at bottom of the axes for missing frames
        ax.autoscale_view()
        ymin, _ = ax.get_ylim()
        ax.scatter(missing, np.full(len(missing), ymin),
                   color="red", marker="x", s=60, linewidths=2, zorder=5,
                   label=f"missing ({len(missing)})")

    ax.set_xlim(0, n - 1)
    ax.set_title("x / y over frames")
    ax.set_xlabel("frame")
    ax.set_ylabel("px")
    ax.legend(fontsize=8)

    n_valid = len(valid)
    fig.suptitle(
        f"Episode {episode:03d}  |  {n_valid}/{n} frames detected"
        f"  ({100 * n_valid / n:.1f}%)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track ball in LeRobot dataset with Grounding DINO + SAM2.")
    parser.add_argument("--episode",      type=int, default=None, help="Single episode to process (default: all)")
    parser.add_argument("--from-episode", type=int, default=None, help="Process all episodes >= this index")
    parser.add_argument("--dataset",    default=DATASET_ID,               help=f"LeRobot dataset path (default: {DATASET_ID})")
    parser.add_argument("--text",       default=DEFAULT_TEXT,             help=f"Grounding DINO text prompt (default: '{DEFAULT_TEXT}')")
    parser.add_argument("--model",      default=DEFAULT_SAM2_MODEL,       help=f"SAM2 model ID (default: {DEFAULT_SAM2_MODEL})")
    parser.add_argument("--output-dir", default=None,   help="Output directory (default: <dataset>/metadata/)")
    parser.add_argument("--device",     default=None,   help="torch device (default: auto)")
    args = parser.parse_args()

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device:     {device}")
    print(f"Output dir: {out_dir}")

    # --- Build dataset index ---
    print(f"\nBuilding index map for {args.dataset} …")
    index_to_episode, episode_ranges = build_index_map(args.dataset)
    video_files = get_video_files(args.dataset)

    offsets, cumulative = [], 0
    for vf in video_files:
        offsets.append(cumulative)
        cumulative += _video_frame_count(vf)

    all_eps  = sorted(episode_ranges)
    if args.episode is not None:
        episodes = [args.episode]
    elif args.from_episode is not None:
        episodes = [ep for ep in all_eps if ep >= args.from_episode]
    else:
        episodes = all_eps
    print(f"Episodes to process: {episodes}")

    # --- Load models once ---
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"\nLoading Grounding DINO ({DEFAULT_GDINO_MODEL}) …")
    gdino_processor = AutoProcessor.from_pretrained(DEFAULT_GDINO_MODEL)
    gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(DEFAULT_GDINO_MODEL).to(device)
    gdino_model.eval()

    print(f"Loading SAM2 ({args.model}) …")
    sam2_image_predictor = SAM2ImagePredictor.from_pretrained(args.model, device=device)
    sam2_predictor       = SAM2VideoPredictor.from_pretrained(args.model, device=device)

    # --- Process each episode ---
    for ep in episodes:
        if ep not in episode_ranges:
            print(f"\n[ep {ep:03d}] not found — skipping")
            continue

        print(f"\n[ep {ep:03d}] Loading frames …")
        frames, fps = load_episode_frames(args.dataset, ep, index_to_episode, video_files, offsets)
        print(f"  {len(frames)} frames @ {fps:.1f} fps")

        print(f"[ep {ep:03d}] Detecting ball with Grounding DINO …")
        try:
            ball_box = detect_box_with_gdino(frames[0], args.text, gdino_processor, gdino_model, device)
        except RuntimeError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"[ep {ep:03d}] Detecting robot arm with Grounding DINO …")
        try:
            robot_box = detect_box_with_gdino(frames[0], "black robot arm", gdino_processor, gdino_model, device)
        except RuntimeError as e:
            print(f"  WARNING: robot arm not detected — {e}")
            robot_box = None

        print(f"[ep {ep:03d}] Detecting table circles (GDINO + SAM2) …")
        circles_result = detect_table_circles_gdino_sam2(
            frames[0], gdino_processor, gdino_model, sam2_image_predictor, device
        )
        if circles_result is None:
            print("  WARNING: fewer than 2 white circles found — circle data will be omitted")
            goal_mask = goal_centroid = obstacle_mask = obstacle_centroid = None
        else:
            (goal_mask, goal_centroid), (obstacle_mask, obstacle_centroid) = circles_result

        print(f"[ep {ep:03d}] Propagating with SAM2 …")
        boxes = {1: ball_box}
        if robot_box is not None:
            boxes[2] = robot_box
        tracking = track_episode_sam2(frames, sam2_predictor, boxes)

        masks, positions = tracking[1]
        robot_masks = tracking[2][0] if 2 in tracking else None

        detected = sum(p is not None for p in positions)
        print(f"  Detected {detected}/{len(frames)} frames ({100*detected/len(frames):.1f}%)")

        positions = interpolate_positions(positions)

        tag = f"ep{ep:03d}"
        save_hdf5(masks, positions, str(out_dir / f"{tag}.h5"),
                  goal_mask=goal_mask, goal_centroid=goal_centroid,
                  obstacle_mask=obstacle_mask, obstacle_centroid=obstacle_centroid,
                  robot_masks=robot_masks)
        save_clean_video(frames, fps, str(out_dir / f"{tag}_clean.mp4"))
        save_annotated_video(frames, masks, positions, fps, str(out_dir / f"{tag}_annotated.mp4"),
                             goal_mask=goal_mask, obstacle_mask=obstacle_mask,
                             robot_masks=robot_masks)
        save_trajectory_plot(positions, ep, str(out_dir / f"{tag}_trajectory.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()

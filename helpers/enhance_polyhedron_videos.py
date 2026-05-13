#!/usr/bin/env python3
"""Enhance collected polyhedron videos for Cosmos Video2World finetuning.

The script reads either a Cosmos dataset directory with a ``video/`` folder or
a flat directory of MP4 files. It writes enhanced MP4s to the output directory
and, for Cosmos datasets, copies ``metas/`` and ``t5_xxl/`` sidecars so the
result can be used directly by ``cosmos_predict2.data.dataset_video.Dataset``.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - exercised only in minimal envs
    imageio = None

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal envs
    cv2 = None


CHARCOAL_RGB = np.array([44, 44, 42], dtype=np.float32)


@dataclass
class TrackStats:
    centers: list[tuple[float, float] | None]
    fallback_frames: int = 0
    lost_frames: int = 0
    longest_missing_run: int = 0
    warning: str | None = None


@dataclass
class EnhancementResult:
    frames: np.ndarray
    masks: np.ndarray
    stats: TrackStats


@dataclass
class Candidate:
    mask: np.ndarray
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    fill_ratio: float


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for polyhedron enhancement. "
            "Run this with the mimic-video environment, which declares opencv-python."
        )


def _require_imageio() -> None:
    if imageio is None:
        raise RuntimeError(
            "imageio is required for MP4 IO. Run this with the mimic-video environment, "
            "which declares imageio[pyav,ffmpeg]."
        )


def _roi_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * 0.26)
    y1 = int(h * 0.92)
    x0 = int(w * 0.03)
    x1 = int(w * 0.97)
    mask[y0:y1, x0:x1] = 255
    return mask


def _component_candidates(frame_rgb: np.ndarray) -> list[Candidate]:
    _require_cv2()
    h, w, _ = frame_rgb.shape
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # The object is white/gray on a darker green mat. We start from bright,
    # low-saturation pixels and then erode/open to remove thin workspace lines.
    bright = ((val >= 148) & (sat <= 95)).astype(np.uint8) * 255
    bright = cv2.bitwise_and(bright, _roi_mask((h, w)))
    bright = cv2.medianBlur(bright, 3)

    line_break_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(bright, cv2.MORPH_OPEN, line_break_kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, line_break_kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    candidates: list[Candidate] = []
    frame_area = h * w

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < 80 or area > frame_area * 0.035:
            continue
        if bw < 7 or bh < 7:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue

        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        fill_ratio = area / max(bw * bh, 1)
        if aspect > 3.2 or fill_ratio < 0.18:
            continue

        comp_mask = (labels == label).astype(np.uint8) * 255
        cx, cy = centroids[label]
        candidates.append(
            Candidate(
                mask=comp_mask,
                center=(float(cx), float(cy)),
                bbox=(int(x), int(y), int(bw), int(bh)),
                area=int(area),
                fill_ratio=float(fill_ratio),
            )
        )

    return candidates


def _select_candidate(
    candidates: list[Candidate], prev_center: tuple[float, float] | None, frame_shape: tuple[int, int]
) -> Candidate | None:
    if not candidates:
        return None

    h, w = frame_shape
    if prev_center is None:
        # Exercise 1 starts with the object on the right. The x preference is
        # intentionally weak so the same script remains useful for snippets.
        def first_score(c: Candidate) -> float:
            area_score = min(c.area / 1400.0, 1.8)
            x_score = c.center[0] / max(w, 1)
            y_penalty = abs(c.center[1] - h * 0.52) / max(h, 1)
            return area_score + 0.35 * x_score + c.fill_ratio - y_penalty

        return max(candidates, key=first_score)

    px, py = prev_center

    def track_score(c: Candidate) -> float:
        cx, cy = c.center
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        dist_penalty = dist / max(h, w)
        area_score = min(c.area / 1400.0, 1.5)
        return area_score + c.fill_ratio - 2.8 * dist_penalty

    best = max(candidates, key=track_score)
    max_jump = max(65.0, min(h, w) * 0.16)
    dx = best.center[0] - px
    dy = best.center[1] - py
    if (dx * dx + dy * dy) ** 0.5 > max_jump:
        return None
    return best


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    _require_cv2()
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)


def _mask_bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    return x0, y0, x1, y1


def _object_enhancement_mask(frame_rgb: np.ndarray, track_mask: np.ndarray) -> np.ndarray:
    _require_cv2()
    if np.count_nonzero(track_mask) == 0:
        return track_mask

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, small_kernel, iterations=1)

    # Only search in a narrow band around the already tracked white object.
    # This prevents rectangular ROI patches or workspace markings from ever
    # entering the output mask while still capturing shaded object faces.
    search_zone = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    face_pixels = ((val >= 118) & (sat <= 115)).astype(np.uint8) * 255
    local = cv2.bitwise_and(face_pixels, search_zone)
    local = cv2.bitwise_or(local, seed)

    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(local, connectivity=8)
    best_label = 0
    best_overlap = 0
    for label in range(1, num_labels):
        overlap = int(np.count_nonzero((labels == label) & (track_mask > 0)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_label:
        obj = (labels == best_label).astype(np.uint8) * 255
    else:
        obj = seed.copy()

    track_area = max(int(np.count_nonzero(track_mask)), 1)
    obj_area = int(np.count_nonzero(obj))
    bbox = _mask_bbox(obj, pad=0)
    rectangular_leak = False
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bw = x1 - x0
        bh = y1 - y0
        fill_ratio = obj_area / max(bw * bh, 1)
        rectangular_leak = fill_ratio > 0.88 and (bw > 1.6 * bh or bh > 1.6 * bw or obj_area > track_area * 2.2)

    if obj_area > track_area * 2.6 or rectangular_leak:
        obj = seed

    return obj


def _amplify_object_luminance(
    frame_rgb: np.ndarray,
    obj_mask: np.ndarray,
    contrast_gain: float,
    gray_strength: float,
) -> np.ndarray:
    _require_cv2()
    if contrast_gain <= 0 and gray_strength <= 0:
        return frame_rgb.copy()

    roi = obj_mask > 0
    if np.count_nonzero(roi) < 20:
        return frame_rgb.copy()

    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_float = l_chan.astype(np.float32)
    values = l_float[roi]

    if gray_strength > 0:
        strength = float(np.clip(gray_strength, 0.0, 1.0))
        # Pull only bright object pixels toward light gray. Darker shaded faces
        # are affected less, preserving the shape cues we want to amplify.
        light_gray_l = 198.0
        bright_weight = np.clip((values - 150.0) / 85.0, 0.0, 1.0)
        gray_values = values * (1.0 - strength * bright_weight) + light_gray_l * (strength * bright_weight)
        l_float[roi] = np.minimum(values, gray_values)
        values = l_float[roi]

        # Reduce tiny chroma differences caused by camera white balance while
        # keeping the object neutral instead of tinted.
        a_float = a_chan.astype(np.float32)
        b_float = b_chan.astype(np.float32)
        a_float[roi] = a_float[roi] * (1.0 - 0.45 * strength) + 128.0 * (0.45 * strength)
        b_float[roi] = b_float[roi] * (1.0 - 0.45 * strength) + 128.0 * (0.45 * strength)
        a_chan = np.clip(a_float, 0, 255).astype(np.uint8)
        b_chan = np.clip(b_float, 0, 255).astype(np.uint8)

    if contrast_gain <= 0:
        lab_out = cv2.merge((np.clip(l_float, 0, 255).astype(np.uint8), a_chan, b_chan))
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    lo, hi = np.percentile(values, [8, 92])
    if hi - lo < 3:
        center = float(values.mean())
        adjusted = center + (values - center) * (1.0 + contrast_gain)
    else:
        center = (lo + hi) * 0.5
        adjusted = center + (values - center) * (1.0 + contrast_gain)

    # Keep the object white/gray overall; only exaggerate the existing face
    # luminance differences enough for the world model to see rotations.
    adjusted = np.clip(adjusted, max(lo - 18, 0), min(hi + 18, 255))
    l_float[roi] = 0.25 * values + 0.75 * adjusted
    lab_out = cv2.merge((np.clip(l_float, 0, 255).astype(np.uint8), a_chan, b_chan))
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


def _enhance_frame(
    frame_rgb: np.ndarray,
    track_mask: np.ndarray,
    *,
    edge_opacity: float,
    edge_dilate: int,
    face_contrast: float,
    gray_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require_cv2()
    obj_mask = _object_enhancement_mask(frame_rgb, track_mask)
    roi = obj_mask > 0
    if not np.any(roi):
        return frame_rgb.copy(), obj_mask

    enhanced = _amplify_object_luminance(
        frame_rgb,
        obj_mask,
        contrast_gain=face_contrast,
        gray_strength=gray_strength,
    ).astype(np.float32)

    if edge_opacity > 0:
        gray = cv2.cvtColor(np.clip(enhanced, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=38, threshold2=96)
        eroded_obj = cv2.erode(obj_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        edges = cv2.bitwise_and(edges, eroded_obj)
        if edge_dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges = cv2.dilate(edges, kernel, iterations=edge_dilate)

        edge_pixels = edges > 0
        if np.any(edge_pixels):
            alpha = float(np.clip(edge_opacity, 0.0, 1.0))
            enhanced[edge_pixels] = (1.0 - alpha) * enhanced[edge_pixels] + alpha * CHARCOAL_RGB

    return np.clip(enhanced, 0, 255).astype(np.uint8), obj_mask


def enhance_frames(
    frames: np.ndarray,
    *,
    edge_opacity: float = 0.0,
    edge_dilate: int = 1,
    face_contrast: float = 0.85,
    gray_strength: float = 0.35,
    max_missing: int = 12,
    strict_tracking: bool = False,
) -> EnhancementResult:
    """Enhance a sequence of RGB uint8 frames."""
    _require_cv2()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    enhanced_frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    centers: list[tuple[float, float] | None] = []
    prev_center: tuple[float, float] | None = None
    prev_observed_center: tuple[float, float] | None = None
    velocity = (0.0, 0.0)
    prev_mask: np.ndarray | None = None
    missing = 0
    stats = TrackStats(centers=centers)

    for frame in frames:
        candidates = _component_candidates(frame)
        candidate = _select_candidate(candidates, prev_center, frame.shape[:2])

        if candidate is None:
            missing += 1
            stats.longest_missing_run = max(stats.longest_missing_run, missing)
            if strict_tracking and missing > max_missing:
                raise RuntimeError(f"Tracking lost for more than {max_missing} consecutive frames")

            if prev_mask is not None:
                # During gripper occlusions the white object may disappear for
                # longer than expected. Keep enhancing the predicted object ROI
                # instead of aborting full-dataset preprocessing.
                dx, dy = velocity
                dx = float(np.clip(dx, -12.0, 12.0))
                dy = float(np.clip(dy, -12.0, 12.0))
                track_mask = _shift_mask(prev_mask, dx, dy)
                if np.count_nonzero(track_mask) == 0:
                    track_mask = prev_mask.copy()
                stats.fallback_frames += 1
                if prev_center is not None:
                    prev_center = (prev_center[0] + dx, prev_center[1] + dy)
            else:
                track_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                stats.lost_frames += 1
        else:
            missing = 0
            track_mask = candidate.mask
            if prev_observed_center is not None:
                velocity = (
                    candidate.center[0] - prev_observed_center[0],
                    candidate.center[1] - prev_observed_center[1],
                )
            prev_center = candidate.center
            prev_observed_center = candidate.center
            prev_mask = track_mask.copy()

        enhanced, obj_mask = _enhance_frame(
            frame,
            track_mask,
            edge_opacity=edge_opacity,
            edge_dilate=edge_dilate,
            face_contrast=face_contrast,
            gray_strength=gray_strength,
        )
        enhanced_frames.append(enhanced)
        masks.append(obj_mask)
        centers.append(prev_center if np.count_nonzero(track_mask) else None)

    if stats.longest_missing_run > max_missing:
        stats.warning = (
            f"tracking used fallback for {stats.longest_missing_run} consecutive frames "
            f"(max_missing={max_missing})"
        )

    return EnhancementResult(
        frames=np.stack(enhanced_frames, axis=0),
        masks=np.stack(masks, axis=0),
        stats=stats,
    )


def _read_video(path: Path) -> tuple[np.ndarray, float]:
    _require_imageio()
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 10.0)
        frames = [frame[:, :, :3] for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0).astype(np.uint8), fps


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    _require_imageio()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def _copy_sidecar(src_dir: Path, dst_dir: Path, src_stem: str, dst_stem: str, suffix: str) -> None:
    src = src_dir / f"{src_stem}{suffix}"
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / f"{dst_stem}{suffix}")


def _copy_dataset_sidecars(input_dir: Path, output_dir: Path, src_stem: str, dst_stem: str) -> None:
    _copy_sidecar(input_dir / "metas", output_dir / "metas", src_stem, dst_stem, ".txt")
    _copy_sidecar(input_dir / "t5_xxl", output_dir / "t5_xxl", src_stem, dst_stem, ".pickle")


def _make_debug_sheet(raw: np.ndarray, masks: np.ndarray, enhanced: np.ndarray, out_path: Path) -> None:
    _require_cv2()
    _require_imageio()
    n = raw.shape[0]
    idxs = np.unique(np.rint(np.linspace(0, n - 1, min(5, n))).astype(int))
    rows: list[np.ndarray] = []
    for idx in idxs:
        mask_rgb = np.repeat(masks[idx, :, :, None], 3, axis=2)
        diff = np.abs(enhanced[idx].astype(np.int16) - raw[idx].astype(np.int16)).max(axis=2)
        diff_vis = np.zeros_like(raw[idx])
        diff_vis[:, :, 0] = np.clip(diff * 8, 0, 255).astype(np.uint8)
        diff_vis[:, :, 1] = np.clip(diff * 3, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([raw[idx], mask_rgb, enhanced[idx], diff_vis], axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, sheet)


def _resolve_video_dir(input_dir: Path) -> tuple[Path, bool]:
    video_dir = input_dir / "video"
    if video_dir.is_dir():
        return video_dir, True
    return input_dir, False


def _episode_filter(paths: list[Path], episodes: list[int] | None) -> list[Path]:
    if not episodes:
        return paths
    keep = {f"episode_{episode:06d}" for episode in episodes}
    return [path for path in paths if path.stem in keep]


def process_videos(args: argparse.Namespace) -> None:
    _require_cv2()
    input_dir = args.input_dir
    output_dir = args.output_dir
    video_dir, is_cosmos_layout = _resolve_video_dir(input_dir)
    if not video_dir.is_dir():
        raise SystemExit(f"Video directory not found: {video_dir}")

    output_video_dir = output_dir / "video" if is_cosmos_layout else output_dir
    output_video_dir.mkdir(parents=True, exist_ok=True)
    if is_cosmos_layout:
        (output_dir / "metas").mkdir(parents=True, exist_ok=True)
        (output_dir / "t5_xxl").mkdir(parents=True, exist_ok=True)

    videos = _episode_filter(sorted(video_dir.glob("*.mp4")), args.episodes)
    if not videos:
        raise SystemExit(f"No MP4 videos found in {video_dir}")

    for video_path in videos:
        enhanced_path = output_video_dir / video_path.name
        raw_copy_path = output_video_dir / f"{video_path.stem}_raw.mp4"
        if enhanced_path.exists() and not args.overwrite:
            print(f"skip (exists): {enhanced_path}")
            continue

        frames, fps = _read_video(video_path)
        result = enhance_frames(
            frames,
            edge_opacity=args.edge_opacity,
            edge_dilate=args.edge_dilate,
            face_contrast=args.face_contrast,
            gray_strength=args.gray_strength,
            max_missing=args.max_missing,
            strict_tracking=args.strict_tracking,
        )
        _write_video(enhanced_path, result.frames, fps)

        if is_cosmos_layout:
            _copy_dataset_sidecars(input_dir, output_dir, video_path.stem, video_path.stem)

        if args.copy_raw:
            if args.overwrite or not raw_copy_path.exists():
                shutil.copy2(video_path, raw_copy_path)
            if is_cosmos_layout:
                _copy_dataset_sidecars(input_dir, output_dir, video_path.stem, f"{video_path.stem}_raw")

        if args.debug_dir is not None:
            _make_debug_sheet(
                frames,
                result.masks,
                result.frames,
                args.debug_dir / f"{video_path.stem}_debug.jpg",
            )

        status = "ok"
        if result.stats.warning:
            status = f"warning: {result.stats.warning}"

        print(
            f"{video_path.name}: enhanced={enhanced_path} "
            f"fallback_frames={result.stats.fallback_frames} lost_frames={result.stats.lost_frames} "
            f"longest_missing_run={result.stats.longest_missing_run} {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--edge-opacity",
        type=float,
        default=0.0,
        help="Optional dark internal edge overlay. Default 0 disables artificial edge drawing.",
    )
    parser.add_argument("--edge-dilate", type=int, default=1)
    parser.add_argument(
        "--face-contrast",
        type=float,
        default=0.85,
        help="How strongly to amplify existing luminance differences across object faces.",
    )
    parser.add_argument(
        "--gray-strength",
        type=float,
        default=0.35,
        help="How strongly to pull bright object faces toward neutral light gray.",
    )
    parser.add_argument("--max-missing", type=int, default=12)
    parser.add_argument(
        "--strict-tracking",
        action="store_true",
        help="Abort a video when tracking is lost for more than --max-missing frames.",
    )
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    process_videos(args)


if __name__ == "__main__":
    main()

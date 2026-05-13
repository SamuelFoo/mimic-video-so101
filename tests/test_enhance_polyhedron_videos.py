from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


HELPER_PATH = Path(__file__).resolve().parents[1] / "helpers" / "enhance_polyhedron_videos.py"
spec = importlib.util.spec_from_file_location("enhance_polyhedron_videos", HELPER_PATH)
enhancer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = enhancer
spec.loader.exec_module(enhancer)


def test_helper_imports_without_opencv() -> None:
    assert hasattr(enhancer, "enhance_frames")


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for frame enhancement tests")
def _synthetic_frames() -> tuple[np.ndarray, list[tuple[int, int]]]:
    assert cv2 is not None
    frames = []
    centers = []
    h, w = 120, 180
    for idx in range(8):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = np.array([38, 74, 65], dtype=np.uint8)

        # Static white workspace markings that should not dominate tracking.
        cv2.line(frame, (20, 55), (160, 55), (235, 235, 230), 2)
        cv2.circle(frame, (45, 65), 18, (238, 238, 234), 2)
        cv2.circle(frame, (95, 65), 20, (238, 238, 234), 2)

        cx = 130 - idx * 8
        cy = 66
        centers.append((cx, cy))
        poly = np.array(
            [
                [cx - 14, cy - 8],
                [cx - 5, cy - 17],
                [cx + 12, cy - 13],
                [cx + 17, cy + 4],
                [cx + 5, cy + 15],
                [cx - 12, cy + 11],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, poly, (226, 226, 220))
        cv2.line(frame, (cx - 5, cy - 17), (cx + 5, cy + 15), (174, 174, 168), 1)
        cv2.line(frame, (cx - 14, cy - 8), (cx + 12, cy - 13), (184, 184, 178), 1)

        if idx == 4:
            cv2.rectangle(frame, (cx - 22, cy - 24), (cx + 24, cy + 22), (20, 20, 20), -1)

        frames.append(frame)
    return np.stack(frames, axis=0), centers


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for frame enhancement tests")
def test_enhancement_tracks_object_through_short_occlusion() -> None:
    frames, centers = _synthetic_frames()

    result = enhancer.enhance_frames(frames, edge_opacity=0.55, edge_dilate=1, max_missing=3)

    assert result.stats.lost_frames == 0
    assert result.stats.fallback_frames >= 1
    assert np.count_nonzero(result.masks[5]) > 100

    tracked_after_occlusion = result.stats.centers[5]
    assert tracked_after_occlusion is not None
    assert abs(tracked_after_occlusion[0] - centers[5][0]) < 25


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for frame enhancement tests")
def test_enhancement_changes_object_more_than_background() -> None:
    frames, _centers = _synthetic_frames()

    result = enhancer.enhance_frames(frames, edge_opacity=0.55, edge_dilate=1)

    diff = np.abs(result.frames.astype(np.int16) - frames.astype(np.int16)).mean(axis=3)
    object_change = float(diff[result.masks > 0].mean())
    background_change = float(diff[result.masks == 0].mean())

    assert object_change > 4.0
    assert background_change < 1.0


@pytest.mark.skipif(cv2 is None, reason="OpenCV is required for frame enhancement tests")
def test_workspace_lines_are_not_selected_as_object_mask() -> None:
    frames, _centers = _synthetic_frames()

    result = enhancer.enhance_frames(frames, edge_opacity=0.45, edge_dilate=1)
    mask = result.masks[0] > 0
    ys, xs = np.where(mask)

    assert len(xs) > 100
    assert (xs.max() - xs.min()) < 70
    assert (ys.max() - ys.min()) < 70

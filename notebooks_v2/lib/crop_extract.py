"""YOLO-aligned crop sequences — matches live per-person inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.constants import CROP_SIZE, DEFAULT_BBOX_PADDING, DEFAULT_BUFFER_FRAMES
from lib.embeddings import extract_clip_from_file
from lib.tracking import crop_person, _get_yolo


def _largest_person_box(frame_bgr: np.ndarray) -> tuple[int, int, int, int, float] | None:
    model = _get_yolo()
    if model is None:
        return None
    try:
        res = model(frame_bgr, classes=[0], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        best = None
        best_area = 0
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, float(box.conf[0]))
        return best
    except Exception:
        return None


def _center_crop(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


def frame_to_person_crop(
    frame_bgr: np.ndarray,
    *,
    padding: float = DEFAULT_BBOX_PADDING,
) -> np.ndarray:
    det = _largest_person_box(frame_bgr)
    if det is None:
        return _center_crop(frame_bgr)
    x1, y1, x2, y2, _ = det
    crop = crop_person(frame_bgr, x1, y1, x2, y2, padding=padding)
    return crop if crop is not None else _center_crop(frame_bgr)


def build_crop_buffer_from_frames(
    frames_bgr: list[np.ndarray],
    *,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    padding: float = DEFAULT_BBOX_PADDING,
) -> list[np.ndarray]:
    """Simulate live PerPersonTracker crop deque on a frame list."""
    buf: list[np.ndarray] = []
    for frame in frames_bgr:
        buf.append(frame_to_person_crop(frame, padding=padding))
        if len(buf) > buffer_frames:
            buf = buf[-buffer_frames:]
    return buf


def build_crop_buffer_from_clip(
    clip_path: Path | str,
    *,
    max_read: int = 64,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    padding: float = DEFAULT_BBOX_PADDING,
) -> list[np.ndarray]:
    frames = extract_clip_from_file(clip_path, max_frames=max_read)
    if not frames:
        return []
    return build_crop_buffer_from_frames(
        frames,
        buffer_frames=buffer_frames,
        padding=padding,
    )


def crops_for_embedding(
    clip_path: Path | str,
    *,
    mode: str = "yolo_crop",
    max_read: int = 64,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
) -> list[np.ndarray]:
    """Return frame list passed to V-JEPA (full clip or YOLO crop buffer)."""
    if mode == "full_clip":
        return extract_clip_from_file(clip_path, max_frames=max_read)
    return build_crop_buffer_from_clip(
        clip_path,
        max_read=max_read,
        buffer_frames=buffer_frames,
    )


def crops_from_jpeg(path: Path | str) -> list[np.ndarray]:
    """Single JPEG → repeated crop buffer for re-embedding human labels."""
    img = cv2.imread(str(path))
    if img is None:
        return []
    crop = frame_to_person_crop(img)
    return [crop] * DEFAULT_BUFFER_FRAMES

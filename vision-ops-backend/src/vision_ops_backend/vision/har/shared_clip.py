"""Shared video clip resolution for all HAR probes."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vision_ops_backend.config import settings
from vision_ops_backend.vision.mock_videos import clip_path_for_camera
from vision_ops_backend.vision.clips import (
    _load_manifest_frames,
    _pseudo_video_from_still,
    _read_video_frames,
    _synthetic_demo_frame,
)
from vision_ops_backend.vision.paths import find_first_mp4, repo_root

logger = logging.getLogger(__name__)

HAR_SHARED_CAMERA = "har-shared"
NUM_FRAMES_DEFAULT = 16


def resolve_shared_clip_path(clip_override: str | None = None) -> Path | None:
    if clip_override:
        path = Path(clip_override)
        if path.is_file():
            return path
    configured = (settings.har_shared_clip_path or "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = repo_root() / path
        if path.is_file():
            return path
    return None


def load_har_frames_for_camera(
    camera_id: str,
    *,
    clip_path: str | None = None,
    max_frames: int = NUM_FRAMES_DEFAULT,
) -> tuple[list[np.ndarray], str]:
    """Load frames for one HAR camera (mock video, override, or fallbacks)."""
    if clip_path:
        path = Path(clip_path)
        if path.is_file():
            frames = _read_video_frames(path, max_frames)
            if frames:
                return frames, str(path)

    mock = clip_path_for_camera(camera_id)
    if mock is not None and mock.is_file():
        frames = _read_video_frames(mock, max_frames)
        if frames:
            return frames, f"mock-videos:{mock.name}"

    return load_shared_har_frames(clip_path=clip_path, max_frames=max_frames)


def load_shared_har_frames(
    *,
    clip_path: str | None = None,
    max_frames: int = NUM_FRAMES_DEFAULT,
) -> tuple[list[np.ndarray], str]:
    """Load frames for HAR inference (shared clip fallback)."""
    resolved = resolve_shared_clip_path(clip_path)
    if resolved is not None:
        frames = _read_video_frames(resolved, max_frames)
        if frames:
            return frames, str(resolved)

    frames, source = _load_manifest_frames(max_frames)
    if frames:
        return frames, source or "outputs/02_segments"

    mp4 = find_first_mp4()
    if mp4 is not None:
        frames = _read_video_frames(mp4, max_frames)
        if frames:
            return frames, str(mp4)

    still = _synthetic_demo_frame("cam-01")
    return _pseudo_video_from_still(still, max_frames), "synthetic_demo"


def shared_clip_metadata(clip_path: str | None = None) -> dict:
    resolved = resolve_shared_clip_path(clip_path)
    if resolved is not None:
        cap = cv2.VideoCapture(str(resolved))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        frames, source = load_shared_har_frames(clip_path=clip_path)
        return {
            "resolved_path": str(resolved),
            "source": source,
            "frame_count_sampled": len(frames),
            "video_frame_count": count,
        }
    frames, source = load_shared_har_frames(clip_path=clip_path)
    return {
        "resolved_path": None,
        "source": source,
        "frame_count_sampled": len(frames),
        "video_frame_count": None,
    }

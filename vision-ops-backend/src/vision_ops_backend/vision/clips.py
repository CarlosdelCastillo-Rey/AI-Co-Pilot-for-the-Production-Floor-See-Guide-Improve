"""Load frames for vision probes (InHARD, pipeline outputs, or mock stills)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from vision_ops_backend.vision.paths import (
    MOCK_STILL_IMAGES,
    find_first_mp4,
    repo_root,
    segments_manifest_path,
)

logger = logging.getLogger(__name__)


def _read_video_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    step = max(1, total // max_frames)
    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def _load_manifest_frames(max_frames: int) -> tuple[list[np.ndarray], str | None]:
    manifest_path = segments_manifest_path()
    if not manifest_path.is_file():
        return [], None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = data.get("clips") or []
    if not clips:
        return [], None
    clip = clips[0]
    clip_dir = Path(clip["path"])
    if not clip_dir.is_absolute():
        clip_dir = repo_root() / clip_dir
    paths = sorted(clip_dir.glob("frame_*.jpg"))[:max_frames]
    frames = [f for p in paths if (f := cv2.imread(str(p))) is not None]
    source = f"outputs/02_segments/{clip.get('clip_id', 'clip')}"
    return frames, source if frames else None


def _download_mock_still(camera_id: str) -> np.ndarray | None:
    url = MOCK_STILL_IMAGES.get(camera_id)
    if not url:
        return None
    try:
        with urlopen(url, timeout=15) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.warning("Mock still download failed for %s: %s", camera_id, exc)
        return None


def _pseudo_video_from_still(still: np.ndarray, count: int) -> list[np.ndarray]:
    """Slight jitter so motion-based probes have temporal signal."""
    frames: list[np.ndarray] = []
    h, w = still.shape[:2]
    for i in range(count):
        dx = int(3 * np.sin(i * 0.7))
        dy = int(3 * np.cos(i * 0.5))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(still, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)
    return frames


def _synthetic_demo_frame(camera_id: str) -> np.ndarray:
    """Local placeholder when mock URLs are unavailable (no network required)."""
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if camera_id == "cam-02":
        img[:] = (62, 58, 52)
        cv2.rectangle(img, (0, int(h * 0.72)), (w, h), (48, 46, 44), -1)
        cv2.line(img, (0, int(h * 0.72)), (w, int(h * 0.72)), (90, 88, 85), 2)
        cv2.rectangle(img, (120, 200), (380, 360), (55, 90, 140), -1)
        cv2.rectangle(img, (130, 210), (370, 340), (70, 110, 170), 2)
        cv2.circle(img, (160, 355), 28, (35, 35, 38), -1)
        cv2.circle(img, (340, 355), 28, (35, 35, 38), -1)
        cv2.putText(img, "ZONE B", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 205, 200), 2)
    else:
        img[:] = (55, 52, 48)
        cv2.rectangle(img, (80, 120), (300, 400), (75, 78, 82), -1)
        cv2.rectangle(img, (360, 160), (560, 380), (65, 68, 72), -1)
        cv2.rectangle(img, (200, 280), (420, 420), (85, 120, 95), -1)
        cv2.putText(img, "LINE 4", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 205, 200), 2)
    return img


def load_probe_frames(
    camera_id: str,
    *,
    clip_path: str | None = None,
    max_frames: int = 16,
) -> tuple[list[np.ndarray], str]:
    if clip_path:
        path = Path(clip_path)
        if path.is_file():
            frames = _read_video_frames(path, max_frames)
            if frames:
                return frames, str(path)

    frames, source = _load_manifest_frames(max_frames)
    if frames:
        return frames, source or "outputs/02_segments"

    mp4 = find_first_mp4()
    if mp4 is not None:
        frames = _read_video_frames(mp4, max_frames)
        if frames:
            return frames, str(mp4)

    still = _download_mock_still(camera_id)
    if still is None:
        still = _synthetic_demo_frame(camera_id)
        return _pseudo_video_from_still(still, max_frames), "synthetic_demo"

    return _pseudo_video_from_still(still, max_frames), f"mock_still:{camera_id}"


def load_still_for_heatmap(camera_id: str, clip_path: str | None = None) -> tuple[np.ndarray, str]:
    frames, source = load_probe_frames(camera_id, clip_path=clip_path, max_frames=1)
    if frames:
        return frames[0], source
    return _synthetic_demo_frame(camera_id), "synthetic_demo"

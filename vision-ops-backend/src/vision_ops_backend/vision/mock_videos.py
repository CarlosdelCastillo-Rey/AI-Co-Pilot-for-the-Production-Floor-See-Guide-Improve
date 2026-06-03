"""Mock MP4 clips from vision-ops-app/public/mock-videos."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from urllib.parse import quote

from vision_ops_backend.vision.har.constants import HAR_CAMERA_IDS
from vision_ops_backend.vision.paths import repo_root

_MOCK_VIDEO_DIR = repo_root() / "vision-ops-app" / "public" / "mock-videos"


def mock_videos_dir() -> Path:
    return _MOCK_VIDEO_DIR


def list_mock_video_files() -> list[Path]:
    directory = mock_videos_dir()
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )


def public_url_for_video(path: Path) -> str:
    """Browser path via Next.js public folder."""
    return f"/mock-videos/{quote(path.name)}"


def assign_videos_to_har_cameras(*, reshuffle: bool = False) -> dict[str, Path]:
    """
    Map each cam-har-* to a distinct mock video (one video per camera when possible).
    Stable assignment uses camera_id hash unless reshuffle=True.
    """
    videos = list_mock_video_files()
    if not videos:
        return {}

    camera_ids = list(HAR_CAMERA_IDS)
    if reshuffle:
        pool = videos.copy()
        random.shuffle(pool)
    else:
        pool = sorted(videos, key=lambda p: p.name)

    out: dict[str, Path] = {}
    for i, cam_id in enumerate(camera_ids):
        if reshuffle:
            out[cam_id] = pool[i % len(pool)]
        else:
            digest = hashlib.sha256(cam_id.encode()).hexdigest()
            idx = int(digest[:8], 16) % len(pool)
            # Prefer unique videos when counts match
            used = set(out.values())
            chosen = pool[idx % len(pool)]
            attempts = 0
            while chosen in used and len(used) < len(pool) and attempts < len(pool):
                idx = (idx + 1) % len(pool)
                chosen = pool[idx]
                attempts += 1
            out[cam_id] = chosen
    return out


def clip_path_for_camera(camera_id: str, assignments: dict[str, Path] | None = None) -> Path | None:
    mapping = assignments or assign_videos_to_har_cameras()
    return mapping.get(camera_id)

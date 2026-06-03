"""Resolve mock-videos paths (shared with vision-ops-app/public)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

HAR_CAMERA_IDS = (
    "cam-har-01",
    "cam-har-02",
    "cam-har-03",
    "cam-har-04",
    "cam-har-05",
)


def mock_videos_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "vision-ops-app" / "public" / "mock-videos"


def list_mock_video_files() -> list[Path]:
    directory = mock_videos_dir()
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def public_url_for_video(path: Path) -> str:
    return f"/mock-videos/{quote(path.name)}"


def assign_videos_to_har_cameras() -> dict[str, Path]:
    videos = list_mock_video_files()
    if not videos:
        return {}
    pool = sorted(videos, key=lambda p: p.name)
    out: dict[str, Path] = {}
    for cam_id in HAR_CAMERA_IDS:
        digest = hashlib.sha256(cam_id.encode()).hexdigest()
        idx = int(digest[:8], 16) % len(pool)
        chosen = pool[idx % len(pool)]
        used = set(out.values())
        attempts = 0
        while chosen in used and len(used) < len(pool) and attempts < len(pool):
            idx = (idx + 1) % len(pool)
            chosen = pool[idx]
            attempts += 1
        out[cam_id] = chosen
    return out

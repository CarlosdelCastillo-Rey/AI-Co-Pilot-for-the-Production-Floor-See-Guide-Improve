"""Repository paths for the notebook pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NOTEBOOKS_DIR.parent
OUTPUTS_DIR = NOTEBOOKS_DIR / "outputs"
CHECKPOINTS_DIR = NOTEBOOKS_DIR / "checkpoints"
MEMORY_DIR = OUTPUTS_DIR / "track_memory"
SESSIONS_DIR = OUTPUTS_DIR / "har_sessions"
MOCK_VIDEOS_DIR = REPO_ROOT / "data_sample" / "mock-videos"
MODELS_DIR = REPO_ROOT / "models"

# External full InHARD copy (Carlos Pano HD). Override with env INHARD_ROOT=...
EXTERNAL_INHARD_ROOT = Path(
    "/Volumes/Carlos Pano HD/MASTER-AI/PROYECTO INTEGRADOR/IN-HARD/01-InHARD"
)

_inhard_override: Path | None = None


def set_inhard_root(path: str | Path | None) -> None:
    """Force InHARD root for this Python session (notebooks)."""
    global _inhard_override
    _inhard_override = Path(path).expanduser().resolve() if path else None


def _is_valid_inhard_root(p: Path) -> bool:
    return (p / "Segmented" / "RGBSegmented").is_dir()


def inhard_root_candidates() -> list[Path]:
    """Search order: session override → INHARD_ROOT env → external HD → repo."""
    seen: set[str] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    if _inhard_override is not None:
        _add(_inhard_override)
    env = os.environ.get("INHARD_ROOT", "").strip()
    if env:
        _add(Path(env).expanduser())
    _add(EXTERNAL_INHARD_ROOT)
    _add(REPO_ROOT / "data_sample" / "InHARD-master" / "01-InHARD")
    _add(REPO_ROOT / "data_sample" / "InHARD" / "01-InHARD")
    return out


def ensure_notebook_paths() -> None:
    """Add notebooks/ to sys.path and create output dirs."""
    nb = str(NOTEBOOKS_DIR)
    if nb not in sys.path:
        sys.path.insert(0, nb)
    for d in (OUTPUTS_DIR, CHECKPOINTS_DIR, MEMORY_DIR, SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def find_inhard_root() -> Path | None:
    for p in inhard_root_candidates():
        if _is_valid_inhard_root(p):
            return p
    return None


def inhard_segmented_dir(root: Path | None = None) -> Path | None:
    root = root or find_inhard_root()
    if root is None:
        return None
    seg = root / "Segmented" / "RGBSegmented"
    return seg if seg.is_dir() else None


def list_mock_videos() -> list[Path]:
    if not MOCK_VIDEOS_DIR.is_dir():
        return []
    return sorted(MOCK_VIDEOS_DIR.glob("*.mp4"))


def default_mock_video() -> Path | None:
    vids = list_mock_videos()
    return vids[0] if vids else None


def yolo_weights_path() -> Path:
    for p in (
        NOTEBOOKS_DIR / "yolov8n.pt",
        REPO_ROOT / "notebooks" / "yolov8n.pt",
    ):
        if p.is_file():
            return p
    return NOTEBOOKS_DIR / "yolov8n.pt"

"""Paths, logging, JSON I/O, and env helpers for VisionOps Fase 0 notebooks."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from loguru import logger

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPTS_DIR.parent


def repo_root() -> Path:
    """Repository root (parent of scripts/)."""
    return _REPO_ROOT


def scripts_dir() -> Path:
    return _SCRIPTS_DIR


def ensure_scripts_on_path() -> Path:
    """Add scripts/ to sys.path so notebooks can `from _common.io import ...`."""
    scripts = str(_SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return _SCRIPTS_DIR


def stage_output_dir(stage: str) -> Path:
    """Create and return outputs/{stage}/ under repo root."""
    out = repo_root() / "outputs" / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_dotenv_repo() -> bool:
    """Load .env from repository root if present."""
    env_path = repo_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
        return True
    load_dotenv()
    return False


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inhard_segmented_rgb_dir() -> Path:
    return repo_root() / "data_sample" / "InHARD-master" / "01-InHARD" / "Segmented" / "RGBSegmented"


def find_first_mp4(root: Path | None = None, max_depth: int = 6) -> Path | None:
    """Return first .mp4 found under root (depth-limited walk)."""
    if root is None:
        root = inhard_segmented_rgb_dir()
    if not root.is_dir():
        return None

    def _walk(path: Path, depth: int) -> Path | None:
        if depth > max_depth:
            return None
        try:
            entries = sorted(path.iterdir(), key=lambda p: p.name)
        except OSError:
            return None
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == ".mp4":
                return entry
        for entry in entries:
            if entry.is_dir():
                found = _walk(entry, depth + 1)
                if found is not None:
                    return found
        return None

    return _walk(root, 0)


def resolve_source_video(
    source: str | int | None = None,
    *,
    prefer_inhard: bool = True,
) -> str | int:
    """
    Resolve video source for capture notebook.

    Order: explicit source -> first InHARD mp4 -> webcam index 0.
    """
    if source is not None and source != "" and source != "auto":
        return source

    if prefer_inhard:
        clip = find_first_mp4()
        if clip is not None:
            logger.info("Using InHARD clip: {}", clip)
            return str(clip)

    logger.warning("No local .mp4 found; falling back to webcam index 0")
    return 0


def default_roi() -> dict[str, float]:
    """Normalized ROI placeholder (x1, y1, x2, y2) in [0, 1]."""
    return {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}


def bbox_centroid(bbox: list[float]) -> tuple[float, float]:
    """Centroid from bbox [x1, y1, x2, y2] (pixel or normalized)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_roi(cx: float, cy: float, roi: dict[str, float], frame_w: int, frame_h: int) -> bool:
    """Check if centroid lies inside ROI (ROI normalized, centroid in pixels)."""
    nx = cx / max(frame_w, 1)
    ny = cy / max(frame_h, 1)
    return roi["x1"] <= nx <= roi["x2"] and roi["y1"] <= ny <= roi["y2"]


def env_or_none(key: str) -> str | None:
    val = os.environ.get(key)
    return val if val else None

"""Persist and load per-camera vision probe results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.vision.paths import (
    baseline_embedding_path,
    camera_artifact_dir,
    embedding_path,
    heatmap_overlay_path,
    heatmap_path,
    last_probe_path,
    preview_path,
    still_path,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_last_probe(camera_id: str) -> dict[str, Any] | None:
    path = last_probe_path(camera_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_probe_result(camera_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    out_dir = camera_artifact_dir(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "camera_id": camera_id, "updated_at": _utc_now()}
    write_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    last_probe_path(camera_id).write_text(write_json, encoding="utf-8")
    return payload


def save_heatmap_artifacts(camera_id: str, heatmap_bgr: np.ndarray, overlay_bgr: np.ndarray) -> None:
    out_dir = camera_artifact_dir(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(heatmap_path(camera_id)), heatmap_bgr)
    cv2.imwrite(str(heatmap_overlay_path(camera_id)), overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def save_embedding(camera_id: str, embedding: np.ndarray, *, as_baseline: bool = False) -> None:
    out_dir = camera_artifact_dir(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path(camera_id), embedding)
    if as_baseline:
        np.save(baseline_embedding_path(camera_id), embedding)


def load_baseline_embedding(camera_id: str) -> np.ndarray | None:
    path = baseline_embedding_path(camera_id)
    if path.is_file():
        return np.load(path)
    path = embedding_path(camera_id)
    if path.is_file():
        return np.load(path)
    return None


def overlays_for_camera(camera_id: str) -> list[dict[str, Any]]:
    probe = load_last_probe(camera_id)
    if not probe:
        return []
    overlay = probe.get("overlay")
    if isinstance(overlay, dict):
        return [overlay]
    return probe.get("overlays") or []


def heatmap_available(camera_id: str) -> bool:
    return heatmap_overlay_path(camera_id).is_file() or heatmap_path(camera_id).is_file()


def still_available(camera_id: str) -> bool:
    return still_path(camera_id).is_file()


def preview_available(camera_id: str) -> bool:
    return preview_path(camera_id).is_file()


def save_still(camera_id: str, frame: np.ndarray) -> None:
    out_dir = camera_artifact_dir(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(still_path(camera_id)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_preview(camera_id: str, frame: np.ndarray) -> None:
    out_dir = camera_artifact_dir(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path(camera_id)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

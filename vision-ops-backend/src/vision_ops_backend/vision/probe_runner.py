"""Run vision probes for cam-01 (heatmap) and cam-02 (V-JEPA)."""

from __future__ import annotations

from typing import Any

import numpy as np

from vision_ops_backend.vision import CAM_ASSEMBLY, CAM_WAREHOUSE
from vision_ops_backend.vision.clips import load_probe_frames, load_still_for_heatmap
from vision_ops_backend.vision.dinov3_heatmap import generate_heatmap
from vision_ops_backend.vision.preview import draw_overlay_on_frame
from vision_ops_backend.vision.store import (
    load_baseline_embedding,
    save_embedding,
    save_heatmap_artifacts,
    save_preview,
    save_probe_result,
    save_still,
    still_available,
)
from vision_ops_backend.vision.vjepa_probe import run_vjepa_probe


def run_dinov_probe(
    camera_id: str = CAM_ASSEMBLY,
    *,
    clip_path: str | None = None,
) -> dict[str, Any]:
    frame, source = load_still_for_heatmap(camera_id, clip_path=clip_path)
    result = generate_heatmap(frame)
    save_heatmap_artifacts(camera_id, result["heatmap_bgr"], result["overlay_bgr"])
    save_still(camera_id, frame)
    save_preview(camera_id, result["overlay_bgr"])

    payload: dict[str, Any] = {
        "mode": "dinov3_heatmap",
        "status": "ok",
        "source": source,
        "backend": result["backend"],
        "overlay": result["overlay"],
        "artifacts": {
            "heatmap": f"/api/vision/artifacts/{camera_id}/heatmap",
            "overlay": f"/api/vision/artifacts/{camera_id}/overlay",
            "preview": f"/api/vision/artifacts/{camera_id}/preview",
        },
    }
    return save_probe_result(camera_id, payload)


def run_vjepa_camera_probe(
    camera_id: str = CAM_WAREHOUSE,
    *,
    clip_path: str | None = None,
    set_baseline: bool = False,
) -> dict[str, Any]:
    frames, source = load_probe_frames(camera_id, clip_path=clip_path)
    baseline = None if set_baseline else load_baseline_embedding(camera_id)
    probe = run_vjepa_probe(frames, baseline=baseline)

    embedding: np.ndarray = probe.pop("embedding")
    save_embedding(camera_id, embedding, as_baseline=set_baseline or baseline is None)

    if frames:
        still = frames[len(frames) // 2]
        save_still(camera_id, still)
        overlay = probe.get("overlay")
        if isinstance(overlay, dict):
            save_preview(camera_id, draw_overlay_on_frame(still, overlay))
        else:
            save_preview(camera_id, still)

    payload: dict[str, Any] = {
        "mode": "vjepa_anomaly",
        "status": probe.get("status", "ok"),
        "source": source,
        "backend": probe.get("backend"),
        "meta": probe.get("meta"),
        "anomaly_score": probe.get("anomaly_score"),
        "severity": probe.get("severity"),
        "event": probe.get("event"),
        "overlay": probe.get("overlay"),
        "baseline_set": set_baseline or baseline is None,
        "artifacts": {
            "preview": f"/api/vision/artifacts/{camera_id}/preview",
            "still": f"/api/vision/artifacts/{camera_id}/still",
        },
    }
    return save_probe_result(camera_id, payload)


def ensure_camera_still(camera_id: str) -> None:
    """Cache a local JPEG so the Live tile never depends on external mock URLs."""
    if still_available(camera_id):
        return
    frame, _ = load_still_for_heatmap(camera_id)
    save_still(camera_id, frame)


def run_probe(
    camera_id: str,
    mode: str,
    *,
    clip_path: str | None = None,
    set_baseline: bool = False,
) -> dict[str, Any]:
    if mode == "dinov3_heatmap" or camera_id == CAM_ASSEMBLY:
        return run_dinov_probe(camera_id, clip_path=clip_path)
    if mode == "vjepa_anomaly" or camera_id == CAM_WAREHOUSE:
        return run_vjepa_camera_probe(
            camera_id, clip_path=clip_path, set_baseline=set_baseline
        )
    raise ValueError(f"Unknown camera/mode: {camera_id} / {mode}")

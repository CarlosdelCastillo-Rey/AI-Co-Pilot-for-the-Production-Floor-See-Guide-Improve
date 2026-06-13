"""Build runtime camera payloads for in-process merge (no HTTP)."""

from __future__ import annotations

from typing import Any

from vision_ops_backend.config import settings
from vision_ops_backend.industrial_cameras import list_industrial_cameras
from vision_ops_backend.webcam import WebcamCapture


def collect_runtime_camera_map(webcam: WebcamCapture | None = None) -> dict[str, dict[str, Any]]:
    """Return backend camera id → runtime dict (streamUrl, overlays, visionProbe, …)."""
    out: dict[str, dict[str, Any]] = {}

    if webcam is not None:
        base = settings.public_api_base.rstrip("/")
        is_live = webcam.is_running and webcam.error is None
        status = "live" if is_live else "offline"
        face = webcam.face_engine
        face_status = ""
        if face and face.is_ready:
            face_status = " | FACE OK" if face.is_enrolled else " | ENROLL FACE"

        payload: dict[str, Any] = {
            "id": settings.camera_id,
            "name": "Camera 01 - Webcam (dev)",
            "location": "Local / MacBook",
            "status": status,
            "image": "",
            "coords": f"WEBCAM | DEV{face_status}",
            "overlays": webcam.get_overlays() if is_live else [],
            "error": webcam.error or (face.error if face else None),
        }
        if is_live:
            payload["streamUrl"] = f"{base}/api/cameras/{settings.camera_id}/stream"
        out[settings.camera_id] = payload

    if settings.vision_enabled:
        for cam in list_industrial_cameras():
            cam_id = cam.get("id")
            if cam_id:
                out[str(cam_id)] = cam

    return out

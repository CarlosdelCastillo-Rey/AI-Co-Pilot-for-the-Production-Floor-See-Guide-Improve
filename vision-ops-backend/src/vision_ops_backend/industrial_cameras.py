"""Mock HAR camera cards (cam-har-*) with mock-videos and probe overlays."""

from __future__ import annotations

from typing import Any

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import HAR_MODELS, HarModelSpec
from vision_ops_backend.vision.har.live_stream import get_har_live_manager
from vision_ops_backend.vision.mock_videos import assign_videos_to_har_cameras, public_url_for_video
from vision_ops_backend.vision.store import load_last_probe, overlays_for_camera, still_available


def build_har_camera(spec: HarModelSpec, video_assignments: dict | None = None) -> dict[str, Any]:
    camera_id = spec.camera_id
    probe = load_last_probe(camera_id)
    assignments = video_assignments or assign_videos_to_har_cameras()
    mock_path = assignments.get(camera_id)
    video_url = public_url_for_video(mock_path) if mock_path else None

    overlays = overlays_for_camera(camera_id)
    if not overlays:
        overlays = [
            {
                "type": "machine",
                "label": f"Run HAR probe — {spec.label}",
                "top": "8%",
                "left": "8%",
                "width": "50%",
                "height": "16%",
                "variant": "tertiary",
            }
        ]

    vision_tag = f" | HAR {spec.label}"
    if probe:
        pred = probe.get("prediction") or {}
        label = pred.get("label")
        conf = pred.get("confidence")
        if label is not None and conf is not None:
            vision_tag = f" | {label} {float(conf):.0%}"

    image_url = video_url or "/mock-videos/"
    if still_available(camera_id):
        image_url = f"/api/vision/artifacts/{camera_id}/still"

    base = settings.public_api_base.rstrip("/")
    payload: dict[str, Any] = {
        "id": camera_id,
        "name": spec.display_name,
        "location": "HAR Lab / Mock video (live)",
        "image": image_url,
        "status": "live",
        "coords": f"42.3605° N, 71.0592° W | HAR{vision_tag}",
        "overlays": overlays,
        "visionProbe": probe,
        "inferenceModel": spec.model_id.replace("-", "_"),
        "inferenceTask": "activity",
        "videoUrl": probe.get("videoUrl") if probe else video_url,
    }
    if settings.har_live_enabled:
        payload["streamUrl"] = f"{base}/api/cameras/{camera_id}/stream"
        live_runtime = get_har_live_manager().runtime_for_camera(camera_id)
        if live_runtime:
            if live_runtime.get("visionProbe"):
                payload["visionProbe"] = live_runtime["visionProbe"]
            if live_runtime.get("overlays"):
                payload["overlays"] = live_runtime["overlays"]
            payload["harLive"] = live_runtime.get("harLive")
    return payload


def list_industrial_cameras() -> list[dict[str, Any]]:
    if not settings.har_enabled:
        return []
    assignments = assign_videos_to_har_cameras()
    return [build_har_camera(spec, assignments) for spec in HAR_MODELS]

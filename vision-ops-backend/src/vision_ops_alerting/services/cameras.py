from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_backend.camera_runtime import collect_runtime_camera_map
from vision_ops_backend.webcam import WebcamCapture
from vision_ops_alerting.db.models import Camera, Event, HealthMetricSample

_runtime_webcam: WebcamCapture | None = None


def set_runtime_webcam(webcam: WebcamCapture | None) -> None:
    global _runtime_webcam
    _runtime_webcam = webcam

DEFAULT_OVERLAYS: dict[str, list[dict[str, Any]]] = {}

MODEL_LABELS = {
    "dinov3": {"model": "DINOv3", "task": "patch_similarity"},
    "vjepa2": {"model": "V-JEPA 2", "task": "anomaly"},
    "yolov8": {"model": "YOLOv8 + ByteTrack", "task": "tracking"},
    "v2-vjepa": {"model": "V-JEPA 2", "task": "activity"},
    "v2-dinov2": {"model": "DINOv2", "task": "activity"},
}


def _model_badge(camera: Camera) -> dict[str, str] | None:
    key = (camera.inference_model or "").lower()
    if key in MODEL_LABELS:
        return MODEL_LABELS[key]
    if camera.inference_model:
        return {"model": camera.inference_model, "task": camera.inference_task or ""}
    return None


def _camera_config(camera: Camera) -> dict[str, Any]:
    if not camera.config_json:
        return {}
    try:
        return json.loads(camera.config_json)
    except json.JSONDecodeError:
        return {}


def camera_to_dict(camera: Camera, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = runtime or {}
    cfg = _camera_config(camera)
    overlays = runtime.get("overlays")
    if not overlays:
        overlays = DEFAULT_OVERLAYS.get(camera.id, DEFAULT_OVERLAYS.get(camera.backend_camera_id or "", []))

    status = runtime.get("status") or camera.status
    if camera.enabled is False:
        status = "offline"

    payload: dict[str, Any] = {
        "id": camera.id,
        "name": camera.name,
        "location": camera.location,
        "zone": camera.zone,
        "sourceType": camera.source_type,
        "streamUrl": runtime.get("streamUrl") or camera.stream_url,
        "coords": runtime.get("coords") or camera.coords or "",
        "image": runtime.get("image") or camera.image_url or "",
        "status": status,
        "enabled": camera.enabled,
        "inferenceModel": camera.inference_model,
        "inferenceTask": camera.inference_task,
        "modelBadge": _model_badge(camera),
        "overlays": overlays,
        "visionProbe": runtime.get("visionProbe"),
        "heatmapUrl": runtime.get("heatmapUrl"),
        "previewUrl": runtime.get("previewUrl"),
        "videoUrl": runtime.get("videoUrl") or cfg.get("mockVideoUrl"),
        "error": runtime.get("error"),
        "backendCameraId": camera.backend_camera_id,
        "sortOrder": camera.sort_order,
        "anomalyScore": runtime.get("anomalyScore"),
    }
    if runtime.get("streamUrl"):
        payload["streamUrl"] = runtime["streamUrl"]
    return payload


async def fetch_backend_cameras() -> dict[str, dict[str, Any]]:
    return collect_runtime_camera_map(_runtime_webcam)


def _merge_runtime(camera: Camera, backend: dict[str, dict[str, Any]]) -> dict[str, Any]:
    backend_id = camera.backend_camera_id or camera.id
    runtime = dict(backend.get(backend_id, {}))
    if runtime.get("visionProbe"):
        probe = runtime["visionProbe"]
        score = probe.get("anomaly_score")
        if score is not None:
            runtime["anomalyScore"] = float(score)
    return runtime


async def list_cameras_merged(db: Session) -> list[dict[str, Any]]:
    backend = await fetch_backend_cameras()
    rows = (
        db.query(Camera)
        .filter(Camera.enabled.is_(True))
        .order_by(Camera.sort_order.asc(), Camera.created_at.asc())
        .all()
    )
    return [camera_to_dict(cam, _merge_runtime(cam, backend)) for cam in rows]


def _avg_latency_ms(db: Session) -> float | None:
    row = (
        db.query(func.avg(HealthMetricSample.latency_ms))
        .filter(
            HealthMetricSample.service == "vision_backend",
            HealthMetricSample.latency_ms.isnot(None),
        )
        .scalar()
    )
    return round(float(row), 1) if row else None


def _prev_avg_latency_ms(db: Session) -> float | None:
    rows = (
        db.query(HealthMetricSample.latency_ms)
        .filter(
            HealthMetricSample.service == "vision_backend",
            HealthMetricSample.latency_ms.isnot(None),
        )
        .order_by(HealthMetricSample.recorded_at.desc())
        .offset(5)
        .limit(5)
        .all()
    )
    vals = [r[0] for r in rows if r[0] is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _events_today(db: Session) -> int:
    today = date.today()
    start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)
    return db.query(Event).filter(Event.occurred_at >= start, Event.occurred_at <= end).count()


def _events_avg_7d(db: Session) -> float:
    from datetime import timedelta

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    count = db.query(Event).filter(Event.occurred_at >= start).count()
    return round(count / 7, 1)


async def live_stats(db: Session, cameras: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from vision_ops_alerting.services.plant_settings import get_plant_config

    config = get_plant_config(db)
    if cameras is None:
        cameras = await list_cameras_merged(db)

    total = len(cameras)
    online = sum(1 for c in cameras if c.get("status") == "live")
    healthy = online == total and total > 0

    events_today = _events_today(db)
    avg_7d = _events_avg_7d(db)
    delta_events = round(events_today - avg_7d)
    events_delta = f"+{delta_events} vs avg" if delta_events > 0 else f"{delta_events} vs avg" if delta_events < 0 else "on avg"

    latency = _avg_latency_ms(db)
    prev_latency = _prev_avg_latency_ms(db)
    latency_delta = ""
    if latency is not None and prev_latency is not None:
        diff = round(latency - prev_latency)
        latency_delta = f"{diff:+d}ms" if diff != 0 else "stable"
    elif latency is not None:
        latency_delta = "stable"

    base = config.inference_base_per_camera * online if online else 0
    probe_bonus = sum(1 for c in cameras if c.get("visionProbe")) * config.inference_probe_bonus
    inferences = base + probe_bonus + (events_today * config.inference_event_multiplier)
    inferences = max(inferences, online * config.inference_min_per_camera)

    prev_events = max(1, int(round(avg_7d)))
    inf_trend_pct = round((events_today / prev_events - 1) * 100, 1)
    inf_trend = f"{inf_trend_pct:+.1f}% vs 7d avg" if inf_trend_pct != 0 else "on 7d avg"

    return {
        "camerasOnline": {"current": online, "total": total, "healthy": healthy},
        "inferencesPerMin": {"value": inferences, "trend": inf_trend},
        "eventsToday": {"value": events_today, "delta": events_delta},
        "avgEdgeLatencyMs": {
            "value": int(latency) if latency is not None else 84,
            "delta": latency_delta or "—",
        },
    }


def parse_config(camera: Camera) -> dict[str, Any]:
    if not camera.config_json:
        return {}
    try:
        return json.loads(camera.config_json)
    except json.JSONDecodeError:
        return {}


def apply_config(camera: Camera, config: dict[str, Any]) -> None:
    camera.config_json = json.dumps(config) if config else None

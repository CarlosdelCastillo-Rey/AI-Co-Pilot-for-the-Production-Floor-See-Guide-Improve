"""In-process HAR activity log ingest (no HTTP)."""

from __future__ import annotations

import logging
from typing import Any

from vision_ops_backend.config import settings

logger = logging.getLogger(__name__)


def format_tracked_detections(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tr in tracks:
        entry: dict[str, Any] = {
            "track_id": tr.get("track_id"),
            "bbox": tr.get("bbox"),
            "det_conf": tr.get("det_conf"),
        }
        if tr.get("action_label"):
            entry["action_label"] = tr["action_label"]
        if tr.get("action_confidence") is not None:
            entry["action_confidence"] = tr["action_confidence"]
        if tr.get("global_person_id"):
            entry["global_person_id"] = tr["global_person_id"]
        if tr.get("display_name"):
            entry["display_name"] = tr["display_name"]
        out.append(entry)
    return out


def format_detections(
    boxes: list[tuple[int, int, int, int, float]],
    *,
    action_label: str | None = None,
    action_confidence: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes):
        entry: dict[str, Any] = {
            "track_index": i,
            "bbox": [x1, y1, x2, y2],
            "det_conf": round(float(det_conf), 4),
        }
        if action_label and i == 0:
            entry["action_label"] = action_label
            if action_confidence is not None:
                entry["action_confidence"] = round(float(action_confidence), 4)
        out.append(entry)
    return out


def build_activity_entry(
    *,
    camera_id: str,
    model_id: str,
    prediction: dict[str, Any],
    source: str,
    backend: str | None = None,
    device: str | None = None,
    detections: list[dict[str, Any]] | None = None,
    frame_index: int | None = None,
    video_offset_sec: float | None = None,
    video_name: str | None = None,
    clip_url: str | None = None,
    session_id: str | None = None,
    new_session: bool = False,
    infer_ms: float | None = None,
    snapshot_url: str | None = None,
    model_label: str | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "model_id": model_id,
        "source": source,
        "prediction": prediction,
        "predicted_label": prediction.get("label"),
        "confidence": prediction.get("confidence"),
        "class_index": prediction.get("class_index"),
        "top_k": prediction.get("top_k"),
        "backend": backend,
        "device": device,
        "detections": detections or [],
        "person_count": len(detections or []),
        "frame_index": frame_index,
        "video_offset_sec": video_offset_sec,
        "video_name": video_name,
        "videoUrl": clip_url,
        "clip_url": clip_url,
        "session_id": session_id,
        "new_session": new_session,
        "infer_ms": infer_ms,
        "snapshot_url": snapshot_url,
        "model_label": model_label,
        "hyperparams": hyperparams,
    }


def ingest_har_activity(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not settings.har_activity_ingest_enabled:
        return None
    try:
        from vision_ops_alerting.db.session import SessionLocal
        from vision_ops_alerting.services.har_activity_store import record_activity_batch
        from vision_ops_alerting.services.har_alert_dispatcher import dispatch_after_ingest

        with SessionLocal() as db:
            recorded = record_activity_batch(db, [entry])
            event_ids = dispatch_after_ingest(db, recorded)
            db.commit()
            return {
                "status": "ok",
                "recorded": len(recorded),
                "promotedEventIds": event_ids,
            }
    except Exception as exc:
        logger.warning("HAR activity ingest failed: %s", exc)
    return None

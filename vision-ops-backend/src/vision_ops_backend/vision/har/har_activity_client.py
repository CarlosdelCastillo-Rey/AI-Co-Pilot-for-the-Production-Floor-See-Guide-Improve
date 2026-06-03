"""POST integral HAR activity logs to vision-ops-alerting."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from vision_ops_backend.config import settings

logger = logging.getLogger(__name__)


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
    }


def ingest_har_activity(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not settings.har_activity_ingest_enabled:
        return None
    base = settings.alerting_api_url.rstrip("/")
    url = f"{base}/api/har/activity"
    body = json.dumps({"entry": entry}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        logger.warning("HAR activity ingest HTTP %s: %s", exc.code, detail)
    except Exception as exc:
        logger.warning("HAR activity ingest failed: %s", exc)
    return None

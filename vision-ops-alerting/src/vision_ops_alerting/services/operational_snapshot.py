"""Compact operational snapshot for the VisionOps advisor (DB + vision backend)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import AlertRule, Camera, Event
from vision_ops_alerting.services.event_workflow import build_timeline_stats
from vision_ops_alerting.services.events import event_to_timeline_dict
from vision_ops_alerting.services.plant_settings import get_plant_config
from vision_ops_alerting.services.timeline_summary import build_shift_summary


def _probe_vision_backend() -> dict[str, Any]:
    base = settings.vision_backend_url.rstrip("/")
    out: dict[str, Any] = {"reachable": False, "health": None, "visionStatus": None}
    try:
        with httpx.Client(timeout=2.5) as client:
            health = client.get(f"{base}/health")
            out["reachable"] = health.status_code == 200
            out["health"] = health.json() if health.status_code == 200 else None
            vs = client.get(f"{base}/api/vision/status")
            if vs.status_code == 200:
                data = vs.json()
                if isinstance(data, dict):
                    out["visionStatus"] = {
                        k: data[k]
                        for k in ("ready", "cameras", "anomaly_score", "status")
                        if k in data
                    }
                    if not out["visionStatus"] and len(data) <= 12:
                        out["visionStatus"] = data
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def build_operational_snapshot(db: Session, *, page: str | None = None) -> dict[str, Any]:
    """Single JSON bundle: plant, floor, cameras, open alerts, rules, backend health."""
    today = date.today().isoformat()
    plant = get_plant_config(db)
    summary = build_shift_summary(db, today)
    stats = build_timeline_stats(db, today)

    open_events = (
        db.query(Event)
        .filter((Event.resolution_status == "OPEN") | (Event.resolution_status.is_(None)))
        .order_by(Event.occurred_at.desc())
        .limit(12)
        .all()
    )

    cameras = (
        db.query(Camera)
        .filter(Camera.enabled.is_(True))
        .order_by(Camera.sort_order, Camera.name)
        .all()
    )
    cam_rows = [
        {
            "id": c.id,
            "name": c.name,
            "zone": c.zone,
            "status": c.status,
            "model": c.inference_model,
            "location": c.location,
        }
        for c in cameras
    ]

    rules_active = (
        db.query(func.count(AlertRule.id))
        .filter(AlertRule.enabled.is_(True))
        .scalar()
        or 0
    )
    rules_total = db.query(func.count(AlertRule.id)).scalar() or 0

    backend = _probe_vision_backend()

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "page": page or "unknown",
        "plant": {
            "siteName": plant.site_name,
            "shiftHours": plant.shift_hours,
            "lineCostPerMinute": plant.line_cost_per_minute,
            "materialCostPerUnit": plant.material_cost_per_unit,
        },
        "floor": {
            "allClear": stats.get("allClear", True),
            "openCount": stats.get("openCount", 0),
            "openCriticalCount": stats.get("openCriticalCount", 0),
            "uptime": summary.get("uptime"),
            "incidentCount": summary.get("incidentCount"),
            "totalEventsToday": summary.get("totalEvents", 0),
        },
        "cameras": {
            "total": len(cam_rows),
            "live": sum(1 for c in cam_rows if c["status"] == "live"),
            "items": cam_rows[:8],
        },
        "openAlerts": [event_to_timeline_dict(e) for e in open_events],
        "alertRules": {"active": rules_active, "total": rules_total},
        "visionBackend": backend,
        "emailDryRun": settings.dry_run,
    }


def snapshot_to_prompt_text(snapshot: dict[str, Any], max_chars: int = 6000) -> str:
    text = json.dumps(snapshot, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "…(truncated)"
    return text

"""Strands/MCP-style DB tools for the VisionOps advisor agent."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session
from strands import tool

from vision_ops_alerting.db.models import Camera, Event
from vision_ops_alerting.services.events import event_to_timeline_dict
from vision_ops_alerting.services.operational_snapshot import build_operational_snapshot
from vision_ops_alerting.services.timeline_summary import build_shift_summary


def _tool_result(data: Any) -> dict:
    import json

    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > 4000:
        text = text[:3980] + "…"
    return {"status": "success", "content": [{"text": text}]}


def make_advisor_tools(db: Session) -> list:
    """Factory: tools close over the request DB session (MCP-equivalent for SQLite)."""

    @tool
    def query_operational_snapshot() -> dict:
        """
        Full plant snapshot: cameras, open alerts, uptime, vision backend reachability.

        Returns:
            JSON operational snapshot from SQLite and health probes.
        """
        snap = build_operational_snapshot(db)
        return _tool_result(snap)

    @tool
    def query_open_alerts(limit: int = 8) -> dict:
        """
        List open timeline incidents (not resolved).

        Args:
            limit: Max rows (default 8, max 20).

        Returns:
            Open alerts with severity, camera, case type, title.
        """
        lim = max(1, min(limit, 20))
        rows = (
            db.query(Event)
            .filter((Event.resolution_status == "OPEN") | (Event.resolution_status.is_(None)))
            .order_by(Event.occurred_at.desc())
            .limit(lim)
            .all()
        )
        return _tool_result([event_to_timeline_dict(e) for e in rows])

    @tool
    def query_cameras_status() -> dict:
        """
        Enabled cameras with status, zone, and inference model.

        Returns:
            Camera registry rows from the alerting database.
        """
        cameras = (
            db.query(Camera)
            .filter(Camera.enabled.is_(True))
            .order_by(Camera.sort_order, Camera.name)
            .all()
        )
        items = [
            {
                "id": c.id,
                "name": c.name,
                "zone": c.zone,
                "status": c.status,
                "inferenceModel": c.inference_model,
                "location": c.location,
            }
            for c in cameras
        ]
        return _tool_result({"total": len(items), "live": sum(1 for i in items if i["status"] == "live"), "cameras": items})

    @tool
    def query_shift_summary(event_date: str = "") -> dict:
        """
        Shift KPIs: uptime, incidents, workflow counts for a calendar day (YYYY-MM-DD).

        Args:
            event_date: ISO date; empty string uses today UTC.

        Returns:
            Merged shift summary and timeline workflow stats.
        """
        from vision_ops_alerting.services.event_workflow import build_timeline_stats

        target = event_date.strip() or None
        summary = build_shift_summary(db, target)
        stats = build_timeline_stats(db, target)
        return _tool_result({**summary, **stats})

    return [query_operational_snapshot, query_open_alerts, query_cameras_status, query_shift_summary]

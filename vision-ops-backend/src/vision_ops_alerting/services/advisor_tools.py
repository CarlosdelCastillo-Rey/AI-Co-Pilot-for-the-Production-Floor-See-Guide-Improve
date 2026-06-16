"""Strands/MCP-style DB tools for the VisionOps advisor agent."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session
from strands import tool

from vision_ops_alerting.db.models import Camera, Event
from vision_ops_alerting.services.events import event_to_timeline_dict
from vision_ops_alerting.services.har_activity_store import (
    activity_summary,
    build_all_cameras_har_dashboard,
    global_har_summary,
    list_activity_logs,
)
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

    @tool
    def query_har_activity_summary(camera_id: str = "", event_date: str = "") -> dict:
        """
        HAR action log summary for one camera: label counts, non-assembly rate.

        Args:
            camera_id: e.g. cam-har-01 (required).
            event_date: YYYY-MM-DD or empty for today.
        """
        cid = camera_id.strip()
        if not cid:
            return _tool_result({"error": "camera_id required"})
        return _tool_result(activity_summary(db, camera_id=cid, target_date=event_date.strip() or None))

    @tool
    def query_har_activity_logs(camera_id: str = "", limit: int = 30) -> dict:
        """
        Recent integral HAR activity log rows for one camera.

        Args:
            camera_id: e.g. cam-har-01.
            limit: Max rows (default 30, max 80).
        """
        cid = camera_id.strip()
        if not cid:
            return _tool_result({"error": "camera_id required"})
        lim = max(1, min(limit, 80))
        logs, total = list_activity_logs(db, camera_id=cid, limit=lim)
        return _tool_result({"total": total, "logs": logs})

    @tool
    def query_har_plant_summary(hours: int = 24) -> dict:
        """
        Plant-wide HAR inference counts across all cameras in the last N hours.
        """
        h = max(1, min(hours, 72))
        return _tool_result(global_har_summary(db, hours=h))

    @tool
    def query_all_cameras_har_dashboard(hours: int = 24) -> dict:
        """
        All HAR live cameras (cam-har-01…02): latest detected action per feed,
        today's inference counts, non-assembly rates, and open HAR timeline incidents.

        Use this when the user asks for a summary across all cameras, actions by camera,
        or HAR incidents on the Live page.

        Args:
            hours: Rolling window for plant-wide inference counts (default 24).
        """
        h = max(1, min(hours, 72))
        return _tool_result(build_all_cameras_har_dashboard(db, hours=h))

    return [
        query_operational_snapshot,
        query_open_alerts,
        query_cameras_status,
        query_shift_summary,
        query_har_activity_summary,
        query_har_activity_logs,
        query_har_plant_summary,
        query_all_cameras_har_dashboard,
    ]


def make_advisor_action_tools(http_base: str = "http://127.0.0.1:8000") -> list:
    """Factory: action/write tools that mutate system state via the REST API."""
    import httpx

    def _patch(path: str, **kw: Any) -> Any:
        with httpx.Client(timeout=15) as c:
            r = c.patch(f"{http_base}{path}", json=kw or None)
            r.raise_for_status()
            return r.json()

    def _post(path: str, **kw: Any) -> Any:
        with httpx.Client(timeout=15) as c:
            r = c.post(f"{http_base}{path}", json=kw or None)
            r.raise_for_status()
            return r.json()

    def _get(path: str, **params: Any) -> Any:
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{http_base}{path}", params=params or None)
            r.raise_for_status()
            return r.json()

    @tool
    def acknowledge_event(event_id: str) -> dict:
        """
        Acknowledge an open timeline incident (mark as seen, not resolved).

        Args:
            event_id: UUID of the timeline event to acknowledge.

        Returns:
            Updated event status.
        """
        return _tool_result(_patch(f"/api/timeline/events/{event_id}/ack"))

    @tool
    def resolve_event(event_id: str, resolution_notes: str = "") -> dict:
        """
        Resolve a timeline incident (mark as closed/fixed).

        Args:
            event_id: UUID of the timeline event.
            resolution_notes: Optional explanation of how it was resolved.

        Returns:
            Updated event status.
        """
        return _tool_result(_patch(f"/api/timeline/events/{event_id}/resolve", notes=resolution_notes))

    @tool
    def dismiss_event(event_id: str) -> dict:
        """
        Dismiss a timeline incident (mark as not actionable / false positive).

        Args:
            event_id: UUID of the timeline event to dismiss.

        Returns:
            Updated event status.
        """
        return _tool_result(_patch(f"/api/timeline/events/{event_id}/dismiss"))

    @tool
    def toggle_alert_rule(rule_id: str, enabled: bool) -> dict:
        """
        Enable or disable an alert rule.

        Args:
            rule_id: The rule's ID or case type string.
            enabled: True to enable, False to disable.

        Returns:
            Updated rule object.
        """
        return _tool_result(_patch(f"/api/alerts/rules/{rule_id}/toggle", enabled=enabled))

    @tool
    def rename_person(global_person_id: str, new_display_name: str) -> dict:
        """
        Assign or update the display name of a person in the HAR registry.

        Args:
            global_person_id: The person's global_person_id UUID.
            new_display_name: Human-readable name to assign (e.g. "Carlos").

        Returns:
            Confirmation with updated name.
        """
        return _tool_result(_patch(f"/api/har/v2/persons/{global_person_id}", display_name=new_display_name))

    @tool
    def trigger_har_probe(model_id: str = "v2-vjepa") -> dict:
        """
        Run a fresh HAR inference probe on the specified model using the current bench video.

        Args:
            model_id: HAR model to probe (e.g. "v2-vjepa", "v2-dinov2"). Default: v2-vjepa.

        Returns:
            Inference result with predicted action and confidence.
        """
        return _tool_result(_post(f"/api/vision/har/{model_id}/probe"))

    @tool
    def send_test_alert(case_type: str = "NO_ACTIVITY", channel: str = "email") -> dict:
        """
        Send a test alert notification to verify alerting is working.

        Args:
            case_type: Alert case type to simulate (e.g. "NO_ACTIVITY", "HAR_ALERT"). Default: NO_ACTIVITY.
            channel: Notification channel — "email" or "telegram". Default: email.

        Returns:
            Confirmation of send attempt.
        """
        return _tool_result(_post("/api/alerts/test", case_type=case_type, channel=channel))

    @tool
    def list_persons(limit: int = 20) -> dict:
        """
        List all registered persons in the HAR identity registry.

        Args:
            limit: Maximum number of persons to return (default 20, max 100).

        Returns:
            Persons with display names, appearance counts, and dominant actions.
        """
        lim = max(1, min(limit, 100))
        return _tool_result(_get("/api/har/v2/persons", limit=lim))

    @tool
    def get_person_report(name_or_id: str) -> dict:
        """
        Get the full activity report for a person by display name or global_person_id.

        Args:
            name_or_id: Display name (partial match OK) or exact global_person_id UUID.

        Returns:
            Person metrics, action breakdown, and session history.
        """
        persons = _get("/api/har/v2/persons", limit=100).get("persons", [])
        target = next(
            (
                p for p in persons
                if name_or_id.lower() in (p.get("display_name") or "").lower()
                or p.get("global_person_id") == name_or_id
            ),
            None,
        )
        if not target:
            return _tool_result({"error": f"Person not found: {name_or_id}"})
        pid = target["global_person_id"]
        return _tool_result(_get(f"/api/har/v2/persons/{pid}/report", snapshot_limit=10, event_limit=30))

    return [
        acknowledge_event,
        resolve_event,
        dismiss_event,
        toggle_alert_rule,
        rename_person,
        trigger_har_probe,
        send_test_alert,
        list_persons,
        get_person_report,
    ]

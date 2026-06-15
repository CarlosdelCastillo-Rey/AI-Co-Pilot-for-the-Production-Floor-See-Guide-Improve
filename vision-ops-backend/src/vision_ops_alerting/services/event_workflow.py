from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Event, IndustrialReasonCode

RESOLUTION_STATUSES = ("OPEN", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE")
TERMINAL_STATUSES = ("RESOLVED", "FALSE_POSITIVE")
INCIDENT_SEVERITIES = frozenset({"critical", "warning"})


def is_incident_event(event: Event) -> bool:
    return (event.severity or "info") in INCIDENT_SEVERITIES


def _require_incident(event: Event) -> None:
    if not is_incident_event(event):
        raise HTTPException(
            status_code=400,
            detail="Info and normal events are activity logs, not incidents",
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def acknowledge_event(db: Session, event: Event, operator_id: str) -> Event:
    _require_incident(event)
    if event.resolution_status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Event is already closed")
    event.resolution_status = "ACKNOWLEDGED"
    event.acknowledged_at = _utcnow()
    event.acknowledged_by = operator_id
    db.flush()
    return event


def resolve_event(
    db: Session,
    event: Event,
    *,
    operator_id: str,
    status: str,
    reason_code: str | None,
    downtime_seconds: int,
    scrap_units: int,
    notes: str | None,
) -> Event:
    _require_incident(event)
    if status not in TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail="status must be RESOLVED or FALSE_POSITIVE")
    if event.resolution_status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Event is already closed")

    code = reason_code or ("FALSE_POS" if status == "FALSE_POSITIVE" else None)
    if code and not db.get(IndustrialReasonCode, code):
        raise HTTPException(status_code=400, detail=f"Unknown reason code: {code}")
    if status == "RESOLVED" and not code:
        raise HTTPException(status_code=400, detail="reason_code required for RESOLVED status")

    if event.resolution_status == "OPEN":
        event.acknowledged_at = event.acknowledged_at or _utcnow()
        event.acknowledged_by = event.acknowledged_by or operator_id

    event.resolution_status = status
    event.industrial_reason_code = code
    event.downtime_caused_seconds = max(0, downtime_seconds)
    event.scrap_caused_units = max(0, scrap_units)
    event.closure_notes = notes
    event.resolved_at = _utcnow()
    event.resolved_by = operator_id
    db.flush()
    return event


def dismiss_from_panel(db: Session, event: Event) -> Event:
    if event.resolution_status not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="Only resolved or false-positive events can be removed from the panel",
        )
    if event.hidden_from_panel:
        raise HTTPException(status_code=409, detail="Already removed from panel")
    event.hidden_from_panel = True
    db.flush()
    return event


def build_timeline_stats(db: Session, event_date: str | None = None) -> dict:
    target = event_date or date.today().isoformat()
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59+00:00")

    events = (
        db.query(Event)
        .filter(Event.occurred_at >= start, Event.occurred_at <= end)
        .all()
    )

    incidents = [e for e in events if is_incident_event(e)]

    open_count = sum(
        1 for e in incidents if (e.resolution_status or "OPEN") == "OPEN"
    )
    ack_count = sum(1 for e in incidents if e.resolution_status == "ACKNOWLEDGED")
    resolved_count = sum(1 for e in incidents if e.resolution_status == "RESOLVED")
    false_pos_count = sum(1 for e in incidents if e.resolution_status == "FALSE_POSITIVE")
    open_critical = sum(
        1
        for e in incidents
        if (e.resolution_status or "OPEN") == "OPEN" and e.severity == "critical"
    )
    open_warning = sum(
        1
        for e in incidents
        if (e.resolution_status or "OPEN") == "OPEN" and e.severity == "warning"
    )

    ack_deltas: list[float] = []
    for e in incidents:
        if e.acknowledged_at:
            ack_deltas.append((e.acknowledged_at - e.occurred_at).total_seconds())

    avg_ack_sec = round(sum(ack_deltas) / len(ack_deltas), 1) if ack_deltas else None

    reason_counts: dict[str, int] = {}
    for e in incidents:
        if e.industrial_reason_code:
            reason_counts[e.industrial_reason_code] = reason_counts.get(e.industrial_reason_code, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "date": target,
        "openCount": open_count,
        "acknowledgedCount": ack_count,
        "resolvedCount": resolved_count,
        "falsePositiveCount": false_pos_count,
        "openCriticalCount": open_critical,
        "openWarningCount": open_warning,
        "totalIncidents": len(incidents),
        "infoActivityCount": sum(1 for e in events if not is_incident_event(e)),
        "allClear": open_count == 0,
        "avgAckSeconds": avg_ack_sec,
        "topReasonCodes": [{"code": code, "count": count} for code, count in top_reasons],
    }


def reason_codes_list(db: Session) -> list[dict]:
    rows = db.query(IndustrialReasonCode).order_by(IndustrialReasonCode.sort_order).all()
    return [{"code": r.code, "label": r.label, "category": r.category} for r in rows]

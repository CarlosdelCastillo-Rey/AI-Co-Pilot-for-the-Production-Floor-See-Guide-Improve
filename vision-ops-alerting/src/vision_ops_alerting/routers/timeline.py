from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from vision_ops_alerting.auth_deps import get_current_user
from vision_ops_alerting.db.models import Event, User
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.schemas import ResolveEventBody
from vision_ops_alerting.services.event_workflow import (
    acknowledge_event,
    build_timeline_stats,
    dismiss_from_panel,
    reason_codes_list,
    resolve_event,
)
from vision_ops_alerting.services.events import event_to_timeline_dict
from vision_ops_alerting.services.shift_ai_summary import build_shift_ai_summary
from vision_ops_alerting.services.timeline_summary import build_shift_summary

router = APIRouter(prefix="/api/timeline", tags=["timeline"])


@router.get("")
def list_timeline(
    db: Session = Depends(get_db),
    severity: str | None = Query(None),
    camera_id: str | None = Query(None, alias="cameraId"),
    resolution_status: str | None = Query(None, alias="resolutionStatus"),
    from_date: str | None = Query(None, alias="from"),
    to_date: str | None = Query(None, alias="to"),
    panel_only: bool = Query(True, alias="panelOnly"),
    case_type: str | None = Query(None, alias="caseType"),
    har_only: bool = Query(False, alias="harOnly"),
    limit: int = Query(50, le=200),
):
    q = db.query(Event).order_by(Event.occurred_at.desc())

    if panel_only:
        q = q.filter(Event.hidden_from_panel.is_(False))

    if severity:
        q = q.filter(Event.severity == severity.lower())
    if case_type:
        q = q.filter(Event.case_type == case_type)
    if har_only:
        q = q.filter(Event.case_type == "har_action_deviation")
    if camera_id:
        q = q.filter(Event.camera_id == camera_id)
    if resolution_status:
        q = q.filter(Event.resolution_status == resolution_status.upper())
    if from_date:
        q = q.filter(Event.occurred_at >= datetime.fromisoformat(from_date))
    if to_date:
        q = q.filter(Event.occurred_at <= datetime.fromisoformat(to_date))

    events = q.limit(limit).all()
    return [event_to_timeline_dict(e) for e in events]


@router.get("/summary")
def timeline_summary(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
):
    summary = build_shift_summary(db, event_date)
    stats = build_timeline_stats(db, event_date)
    return {**summary, **stats}


@router.get("/ai-summary")
def timeline_ai_summary(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
):
    return build_shift_ai_summary(db, event_date)


@router.get("/stats")
def timeline_stats(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
):
    return build_timeline_stats(db, event_date)


@router.get("/reason-codes")
def list_reason_codes(db: Session = Depends(get_db)):
    return reason_codes_list(db)


@router.patch("/{event_id}/acknowledge")
def acknowledge_timeline_event(
    event_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    acknowledge_event(db, event, user.name)
    db.commit()
    return event_to_timeline_dict(event)


@router.patch("/{event_id}/resolve")
def resolve_timeline_event(
    event_id: str,
    body: ResolveEventBody,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    resolve_event(
        db,
        event,
        operator_id=user.name,
        status=body.status,
        reason_code=body.reasonCode,
        downtime_seconds=body.downtimeSeconds,
        scrap_units=body.scrapUnits,
        notes=body.notes,
    )
    db.commit()
    return event_to_timeline_dict(event)


@router.patch("/{event_id}/dismiss")
def dismiss_timeline_event(
    event_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    dismiss_from_panel(db, event)
    db.commit()
    return event_to_timeline_dict(event)


@router.get("/{event_id}")
def get_timeline_event(event_id: str, db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event_to_timeline_dict(event)

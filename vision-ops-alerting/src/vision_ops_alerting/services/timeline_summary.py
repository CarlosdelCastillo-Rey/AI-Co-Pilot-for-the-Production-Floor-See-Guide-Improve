from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AnalyticsDaily, Camera, Event, HealthMetricSample


def _day_bounds(target: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59+00:00")
    return start, end


def _events_for_day(db: Session, target: str) -> list[Event]:
    start, end = _day_bounds(target)
    return (
        db.query(Event)
        .filter(Event.occurred_at >= start, Event.occurred_at <= end)
        .order_by(Event.occurred_at.desc())
        .all()
    )


def _uptime_from_health(db: Session, start: datetime, end: datetime) -> float | None:
    rows = (
        db.query(HealthMetricSample.value_pct)
        .filter(
            HealthMetricSample.service.in_(("vision_backend", "vision_models")),
            HealthMetricSample.recorded_at >= start,
            HealthMetricSample.recorded_at <= end,
        )
        .all()
    )
    if not rows:
        return None
    return round(sum(r[0] for r in rows) / len(rows), 1)


def _uptime_from_cameras(db: Session) -> float | None:
    total = db.query(func.count(Camera.id)).filter(Camera.enabled.is_(True)).scalar() or 0
    if total == 0:
        return None
    live = (
        db.query(func.count(Camera.id))
        .filter(Camera.enabled.is_(True), Camera.status == "live")
        .scalar()
        or 0
    )
    return round(live / total * 100, 1)


def _uptime_from_events(events: list[Event], config) -> float:
    critical = sum(1 for e in events if e.severity == "critical")
    warning = sum(1 for e in events if e.severity == "warning")
    penalty = critical * config.uptime_critical_penalty + warning * config.uptime_warning_penalty
    return round(
        max(config.uptime_floor_pct, min(config.uptime_ceiling_pct, config.uptime_ceiling_pct - penalty)),
        1,
    )


def compute_uptime_pct(db: Session, target: str, events: list[Event]) -> float:
    from vision_ops_alerting.services.plant_settings import get_plant_config

    config = get_plant_config(db)
    start, end = _day_bounds(target)

    daily = (
        db.query(AnalyticsDaily)
        .filter(
            AnalyticsDaily.event_date == target,
            AnalyticsDaily.camera_id.is_(None),
        )
        .order_by(AnalyticsDaily.created_at.desc())
        .first()
    )
    if daily and daily.uptime_pct:
        return round(float(daily.uptime_pct), 1)

    health = _uptime_from_health(db, start, end)
    if health is not None:
        return health

    cameras = _uptime_from_cameras(db)
    if cameras is not None:
        return cameras

    return _uptime_from_events(events, config)


def _incident_delta(db: Session, target: str, critical_today: int) -> str:
    prev_day = (datetime.strptime(target, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()
    prev_start, prev_end = _day_bounds(prev_day)
    prev_critical = (
        db.query(func.count(Event.id))
        .filter(
            Event.severity == "critical",
            Event.occurred_at >= prev_start,
            Event.occurred_at <= prev_end,
        )
        .scalar()
        or 0
    )
    diff = critical_today - prev_critical
    if diff > 0:
        return f"+{diff} vs yesterday"
    if diff < 0:
        return f"{diff} vs yesterday"
    return "same as yesterday"


def build_shift_summary(db: Session, event_date: str | None = None) -> dict:
    target = event_date or date.today().isoformat()
    events = _events_for_day(db, target)

    critical_count = sum(1 for e in events if e.severity == "critical")
    uptime_val = compute_uptime_pct(db, target, events)

    assets: dict[str, int] = {}
    for e in events:
        key = e.camera_id or e.line_id or "Unknown"
        assets[key] = assets.get(key, 0) + 1

    top_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "date": datetime.strptime(target, "%Y-%m-%d").strftime("%A, %b %d, %Y"),
        "incidentCount": critical_count,
        "incidentDelta": _incident_delta(db, target, critical_count),
        "uptime": f"{uptime_val:.1f}%",
        "uptimePct": uptime_val,
        "assets": [{"name": name, "events": count} for name, count in top_assets],
        "totalEvents": len(events),
    }

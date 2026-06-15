from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AnalyticsDaily, Event, IndustrialReasonCode
from vision_ops_alerting.services.plant_settings import get_plant_config
from vision_ops_alerting.services.timeline_summary import compute_uptime_pct

QUALITY_CASE_TYPES = {"user_not_working", "user_left_position", "forklift_in_zone", "unknown"}


def _day_bounds(target: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59+00:00")
    return start, end


def _events_for_day(db: Session, target: str, camera_id: str | None) -> list[Event]:
    start, end = _day_bounds(target)
    q = db.query(Event).filter(Event.occurred_at >= start, Event.occurred_at <= end)
    if camera_id:
        q = q.filter(Event.camera_id == camera_id)
    return q.all()


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_oee(db: Session, target: str, shift: str, camera_id: str | None = None) -> dict:
    config = get_plant_config(db)
    daily = (
        db.query(AnalyticsDaily)
        .filter(
            AnalyticsDaily.event_date == target,
            AnalyticsDaily.shift == shift,
            AnalyticsDaily.camera_id == camera_id if camera_id else AnalyticsDaily.camera_id.is_(None),
        )
        .first()
    )
    if daily and daily.oee_pct is not None:
        return {
            "date": target,
            "shift": shift,
            "cameraId": camera_id,
            "availability": round(daily.availability_pct or 0, 1),
            "performance": round(daily.performance_pct or 0, 1),
            "quality": round(daily.quality_pct or 0, 1),
            "oee": round(daily.oee_pct, 1),
        }

    events = _events_for_day(db, target, camera_id)
    uptime = compute_uptime_pct(db, target, events)
    downtime_sec = sum(
        e.downtime_caused_seconds or 0 for e in events if e.resolution_status == "RESOLVED"
    )
    availability = _clamp(uptime - (downtime_sec / 60 / config.shift_hours), 0.0, 100.0)

    critical_warning = sum(1 for e in events if e.severity in ("critical", "warning"))
    target_events = max(1, len(events) + 10)
    performance = _clamp(
        100.0 - (critical_warning / target_events * 100),
        config.performance_floor_pct,
        config.performance_ceiling_pct,
    )

    defect_events = sum(
        1 for e in events if e.case_type in QUALITY_CASE_TYPES and e.severity in ("critical", "warning")
    )
    quality = _clamp(
        100.0 - (defect_events / max(1, len(events)) * 100),
        config.quality_floor_pct,
        config.quality_ceiling_pct,
    )

    oee = round((availability / 100) * (performance / 100) * (quality / 100) * 100, 1)

    return {
        "date": target,
        "shift": shift,
        "cameraId": camera_id,
        "availability": round(availability, 1),
        "performance": round(performance, 1),
        "quality": round(quality, 1),
        "oee": oee,
    }


def compute_flow_efficiency(oee_row: dict) -> float:
    return round((oee_row["performance"] + oee_row["quality"]) / 2, 1)


def compute_flow_history(db: Session, camera_id: str | None, days: int = 7) -> list[dict]:
    history: list[dict] = []
    today = date.today()
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        row = compute_oee(db, d, "morning", camera_id)
        flow = compute_flow_efficiency(row)
        history.append({"date": d, "value": flow})
    return history


def compute_coq(db: Session, target: str, shift: str, camera_id: str | None = None) -> dict:
    config = get_plant_config(db)
    daily = (
        db.query(AnalyticsDaily)
        .filter(
            AnalyticsDaily.event_date == target,
            AnalyticsDaily.shift == shift,
            AnalyticsDaily.camera_id == camera_id if camera_id else AnalyticsDaily.camera_id.is_(None),
        )
        .first()
    )
    if daily and daily.coq_total_usd is not None:
        downtime_min = daily.downtime_minutes or 0
        scrap = daily.scrap_units or 0
        downtime_cost = downtime_min * config.line_cost_per_minute
        scrap_cost = scrap * config.material_cost_per_unit
        return {
            "date": target,
            "shift": shift,
            "downtimeMinutes": round(downtime_min, 1),
            "scrapUnits": scrap,
            "downtimeCostUsd": round(downtime_cost, 2),
            "scrapCostUsd": round(scrap_cost, 2),
            "totalCostUsd": round(daily.coq_total_usd, 2),
            "lineCostPerMinute": config.line_cost_per_minute,
            "materialCostPerUnit": config.material_cost_per_unit,
        }

    events = _events_for_day(db, target, camera_id)
    downtime_sec = sum(
        e.downtime_caused_seconds or 0 for e in events if e.resolution_status == "RESOLVED"
    )
    scrap_units = sum(
        e.scrap_caused_units or 0 for e in events if e.resolution_status == "RESOLVED"
    )

    downtime_min = downtime_sec / 60
    downtime_cost = downtime_min * config.line_cost_per_minute
    scrap_cost = scrap_units * config.material_cost_per_unit
    total = downtime_cost + scrap_cost

    return {
        "date": target,
        "shift": shift,
        "downtimeMinutes": round(downtime_min, 1),
        "scrapUnits": scrap_units,
        "downtimeCostUsd": round(downtime_cost, 2),
        "scrapCostUsd": round(scrap_cost, 2),
        "totalCostUsd": round(total, 2),
        "lineCostPerMinute": config.line_cost_per_minute,
        "materialCostPerUnit": config.material_cost_per_unit,
    }


def compute_pareto(db: Session, target: str, shift: str, camera_id: str | None = None) -> dict:
    target_date = target or date.today().isoformat()
    events = _events_for_day(db, target_date, camera_id)
    closed = [e for e in events if e.industrial_reason_code]

    counts: dict[str, int] = {}
    for e in closed:
        counts[e.industrial_reason_code] = counts.get(e.industrial_reason_code, 0) + 1

    labels = {r.code: r.label for r in db.query(IndustrialReasonCode).all()}
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    total = sum(c for _, c in items) or 1

    cumulative = 0.0
    bars = []
    for code, count in items:
        cumulative += count / total * 100
        bars.append(
            {
                "code": code,
                "label": labels.get(code, code),
                "count": count,
                "pct": round(count / total * 100, 1),
                "cumulativePct": round(cumulative, 1),
            }
        )

    return {"date": target_date, "shift": shift, "items": bars, "totalTagged": sum(counts.values())}


def build_recommendation(db: Session, target: str, camera_id: str | None) -> str:
    events = _events_for_day(db, target, camera_id)
    open_critical = [
        e for e in events if (e.resolution_status or "OPEN") == "OPEN" and e.severity == "critical"
    ]
    if open_critical:
        top = open_critical[0]
        cam = top.camera_id or "the floor"
        return f"Priority: triage {len(open_critical)} open critical incident(s). Start with '{top.title}' on {cam}."

    pareto = compute_pareto(db, target, "morning", camera_id)
    if pareto["items"]:
        top = pareto["items"][0]
        return f"Top root cause today is '{top['label']}' ({top['pct']}% of closures). Review SOPs for {top['code']}."

    by_camera: dict[str, int] = {}
    for e in events:
        if e.severity in ("critical", "warning"):
            key = e.camera_id or e.line_id or "Unknown"
            by_camera[key] = by_camera.get(key, 0) + 1
    if by_camera:
        worst = max(by_camera.items(), key=lambda x: x[1])
        return f"Highest alert volume on {worst[0]} ({worst[1]} events). Consider adjusting thresholds or staffing."

    return "No active anomalies. Maintain current run rates and monitor shift handoff."

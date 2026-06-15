from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AnalyticsDaily, Camera, Event
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.industrial_analytics import (
    build_recommendation,
    compute_coq,
    compute_flow_efficiency,
    compute_flow_history,
    compute_oee,
    compute_pareto,
)
from vision_ops_alerting.services.plant_settings import get_plant_config
from vision_ops_alerting.services.timeline_summary import compute_uptime_pct

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _today() -> str:
    return date.today().isoformat()


def _build_har_action_heatmap_grid(plant: dict) -> dict:
    """Map HAR camera non-assembly rates to a 10×10 dwell grid (live action data)."""
    w, h = 10, 10
    base = 0.08
    cells = [[base for _ in range(w)] for _ in range(h)]
    cameras = plant.get("byCamera") or []
    zone_count = max(1, len(cameras))
    for zi, cam in enumerate(cameras):
        intensity = min(1.0, 0.12 + (cam.get("nonAssemblyRatePct", 0) / 100.0) * 0.88)
        x_start = int((zi / zone_count) * w)
        x_end = int(((zi + 1) / zone_count) * w)
        for y in range(h):
            for x in range(x_start, max(x_start + 1, x_end)):
                cells[y][x] = max(cells[y][x], intensity)
    hotspots = [
        {"x": int((i + 0.5) / zone_count * w), "y": h // 2, "label": c.get("cameraId")}
        for i, c in enumerate(cameras)
        if (c.get("nonAssemblyRatePct") or 0) >= 40
    ]
    return {"width": w, "height": h, "cells": cells, "hotspots": hotspots}


def _heatmap_source_label(source: str) -> str:
    labels = {
        "stored": "demo_seed",
        "events_fallback": "event_density",
        "har_actions": "har_actions",
    }
    return labels.get(source, source)


@router.get("/summary")
def analytics_summary(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    daily = (
        db.query(AnalyticsDaily)
        .filter(
            AnalyticsDaily.event_date == target,
            AnalyticsDaily.shift == shift,
            AnalyticsDaily.camera_id == camera_id if camera_id else AnalyticsDaily.camera_id.is_(None),
        )
        .first()
    )

    start_filter = f"{target}T00:00:00"
    day_events = db.query(Event).filter(Event.occurred_at >= start_filter)
    if camera_id:
        day_events = day_events.filter(Event.camera_id == camera_id)
    day_events_list = day_events.all()

    incident_count = sum(1 for e in day_events_list if e.severity == "critical")
    uptime_val = daily.uptime_pct if daily else compute_uptime_pct(db, target, day_events_list)
    oee_row = compute_oee(db, target, shift, camera_id)
    flow_val = compute_flow_efficiency(oee_row)

    yesterday = (datetime.strptime(target, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()
    prev_oee = compute_oee(db, yesterday, shift, camera_id)
    prev_flow = compute_flow_efficiency(prev_oee)
    flow_delta = round(flow_val - prev_flow, 1)
    flow_trend = f"{flow_delta:+.1f}% vs yesterday" if flow_delta != 0 else "same as yesterday"

    return {
        "date": target,
        "shift": shift,
        "cameraId": camera_id,
        "flowEfficiency": f"{flow_val:.1f}%",
        "flowEfficiencyTrend": flow_trend,
        "flowEfficiencyPct": flow_val,
        "incidentCount": daily.incident_count if daily else incident_count,
        "uptime": f"{uptime_val:.1f}%",
        "oee": f"{oee_row['oee']:.1f}%",
        "openCriticalCount": sum(
            1
            for e in day_events_list
            if (e.resolution_status or "OPEN") == "OPEN" and e.severity == "critical"
        ),
    }


@router.get("/heatmap")
def analytics_heatmap(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str = Query("cam-har-01", alias="cameraId"),
):
    """Live HAR action density only — no demo seed or synthetic fallback grids."""
    from vision_ops_alerting.services.har_activity_store import analytics_plant_actions

    target = event_date or _today()
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    cam_events = (
        db.query(Event)
        .filter(Event.occurred_at >= start, Event.camera_id == camera_id)
        .all()
    )
    event_anomalies = sum(1 for e in cam_events if e.severity in ("critical", "warning"))
    info_count = sum(1 for e in cam_events if e.severity == "info")
    sensors = (
        db.query(func.count(Camera.id))
        .filter(Camera.enabled.is_(True), Camera.inference_model.isnot(None))
        .scalar()
        or 0
    )

    scope = camera_id if camera_id.startswith("cam-har") else None
    plant = analytics_plant_actions(db, target_date=target, camera_id=scope)
    if plant.get("hasData"):
        tags = plant.get("severityTags") or {}
        return {
            "cameraId": camera_id,
            "date": target,
            "shift": shift,
            "grid": _build_har_action_heatmap_grid(plant),
            "anomalyCount": sum(tags.values()) or event_anomalies,
            "sensorsActive": sensors,
            "source": "har_actions",
            "sourceLabel": _heatmap_source_label("har_actions"),
            "severityTags": tags,
            "hasData": True,
        }

    return {
        "cameraId": camera_id,
        "date": target,
        "shift": shift,
        "grid": {"width": 10, "height": 10, "cells": [], "hotspots": []},
        "anomalyCount": event_anomalies,
        "sensorsActive": sensors,
        "source": "none",
        "sourceLabel": "no_data",
        "severityTags": {
            "critical": sum(1 for e in cam_events if e.severity == "critical"),
            "warning": sum(1 for e in cam_events if e.severity == "warning"),
            "info": info_count,
        },
        "hasData": False,
    }


@router.get("/insights")
def analytics_insights(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    config = get_plant_config(db)
    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")

    q = db.query(Event).filter(Event.occurred_at >= start).order_by(Event.occurred_at.desc())
    if camera_id:
        q = q.filter(Event.camera_id == camera_id)
    events = q.limit(50).all()

    default_clip = config.default_clip_duration_sec
    threshold = config.downtime_critical_threshold_pct

    downtime_by_camera: dict[str, int] = {}
    for e in events:
        if e.severity in ("critical", "warning"):
            key = e.camera_id or e.line_id or "Unknown"
            downtime_by_camera[key] = downtime_by_camera.get(key, 0) + (
                e.clip_duration_sec or default_clip
            )

    stations = sorted(downtime_by_camera.items(), key=lambda x: x[1], reverse=True)[:5]
    max_val = max((s[1] for s in stations), default=1)

    bottlenecks = [
        {
            "id": e.id,
            "title": e.title,
            "severity": e.severity,
            "description": f"Detected on {e.camera_id or 'unknown'} • {e.occurred_at.strftime('%H:%M')} • {e.resolution_status or 'OPEN'}",
            "critical": e.severity == "critical",
        }
        for e in events if e.severity in ("critical", "warning")
    ][:5]

    oee_row = compute_oee(db, target, shift, camera_id)
    flow = compute_flow_efficiency(oee_row)
    flow_history = compute_flow_history(db, camera_id)

    har_actions = None
    if not camera_id or str(camera_id).startswith("cam-har"):
        from vision_ops_alerting.services.har_activity_store import analytics_plant_actions

        scope = camera_id if camera_id and str(camera_id).startswith("cam-har") else None
        har_actions = analytics_plant_actions(db, target_date=target, camera_id=scope)

    recommendation = build_recommendation(db, target, camera_id)
    if har_actions and har_actions.get("hasData"):
        tags = har_actions.get("severityTags") or {}
        prod = har_actions.get("productivityScore", 0)
        cost = har_actions.get("actionCostUsd", 0)
        recommendation = (
            f"HAR productivity {prod}% ({har_actions.get('assembleSharePct', 0)}% primary action). "
            f"Severity today: {tags.get('critical', 0)} critical, {tags.get('warning', 0)} warning, "
            f"{tags.get('info', 0)} info. Estimated action deviation cost ${cost:,.0f}. "
            f"{recommendation}"
        )

    return {
        "date": target,
        "shift": shift,
        "cameraId": camera_id,
        "flowEfficiency": f"{flow:.1f}%",
        "flowHistory": flow_history,
        "downtimeByStation": [
            {
                "name": name,
                "minutes": round(secs / 60, 1),
                "widthPct": f"{min(100, int(secs / max_val * 100))}%",
                "critical": secs >= max_val * threshold,
            }
            for name, secs in stations
        ],
        "bottlenecks": bottlenecks,
        "recommendation": recommendation,
        "severityTags": har_actions.get("severityTags") if har_actions else {
            "critical": sum(1 for e in events if e.severity == "critical"),
            "warning": sum(1 for e in events if e.severity == "warning"),
            "info": sum(1 for e in events if e.severity == "info"),
        },
        "harActions": har_actions,
    }


@router.get("/oee")
def analytics_oee(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    return compute_oee(db, target, shift, camera_id)


@router.get("/coq")
def analytics_coq(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    coq = compute_coq(db, target, shift, camera_id)
    if not camera_id or str(camera_id).startswith("cam-har"):
        from vision_ops_alerting.services.har_activity_store import analytics_plant_actions

        scope = camera_id if camera_id and str(camera_id).startswith("cam-har") else None
        har = analytics_plant_actions(db, target_date=target, camera_id=scope)
        if har.get("hasData"):
            action_cost = float(har.get("actionCostUsd") or 0)
            coq = {
                **coq,
                "actionDeviationCostUsd": action_cost,
                "actionDowntimeMinutes": har.get("actionDowntimeMinutes", 0),
                "totalCostUsd": round(float(coq.get("totalCostUsd", 0)) + action_cost, 2),
            }
    return coq


@router.get("/pareto")
def analytics_pareto(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    return compute_pareto(db, target, shift, camera_id)

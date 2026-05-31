from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AnalyticsDaily, AnalyticsHeatmap, Camera, Event
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
    camera_id: str = Query("cam-01", alias="cameraId"),
):
    target = event_date or _today()
    row = (
        db.query(AnalyticsHeatmap)
        .filter(
            AnalyticsHeatmap.event_date == target,
            AnalyticsHeatmap.shift == shift,
            AnalyticsHeatmap.camera_id == camera_id,
        )
        .first()
    )

    start = datetime.fromisoformat(f"{target}T00:00:00+00:00")
    cam_events = (
        db.query(Event)
        .filter(Event.occurred_at >= start, Event.camera_id == camera_id)
        .all()
    )
    event_anomalies = sum(1 for e in cam_events if e.severity in ("critical", "warning"))
    sensors = (
        db.query(func.count(Camera.id))
        .filter(Camera.enabled.is_(True), Camera.inference_model.isnot(None))
        .scalar()
        or 0
    )

    if row:
        grid = json.loads(row.grid_json)
        return {
            "cameraId": row.camera_id,
            "date": row.event_date,
            "shift": row.shift,
            "grid": grid,
            "anomalyCount": row.anomaly_count or event_anomalies,
            "sensorsActive": row.sensors_active or sensors,
            "source": "stored",
        }

    intensity = min(1.0, 0.1 + event_anomalies * 0.08)
    grid = {
        "width": 10,
        "height": 10,
        "cells": [[intensity for _ in range(10)] for _ in range(10)],
        "hotspots": [],
    }
    return {
        "cameraId": camera_id,
        "date": target,
        "shift": shift,
        "grid": grid,
        "anomalyCount": event_anomalies,
        "sensorsActive": sensors,
        "source": "events_fallback",
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
        "recommendation": build_recommendation(db, target, camera_id),
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
    return compute_coq(db, target, shift, camera_id)


@router.get("/pareto")
def analytics_pareto(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    shift: str = Query("morning"),
    camera_id: str | None = Query(None, alias="cameraId"),
):
    target = event_date or _today()
    return compute_pareto(db, target, shift, camera_id)

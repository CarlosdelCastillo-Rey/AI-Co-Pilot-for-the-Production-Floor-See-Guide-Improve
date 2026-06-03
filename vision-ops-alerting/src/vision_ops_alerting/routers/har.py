"""HAR inference history and integral activity log API (SQLite)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.har_activity_store import (
    activity_summary,
    analytics_daily,
    analytics_plant_actions,
    analytics_realtime,
    list_activity_logs,
    list_sessions,
    record_activity_batch,
)
from vision_ops_alerting.services.har_alert_dispatcher import dispatch_after_ingest
from vision_ops_alerting.services.har_inference_store import (
    get_har_run,
    latest_results_by_model,
    list_har_results,
    list_har_runs,
    record_har_run,
    run_to_dict,
)

router = APIRouter(prefix="/api/har", tags=["har"])


class HarRunIngestBody(BaseModel):
    run_type: str = Field("batch", description="batch | single")
    source: str = ""
    clip_path: str | None = None
    frame_count: int | None = None
    shared_clip: bool = True
    status: str = "ok"
    probes: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/runs")
def ingest_har_run(body: HarRunIngestBody, db: Session = Depends(get_db)) -> dict:
    """Ingest a HAR probe batch from vision-ops-backend (internal)."""
    run = record_har_run(db, body.model_dump())
    return {"status": "ok", "run": run_to_dict(run, include_results=True)}


@router.get("/runs")
def get_har_runs(
    db: Session = Depends(get_db),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    camera_id: str | None = Query(None, alias="cameraId"),
    model_id: str | None = Query(None, alias="modelId"),
) -> dict:
    runs, total = list_har_runs(
        db, limit=limit, offset=offset, camera_id=camera_id, model_id=model_id
    )
    return {"total": total, "limit": limit, "offset": offset, "runs": runs}


@router.get("/runs/{run_id}")
def get_har_run_detail(run_id: str, db: Session = Depends(get_db)) -> dict:
    data = get_har_run(db, run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="HAR run not found")
    return data


@router.get("/results")
def get_har_results(
    db: Session = Depends(get_db),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    camera_id: str | None = Query(None, alias="cameraId"),
    model_id: str | None = Query(None, alias="modelId"),
) -> dict:
    results, total = list_har_results(
        db, limit=limit, offset=offset, camera_id=camera_id, model_id=model_id
    )
    return {"total": total, "limit": limit, "offset": offset, "results": results}


@router.get("/results/latest")
def get_latest_har_results(db: Session = Depends(get_db)) -> dict:
    """Latest successful prediction per model (watch data / comparison strip)."""
    return {"results": latest_results_by_model(db)}


class HarActivityIngestBody(BaseModel):
    entries: list[dict[str, Any]] = Field(default_factory=list)
    entry: dict[str, Any] | None = None


@router.post("/activity")
def ingest_har_activity(body: HarActivityIngestBody, db: Session = Depends(get_db)) -> dict:
    """Ingest live or probe HAR activity logs from vision-ops-backend."""
    items = list(body.entries)
    if body.entry:
        items.append(body.entry)
    if not items:
        raise HTTPException(status_code=400, detail="entry or entries required")
    recorded = record_activity_batch(db, items)
    event_ids = dispatch_after_ingest(db, recorded)
    return {
        "status": "ok",
        "recorded": len(recorded),
        "skipped": len(items) - len(recorded),
        "promotedEventIds": event_ids,
        "logs": [{"id": r.id, "predictedLabel": r.predicted_label} for r in recorded],
    }


@router.get("/activity")
def get_har_activity(
    db: Session = Depends(get_db),
    camera_id: str | None = Query(None, alias="cameraId"),
    session_id: str | None = Query(None, alias="sessionId"),
    label: str | None = Query(None),
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    limit: int = Query(80, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    from_dt = datetime.fromisoformat(from_ts.replace("Z", "+00:00")) if from_ts else None
    to_dt = datetime.fromisoformat(to_ts.replace("Z", "+00:00")) if to_ts else None
    logs, total = list_activity_logs(
        db,
        camera_id=camera_id,
        session_id=session_id,
        label=label,
        from_dt=from_dt,
        to_dt=to_dt,
        limit=limit,
        offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "logs": logs}


@router.get("/activity/summary")
def get_har_activity_summary(
    db: Session = Depends(get_db),
    camera_id: str = Query(..., alias="cameraId"),
    event_date: str | None = Query(None, alias="date"),
) -> dict:
    return activity_summary(db, camera_id=camera_id, target_date=event_date)


@router.get("/sessions")
def get_har_sessions(
    db: Session = Depends(get_db),
    camera_id: str | None = Query(None, alias="cameraId"),
    limit: int = Query(20, le=100),
) -> dict:
    return {"sessions": list_sessions(db, camera_id=camera_id, limit=limit)}


@router.get("/analytics/daily")
def get_har_analytics_daily(
    db: Session = Depends(get_db),
    camera_id: str = Query(..., alias="cameraId"),
    event_date: str | None = Query(None, alias="date"),
) -> dict:
    return analytics_daily(db, camera_id=camera_id, target_date=event_date)


@router.get("/analytics/realtime")
def get_har_analytics_realtime(
    db: Session = Depends(get_db),
    camera_id: str = Query(..., alias="cameraId"),
    minutes: int = Query(30, ge=5, le=120),
) -> dict:
    return analytics_realtime(db, camera_id=camera_id, minutes=minutes)


@router.get("/analytics/plant")
def get_har_analytics_plant(
    db: Session = Depends(get_db),
    event_date: str | None = Query(None, alias="date"),
    camera_id: str | None = Query(None, alias="cameraId"),
) -> dict:
    """Plant-wide HAR action productivity, severity tags, and action-based CoQ estimate."""
    return analytics_plant_actions(db, target_date=event_date, camera_id=camera_id)

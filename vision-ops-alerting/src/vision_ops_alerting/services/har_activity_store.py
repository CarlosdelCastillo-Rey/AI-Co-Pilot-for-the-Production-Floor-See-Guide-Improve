"""Integral HAR activity logs and watch sessions (SQLite)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import (
    HAR_PRIMARY_ACTION_LABEL,
    HarActivityLog,
    HarWatchSession,
    new_id,
)

PRIMARY_ACTION = settings.har_primary_action_label or HAR_PRIMARY_ACTION_LABEL


def _primary_label(label: str | None) -> bool:
    if not label:
        return False
    return label == PRIMARY_ACTION


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    """SQLite often returns naive UTC; normalize before comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_dt(raw: str | None) -> datetime:
    if raw:
        try:
            return _as_utc_aware(datetime.fromisoformat(raw.replace("Z", "+00:00"))) or _utc_now()
        except ValueError:
            pass
    return _utc_now()


def _parse_hyperparams(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def hyperparam_key(hyperparams: dict[str, Any] | None) -> str:
    h = hyperparams or {}
    return (
        f"infer{h.get('infer_every', '—')}_"
        f"buf{h.get('buffer_frames', '—')}_"
        f"fps{h.get('stream_fps', '—')}_"
        f"hm{int(bool(h.get('show_heatmap', True)))}_"
        f"yolo{int(bool(h.get('show_yolo_boxes', True)))}_"
        f"topk{h.get('top_k', '—')}"
    )


def hyperparam_label(hyperparams: dict[str, Any] | None) -> str:
    h = hyperparams or {}
    if not h:
        return "default"
    return (
        f"every {h.get('infer_every', '?')}f · "
        f"win {h.get('buffer_frames', '?')} · "
        f"fps {h.get('stream_fps', '?')} · "
        f"top-{h.get('top_k', '?')}"
    )


def prune_old_logs(db: Session) -> int:
    days = max(1, settings.har_log_retention_days)
    cutoff = _utc_now() - timedelta(days=days)
    # Compare using ISO strings so naive/aware SQLite rows both match.
    cutoff_iso = cutoff.replace(tzinfo=None).isoformat(sep=" ")
    deleted = (
        db.query(HarActivityLog)
        .filter(HarActivityLog.occurred_at < cutoff_iso)
        .delete(synchronize_session=False)
    )
    if deleted:
        db.commit()
    return deleted


def get_or_create_session(
    db: Session,
    *,
    camera_id: str,
    model_id: str,
    session_id: str | None = None,
    video_name: str | None = None,
    clip_url: str | None = None,
    model_label: str | None = None,
    hyperparams: dict[str, Any] | None = None,
    new_session: bool = False,
) -> HarWatchSession:
    hp_json = json.dumps(hyperparams, ensure_ascii=False) if hyperparams else None
    if session_id and not new_session:
        existing = db.query(HarWatchSession).filter(HarWatchSession.id == session_id).first()
        if existing:
            if video_name and not existing.video_name:
                existing.video_name = video_name
            if clip_url and not existing.clip_url:
                existing.clip_url = clip_url
            if model_label and not existing.model_label:
                existing.model_label = model_label
            if hp_json and not existing.hyperparams_json:
                existing.hyperparams_json = hp_json
            return existing

    if not new_session:
        open_sess = (
            db.query(HarWatchSession)
            .filter(
                HarWatchSession.camera_id == camera_id,
                HarWatchSession.ended_at.is_(None),
            )
            .order_by(HarWatchSession.started_at.desc())
            .first()
        )
        if open_sess:
            if video_name and not open_sess.video_name:
                open_sess.video_name = video_name
            if clip_url and not open_sess.clip_url:
                open_sess.clip_url = clip_url
            if model_label and not open_sess.model_label:
                open_sess.model_label = model_label
            if hp_json and not open_sess.hyperparams_json:
                open_sess.hyperparams_json = hp_json
            return open_sess

    sess = HarWatchSession(
        id=session_id or new_id("har-sess"),
        camera_id=camera_id,
        model_id=model_id,
        video_name=video_name,
        clip_url=clip_url,
        model_label=model_label,
        hyperparams_json=hp_json,
    )
    db.add(sess)
    db.flush()
    return sess


def end_session(db: Session, session_id: str) -> None:
    sess = db.query(HarWatchSession).filter(HarWatchSession.id == session_id).first()
    if sess and sess.ended_at is None:
        sess.ended_at = _utc_now()


def _last_log_for_camera(db: Session, camera_id: str) -> HarActivityLog | None:
    return (
        db.query(HarActivityLog)
        .filter(HarActivityLog.camera_id == camera_id)
        .order_by(HarActivityLog.occurred_at.desc())
        .first()
    )


def _should_record(db: Session, entry: dict[str, Any]) -> bool:
    """Dedup: label change, confidence delta, or heartbeat."""
    camera_id = str(entry.get("camera_id") or "")
    if not camera_id:
        return False
    last = _last_log_for_camera(db, camera_id)
    if last is None:
        return True

    label = entry.get("predicted_label") or (entry.get("prediction") or {}).get("label")
    conf = entry.get("confidence")
    if conf is None and entry.get("prediction"):
        conf = entry["prediction"].get("confidence")
    conf = float(conf) if conf is not None else 0.0

    if (last.predicted_label or "") != (label or ""):
        return True
    if last.confidence is not None and abs(float(last.confidence) - conf) >= settings.har_ingest_confidence_delta:
        return True
    hb = timedelta(seconds=max(30, settings.har_ingest_heartbeat_sec))
    last_at = _as_utc_aware(last.occurred_at)
    if last_at and _utc_now() - last_at >= hb:
        return True
    return False


def _detections_to_actor(detections: list[dict[str, Any]]) -> tuple[str | None, str | None, str | None, int]:
    if not detections:
        return None, None, None, 0
    best = max(detections, key=lambda d: float(d.get("det_conf") or d.get("confidence") or 0))
    track = best.get("track_index")
    name = best.get("person_name")
    track_id = str(track) if track is not None else None
    actor_name = name or (f"Person {track_id}" if track_id else None)
    return "operator", track_id, actor_name, len(detections)


def log_to_dict(row: HarActivityLog) -> dict[str, Any]:
    top_k: list[Any] = []
    if row.top_k_json:
        try:
            top_k = json.loads(row.top_k_json)
        except json.JSONDecodeError:
            pass
    detections: list[Any] = []
    if row.detections_json:
        try:
            detections = json.loads(row.detections_json)
        except json.JSONDecodeError:
            pass
    hyperparams = _parse_hyperparams(row.hyperparams_json)
    preview_url = row.snapshot_url or row.clip_url
    return {
        "id": row.id,
        "occurredAt": row.occurred_at.isoformat() if row.occurred_at else None,
        "cameraId": row.camera_id,
        "modelId": row.model_id,
        "modelLabel": row.model_label,
        "sessionId": row.session_id,
        "source": row.source,
        "frameIndex": row.frame_index,
        "videoOffsetSec": row.video_offset_sec,
        "videoName": row.video_name,
        "clipUrl": row.clip_url,
        "previewUrl": preview_url,
        "hyperparams": hyperparams,
        "hyperparamKey": hyperparam_key(hyperparams),
        "hyperparamLabel": hyperparam_label(hyperparams),
        "predictedLabel": row.predicted_label,
        "classIndex": row.class_index,
        "confidence": row.confidence,
        "topK": top_k,
        "isPrimaryAction": row.is_primary_action,
        "personCount": row.person_count,
        "detections": detections,
        "actorType": row.actor_type,
        "actorTrackId": row.actor_track_id,
        "actorName": row.actor_name,
        "backend": row.backend,
        "device": row.device,
        "inferMs": row.infer_ms,
        "promotedToEventId": row.promoted_to_event_id,
        "snapshotUrl": row.snapshot_url,
    }


def record_activity(
    db: Session,
    entry: dict[str, Any],
    *,
    skip_dedup: bool = False,
) -> HarActivityLog | None:
    """Ingest one activity log row. Returns None if deduped."""
    if not skip_dedup and not _should_record(db, entry):
        return None

    pred = entry.get("prediction") or {}
    label = entry.get("predicted_label") or pred.get("label")
    conf = entry.get("confidence")
    if conf is None:
        conf = pred.get("confidence")
    top_k = entry.get("top_k") or pred.get("top_k") or []

    camera_id = str(entry.get("camera_id") or "")
    model_id = str(entry.get("model_id") or "")
    if not camera_id or not model_id:
        return None

    new_session = bool(entry.get("new_session"))
    hyperparams = entry.get("hyperparams") or _parse_hyperparams(entry.get("hyperparams_json"))
    model_label = entry.get("model_label") or entry.get("modelLabel")
    video_name = entry.get("video_name") or entry.get("video")
    clip_url = entry.get("clip_url") or entry.get("videoUrl")
    sess = get_or_create_session(
        db,
        camera_id=camera_id,
        model_id=model_id,
        session_id=entry.get("session_id"),
        video_name=video_name,
        clip_url=clip_url,
        model_label=model_label,
        hyperparams=hyperparams or None,
        new_session=new_session,
    )

    detections = entry.get("detections") or []
    actor_type, actor_track_id, actor_name, person_count = _detections_to_actor(detections)
    if entry.get("person_count") is not None:
        person_count = int(entry["person_count"])

    row = HarActivityLog(
        id=new_id("har-log"),
        occurred_at=_parse_dt(entry.get("occurred_at")).replace(tzinfo=None),
        camera_id=camera_id,
        model_id=model_id,
        session_id=sess.id,
        source=str(entry.get("source") or "live"),
        frame_index=entry.get("frame_index"),
        video_offset_sec=entry.get("video_offset_sec"),
        predicted_label=label,
        class_index=entry.get("class_index") or pred.get("class_index"),
        confidence=float(conf) if conf is not None else None,
        top_k_json=json.dumps(top_k, ensure_ascii=False),
        is_primary_action=_primary_label(label),
        person_count=person_count,
        detections_json=json.dumps(detections, ensure_ascii=False) if detections else None,
        actor_type=actor_type,
        actor_track_id=actor_track_id,
        actor_name=actor_name,
        backend=entry.get("backend"),
        device=entry.get("device"),
        infer_ms=entry.get("infer_ms"),
        snapshot_url=entry.get("snapshot_url") or entry.get("snapshotUrl"),
        video_name=video_name,
        clip_url=clip_url,
        model_label=model_label,
        hyperparams_json=json.dumps(hyperparams, ensure_ascii=False) if hyperparams else None,
    )
    db.add(row)
    db.flush()
    return row


def record_activity_batch(
    db: Session,
    entries: list[dict[str, Any]],
) -> list[HarActivityLog]:
    prune_old_logs(db)
    recorded: list[HarActivityLog] = []
    for entry in entries:
        row = record_activity(db, entry)
        if row:
            recorded.append(row)
    if recorded:
        db.commit()
        for row in recorded:
            db.refresh(row)
    return recorded


def list_activity_logs(
    db: Session,
    *,
    camera_id: str | None = None,
    session_id: str | None = None,
    label: str | None = None,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    limit: int = 80,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    q = db.query(HarActivityLog).order_by(HarActivityLog.occurred_at.desc())
    if camera_id:
        q = q.filter(HarActivityLog.camera_id == camera_id)
    if session_id:
        q = q.filter(HarActivityLog.session_id == session_id)
    if label:
        q = q.filter(HarActivityLog.predicted_label == label)
    if from_dt:
        q = q.filter(HarActivityLog.occurred_at >= from_dt)
    if to_dt:
        q = q.filter(HarActivityLog.occurred_at <= to_dt)
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return [log_to_dict(r) for r in rows], total


def list_sessions(
    db: Session,
    *,
    camera_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    q = db.query(HarWatchSession).order_by(HarWatchSession.started_at.desc())
    if camera_id:
        q = q.filter(HarWatchSession.camera_id == camera_id)
    rows = q.limit(limit).all()
    return [
        {
            "id": s.id,
            "cameraId": s.camera_id,
            "modelId": s.model_id,
            "videoName": s.video_name,
            "clipUrl": s.clip_url,
            "modelLabel": s.model_label,
            "hyperparams": _parse_hyperparams(s.hyperparams_json),
            "hyperparamLabel": hyperparam_label(_parse_hyperparams(s.hyperparams_json)),
            "startedAt": s.started_at.isoformat() if s.started_at else None,
            "endedAt": s.ended_at.isoformat() if s.ended_at else None,
        }
        for s in rows
    ]


def activity_summary(
    db: Session,
    *,
    camera_id: str,
    target_date: str | None = None,
) -> dict[str, Any]:
    target = target_date or date.today().isoformat()
    start = datetime.fromisoformat(f"{target}T00:00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59.999999")

    rows = (
        db.query(HarActivityLog)
        .filter(
            HarActivityLog.camera_id == camera_id,
            HarActivityLog.occurred_at >= start,
            HarActivityLog.occurred_at <= end,
        )
        .all()
    )
    by_label: dict[str, int] = {}
    conf_sum: dict[str, float] = {}
    for r in rows:
        lbl = r.predicted_label or "unknown"
        by_label[lbl] = by_label.get(lbl, 0) + 1
        if r.confidence is not None:
            conf_sum[lbl] = conf_sum.get(lbl, 0.0) + float(r.confidence)

    total = len(rows) or 1
    assemble = by_label.get(PRIMARY_ACTION, 0)
    return {
        "date": target,
        "cameraId": camera_id,
        "totalInferences": len(rows),
        "byLabel": by_label,
        "avgConfidenceByLabel": {
            k: round(conf_sum[k] / by_label[k], 4) for k in by_label if by_label[k]
        },
        "nonAssemblyRatePct": round((1 - assemble / total) * 100, 1),
        "assembleSharePct": round(assemble / total * 100, 1),
    }


def analytics_daily(
    db: Session,
    *,
    camera_id: str,
    target_date: str | None = None,
) -> dict[str, Any]:
    summary = activity_summary(db, camera_id=camera_id, target_date=target_date)
    target = summary["date"]
    start = datetime.fromisoformat(f"{target}T00:00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59.999999")

    rows = (
        db.query(HarActivityLog)
        .filter(
            HarActivityLog.camera_id == camera_id,
            HarActivityLog.occurred_at >= start,
            HarActivityLog.occurred_at <= end,
        )
        .order_by(HarActivityLog.occurred_at)
        .all()
    )

    hourly: dict[int, int] = {h: 0 for h in range(24)}
    for r in rows:
        at = _as_utc_aware(r.occurred_at)
        if at:
            hourly[at.hour] += 1

    pareto = sorted(
        [{"label": k, "count": v} for k, v in summary["byLabel"].items() if k != PRIMARY_ACTION],
        key=lambda x: x["count"],
        reverse=True,
    )[:8]

    return {
        **summary,
        "hourlyCounts": [{"hour": h, "count": hourly[h]} for h in range(24)],
        "topDeviations": pareto,
        "hasData": len(rows) > 0,
    }


def analytics_realtime(
    db: Session,
    *,
    camera_id: str,
    minutes: int = 30,
) -> dict[str, Any]:
    since = _utc_now() - timedelta(minutes=max(5, minutes))
    since_naive = since.replace(tzinfo=None)
    rows = (
        db.query(HarActivityLog)
        .filter(
            HarActivityLog.camera_id == camera_id,
            HarActivityLog.occurred_at >= since_naive,
        )
        .order_by(HarActivityLog.occurred_at.desc())
        .all()
    )
    latest = log_to_dict(rows[0]) if rows else None
    non_primary = sum(1 for r in rows if not r.is_primary_action)
    return {
        "cameraId": camera_id,
        "windowMinutes": minutes,
        "inferenceCount": len(rows),
        "nonPrimaryCount": non_primary,
        "latest": latest,
        "recent": [log_to_dict(r) for r in rows[:20]],
    }


def logs_for_advisor(
    db: Session,
    *,
    camera_id: str,
    limit: int = 200,
    session_id: str | None = None,
) -> dict[str, Any]:
    def _fetch(sid: str | None) -> list[HarActivityLog]:
        q = db.query(HarActivityLog).filter(HarActivityLog.camera_id == camera_id)
        if sid:
            q = q.filter(HarActivityLog.session_id == sid)
        return q.order_by(HarActivityLog.occurred_at.desc()).limit(limit).all()

    rows = _fetch(session_id)
    # Current live session may not have persisted rows yet — use full camera history.
    if session_id and not rows:
        rows = _fetch(None)

    summary = activity_summary(db, camera_id=camera_id)
    sess = None
    if session_id:
        s = db.query(HarWatchSession).filter(HarWatchSession.id == session_id).first()
        if s:
            sess = {
                "id": s.id,
                "videoName": s.video_name,
                "clipUrl": s.clip_url,
                "startedAt": s.started_at.isoformat() if s.started_at else None,
            }
    elif rows and rows[0].session_id:
        s = db.query(HarWatchSession).filter(HarWatchSession.id == rows[0].session_id).first()
        if s:
            sess = {
                "id": s.id,
                "videoName": s.video_name,
                "clipUrl": s.clip_url,
                "startedAt": s.started_at.isoformat() if s.started_at else None,
            }
    return {
        "cameraId": camera_id,
        "session": sess,
        "summary": summary,
        "logs": [log_to_dict(r) for r in reversed(rows)],
    }


def global_har_summary(db: Session, *, hours: int = 24) -> dict[str, Any]:
    since = _utc_now() - timedelta(hours=hours)
    since_naive = since.replace(tzinfo=None)
    total = (
        db.query(func.count(HarActivityLog.id))
        .filter(HarActivityLog.occurred_at >= since_naive)
        .scalar()
        or 0
    )
    non_primary = (
        db.query(func.count(HarActivityLog.id))
        .filter(HarActivityLog.occurred_at >= since_naive, HarActivityLog.is_primary_action.is_(False))
        .scalar()
        or 0
    )
    cameras = (
        db.query(HarActivityLog.camera_id, func.count(HarActivityLog.id))
        .filter(HarActivityLog.occurred_at >= since_naive)
        .group_by(HarActivityLog.camera_id)
        .all()
    )
    return {
        "hours": hours,
        "totalInferences": total,
        "nonPrimaryCount": non_primary,
        "byCamera": {cid: cnt for cid, cnt in cameras},
    }


def build_all_cameras_har_dashboard(db: Session, *, hours: int = 24) -> dict[str, Any]:
    """Per-camera HAR snapshot for Live page advisor (all cam-har-* feeds)."""
    from vision_ops_alerting.db.models import Camera, Event
    from vision_ops_alerting.services.events import event_to_timeline_dict

    har_cameras = (
        db.query(Camera)
        .filter(Camera.enabled.is_(True), Camera.id.like("cam-har-%"))
        .order_by(Camera.sort_order, Camera.id)
        .all()
    )

    camera_rows: list[dict[str, Any]] = []
    for cam in har_cameras:
        logs, _ = list_activity_logs(db, camera_id=cam.id, limit=3)
        latest = logs[0] if logs else None
        summary = activity_summary(db, camera_id=cam.id)
        pct = None
        if latest and latest.get("confidence") is not None:
            pct = int(round(float(latest["confidence"]) * 100))
        camera_rows.append(
            {
                "cameraId": cam.id,
                "name": cam.name,
                "location": cam.location,
                "zone": cam.zone,
                "status": cam.status,
                "modelId": cam.inference_model,
                "latestAction": latest.get("predictedLabel") if latest else None,
                "latestConfidencePct": pct,
                "latestAt": latest.get("occurredAt") if latest else None,
                "actorName": latest.get("actorName") if latest else None,
                "inferencesToday": summary.get("totalInferences", 0),
                "nonAssemblyRatePct": summary.get("nonAssemblyRatePct", 0),
                "assembleSharePct": summary.get("assembleSharePct", 0),
                "actionsToday": summary.get("byLabel", {}),
                "recentLogs": logs,
            }
        )

    open_har_events = (
        db.query(Event)
        .filter(
            Event.case_type == "har_action_deviation",
            (Event.resolution_status == "OPEN") | (Event.resolution_status.is_(None)),
        )
        .order_by(Event.occurred_at.desc())
        .limit(25)
        .all()
    )
    all_open = (
        db.query(Event)
        .filter(
            (Event.resolution_status == "OPEN") | (Event.resolution_status.is_(None)),
            Event.hidden_from_panel.is_(False),
        )
        .order_by(Event.occurred_at.desc())
        .limit(25)
        .all()
    )

    plant = global_har_summary(db, hours=hours)
    return {
        "hours": hours,
        "cameraCount": len(camera_rows),
        "cameras": camera_rows,
        "plantHarSummary": plant,
        "openHarIncidents": [event_to_timeline_dict(e) for e in open_har_events],
        "openIncidents": [event_to_timeline_dict(e) for e in all_open],
    }


def slim_har_dashboard_for_advisor(dash: dict[str, Any]) -> dict[str, Any]:
    """Compact HAR dashboard for LLM prompts (no full log payloads)."""
    cameras: list[dict[str, Any]] = []
    for cam in dash.get("cameras") or []:
        cameras.append(
            {
                "cameraId": cam.get("cameraId"),
                "name": cam.get("name"),
                "status": cam.get("status"),
                "latestAction": cam.get("latestAction"),
                "latestConfidencePct": cam.get("latestConfidencePct"),
                "latestAt": cam.get("latestAt"),
                "actorName": cam.get("actorName"),
                "inferencesToday": cam.get("inferencesToday"),
                "nonAssemblyRatePct": cam.get("nonAssemblyRatePct"),
                "actionsToday": cam.get("actionsToday"),
            }
        )
    return {
        "hours": dash.get("hours"),
        "cameraCount": dash.get("cameraCount", len(cameras)),
        "cameras": cameras,
        "plantHarSummary": dash.get("plantHarSummary"),
        "openHarIncidents": (dash.get("openHarIncidents") or [])[:10],
        "openIncidents": (dash.get("openIncidents") or [])[:10],
    }


# Estimated rework minutes attributed to each non-primary HAR inference (CoQ proxy).
HAR_DEVIATION_MINUTES = 0.75
HAR_CRITICAL_CONFIDENCE = 0.12


def analytics_plant_actions(
    db: Session,
    *,
    target_date: str | None = None,
    camera_id: str | None = None,
) -> dict[str, Any]:
    """Plant-wide HAR action analytics for the Analytics 360 dashboard."""
    from vision_ops_alerting.db.models import Camera, Event
    from vision_ops_alerting.services.plant_settings import get_plant_config

    target = target_date or date.today().isoformat()
    start = datetime.fromisoformat(f"{target}T00:00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59.999999")

    cam_q = db.query(Camera).filter(Camera.enabled.is_(True), Camera.id.like("cam-har-%"))
    if camera_id and camera_id.startswith("cam-har"):
        cam_q = cam_q.filter(Camera.id == camera_id)
    har_cameras = cam_q.order_by(Camera.sort_order, Camera.id).all()
    cam_ids = [c.id for c in har_cameras]

    log_q = db.query(HarActivityLog).filter(
        HarActivityLog.occurred_at >= start,
        HarActivityLog.occurred_at <= end,
    )
    if cam_ids:
        log_q = log_q.filter(HarActivityLog.camera_id.in_(cam_ids))
    elif camera_id:
        log_q = log_q.filter(HarActivityLog.camera_id == camera_id)
    rows = log_q.order_by(HarActivityLog.occurred_at).all()

    by_label: dict[str, int] = {}
    by_camera: dict[str, dict[str, Any]] = {}
    hourly: dict[int, int] = {h: 0 for h in range(24)}
    derived_tags = {"critical": 0, "warning": 0, "info": 0}

    for r in rows:
        lbl = r.predicted_label or "unknown"
        by_label[lbl] = by_label.get(lbl, 0) + 1
        cid = r.camera_id
        bucket = by_camera.setdefault(
            cid,
            {"cameraId": cid, "totalInferences": 0, "primaryCount": 0, "nonPrimaryCount": 0, "byLabel": {}},
        )
        bucket["totalInferences"] += 1
        bucket["byLabel"][lbl] = bucket["byLabel"].get(lbl, 0) + 1
        if r.is_primary_action:
            bucket["primaryCount"] += 1
        else:
            bucket["nonPrimaryCount"] += 1

        conf = float(r.confidence or 0)
        if not r.is_primary_action and conf < HAR_CRITICAL_CONFIDENCE:
            derived_tags["critical"] += 1
        elif not r.is_primary_action:
            derived_tags["warning"] += 1
        elif conf < settings.har_low_confidence_threshold:
            derived_tags["info"] += 1

        at = _as_utc_aware(r.occurred_at)
        if at:
            hourly[at.hour] += 1

    total = len(rows)
    primary = sum(1 for r in rows if r.is_primary_action)
    non_primary = total - primary
    denom = total or 1
    assemble_share = round(primary / denom * 100, 1)
    non_asm_rate = round(non_primary / denom * 100, 1)
    productivity_score = round(
        assemble_share * (1 - min(non_asm_rate / 100, 0.45)),
        1,
    )

    event_q = db.query(Event).filter(
        Event.case_type == "har_action_deviation",
        Event.occurred_at >= start,
        Event.occurred_at <= end,
    )
    if cam_ids:
        event_q = event_q.filter(Event.camera_id.in_(cam_ids))
    elif camera_id:
        event_q = event_q.filter(Event.camera_id == camera_id)
    har_events = event_q.all()

    severity_tags = {"critical": 0, "warning": 0, "info": 0}
    for e in har_events:
        sev = (e.severity or "info").lower()
        if sev in severity_tags:
            severity_tags[sev] += 1

    if total and sum(severity_tags.values()) == 0:
        severity_tags = derived_tags

    config = get_plant_config(db)
    line_cost = float(config.line_cost_per_minute or 125.0)
    action_downtime_min = round(non_primary * HAR_DEVIATION_MINUTES, 1)
    action_cost_usd = round(action_downtime_min * line_cost, 2)

    action_pareto = sorted(
        [{"label": k, "count": v} for k, v in by_label.items() if k != PRIMARY_ACTION],
        key=lambda x: x["count"],
        reverse=True,
    )
    pareto_total = sum(p["count"] for p in action_pareto) or 1
    for p in action_pareto:
        p["pct"] = round(p["count"] / pareto_total * 100, 1)

    cam_name = {c.id: c.name for c in har_cameras}
    camera_rows: list[dict[str, Any]] = []
    for cid in cam_ids:
        stats = by_camera.get(
            cid,
            {"totalInferences": 0, "primaryCount": 0, "nonPrimaryCount": 0, "byLabel": {}},
        )
        t = stats["totalInferences"] or 1
        asm = stats["primaryCount"]
        camera_rows.append(
            {
                "cameraId": cid,
                "name": cam_name.get(cid, cid),
                "totalInferences": stats["totalInferences"],
                "assembleSharePct": round(asm / t * 100, 1),
                "nonAssemblyRatePct": round(stats["nonPrimaryCount"] / t * 100, 1),
                "productivityScore": round(
                    (asm / t * 100) * (1 - min(stats["nonPrimaryCount"] / t, 0.45)),
                    1,
                ),
                "topAction": max(stats["byLabel"].items(), key=lambda x: x[1])[0]
                if stats["byLabel"]
                else None,
            }
        )

    return {
        "date": target,
        "cameraId": camera_id,
        "cameraCount": len(cam_ids),
        "totalInferences": total,
        "assembleSharePct": assemble_share,
        "nonAssemblyRatePct": non_asm_rate,
        "productivityScore": productivity_score,
        "primaryActionLabel": PRIMARY_ACTION,
        "severityTags": severity_tags,
        "actionDowntimeMinutes": action_downtime_min,
        "actionCostUsd": action_cost_usd,
        "lineCostPerMinute": line_cost,
        "byCamera": camera_rows,
        "actionPareto": action_pareto[:10],
        "hourlyCounts": [{"hour": h, "count": hourly[h]} for h in range(24)],
        "hasData": total > 0,
    }


def _aggregate_log_bucket(rows: list[HarActivityLog]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "totalInferences": 0,
            "avgConfidence": None,
            "primaryActionRatePct": None,
            "avgInferMs": None,
            "topLabel": None,
            "topLabelPct": None,
            "byLabel": {},
        }
    confs = [float(r.confidence) for r in rows if r.confidence is not None]
    infer_ms = [float(r.infer_ms) for r in rows if r.infer_ms is not None]
    primary = sum(1 for r in rows if r.is_primary_action)
    by_label: dict[str, int] = {}
    for r in rows:
        lbl = r.predicted_label or "unknown"
        by_label[lbl] = by_label.get(lbl, 0) + 1
    top_label, top_count = max(by_label.items(), key=lambda x: x[1])
    return {
        "totalInferences": total,
        "avgConfidence": round(sum(confs) / len(confs), 4) if confs else None,
        "primaryActionRatePct": round(primary / total * 100, 1),
        "avgInferMs": round(sum(infer_ms) / len(infer_ms), 1) if infer_ms else None,
        "topLabel": top_label,
        "topLabelPct": round(top_count / total * 100, 1),
        "byLabel": by_label,
    }


def _combo_key_for_row(row: HarActivityLog) -> str:
    hp = _parse_hyperparams(row.hyperparams_json)
    video = row.video_name or "unknown"
    return f"{row.model_id}|{hyperparam_key(hp)}|{video}"


def analytics_model_performance(
    db: Session,
    *,
    target_date: str | None = None,
    model_id: str | None = None,
    hyperparam_key_filter: str | None = None,
    combo_key: str | None = None,
    source: str | None = None,
    logs_limit: int = 60,
) -> dict[str, Any]:
    """Group HAR logs by model, hyperparameter preset, and video for model lab analysis."""
    target = target_date or date.today().isoformat()
    start = datetime.fromisoformat(f"{target}T00:00:00")
    end = datetime.fromisoformat(f"{target}T23:59:59.999999")
    logs_limit = max(1, min(500, int(logs_limit)))

    q = db.query(HarActivityLog).filter(
        HarActivityLog.occurred_at >= start,
        HarActivityLog.occurred_at <= end,
    )
    if source:
        q = q.filter(HarActivityLog.source == source)
    rows = q.order_by(HarActivityLog.occurred_at.desc()).all()

    filtered_rows = rows
    if model_id:
        filtered_rows = [r for r in filtered_rows if r.model_id == model_id]
    if hyperparam_key_filter:
        filtered_rows = [
            r
            for r in filtered_rows
            if hyperparam_key(_parse_hyperparams(r.hyperparams_json)) == hyperparam_key_filter
        ]
    if combo_key:
        filtered_rows = [r for r in filtered_rows if _combo_key_for_row(r) == combo_key]

    filter_active = bool(model_id or hyperparam_key_filter or combo_key)
    log_rows = filtered_rows if filter_active else rows
    display_limit = logs_limit if (filter_active or logs_limit > 60) else min(logs_limit, 60)

    by_model: dict[str, list[HarActivityLog]] = {}
    by_hyper: dict[str, list[HarActivityLog]] = {}
    by_combo: dict[str, list[HarActivityLog]] = {}

    for r in rows:
        by_model.setdefault(r.model_id, []).append(r)
        hp = _parse_hyperparams(r.hyperparams_json)
        hp_key = hyperparam_key(hp)
        by_hyper.setdefault(hp_key, []).append(r)
        video = r.video_name or "unknown"
        combo_key = f"{r.model_id}|{hp_key}|{video}"
        by_combo.setdefault(combo_key, []).append(r)

    model_rows: list[dict[str, Any]] = []
    for mid, group in sorted(by_model.items()):
        stats = _aggregate_log_bucket(group)
        sample = group[0]
        model_rows.append(
            {
                "modelId": mid,
                "modelLabel": sample.model_label or mid,
                **stats,
            }
        )
    model_rows.sort(key=lambda x: x["totalInferences"], reverse=True)

    hyper_rows: list[dict[str, Any]] = []
    for hp_key, group in by_hyper.items():
        stats = _aggregate_log_bucket(group)
        sample = group[0]
        hp = _parse_hyperparams(sample.hyperparams_json)
        hyper_rows.append(
            {
                "hyperparamKey": hp_key,
                "hyperparamLabel": hyperparam_label(hp),
                "hyperparams": hp,
                "modelCount": len({r.model_id for r in group}),
                "videoCount": len({r.video_name for r in group if r.video_name}),
                **stats,
            }
        )
    hyper_rows.sort(key=lambda x: x["totalInferences"], reverse=True)

    combo_rows: list[dict[str, Any]] = []
    for combo_key, group in by_combo.items():
        stats = _aggregate_log_bucket(group)
        sample = group[0]
        hp = _parse_hyperparams(sample.hyperparams_json)
        combo_rows.append(
            {
                "comboKey": combo_key,
                "modelId": sample.model_id,
                "modelLabel": sample.model_label or sample.model_id,
                "videoName": sample.video_name,
                "clipUrl": sample.clip_url,
                "previewUrl": sample.snapshot_url or sample.clip_url,
                "hyperparamKey": hyperparam_key(hp),
                "hyperparamLabel": hyperparam_label(hp),
                "hyperparams": hp,
                "source": sample.source,
                "cameraId": sample.camera_id,
                **stats,
            }
        )
    combo_rows.sort(key=lambda x: x["totalInferences"], reverse=True)

    recent_logs = [log_to_dict(r) for r in log_rows[:display_limit]]

    return {
        "date": target,
        "totalLogs": len(rows),
        "filteredCount": len(log_rows) if filter_active else None,
        "hasData": len(rows) > 0,
        "byModel": model_rows,
        "byHyperparams": hyper_rows,
        "byCombo": combo_rows[:40],
        "recentLogs": recent_logs,
    }

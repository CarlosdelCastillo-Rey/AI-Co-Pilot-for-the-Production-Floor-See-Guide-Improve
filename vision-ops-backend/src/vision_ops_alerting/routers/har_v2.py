"""har-research parity API — full sessions, Re-ID, HITL."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import HarAuditSession
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.har_class_labels import list_action_labels
from vision_ops_alerting.services.har_session_store import (
    create_audit_session,
    finalize_audit_session,
    get_audit_session,
    get_preview_summary,
    get_person,
    hitl_review_queue,
    list_audit_sessions,
    list_persons,
    list_preview_frames,
    list_session_events,
    list_track_labels,
    merge_persons,
    person_history,
    person_report,
    person_sessions_summary,
    record_session_event,
    resolve_artifact_path,
    split_track_identity,
    save_human_label,
    save_preview_frame,
    save_preview_manifest,
    save_track_label,
    set_person_display_name,
    tracks_summary,
)

router = APIRouter(prefix="/api/har/v2", tags=["har-v2"])


class SessionCreateBody(BaseModel):
    session_id: str | None = None
    source: str = "eval"
    camera_id: str | None = None
    video_name: str | None = None
    model_id: str | None = None
    model_tag: str | None = None
    checkpoint_name: str | None = None
    classifier_version: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None
    class_names: list[str] = Field(default_factory=list)
    hyperparams: dict[str, Any] = Field(default_factory=dict)


class EventIngestBody(BaseModel):
    track_id: int
    frame_idx: int | None = None
    bbox: list[int] | None = None
    prediction: dict[str, Any]
    label_changed: bool = False
    uncertain: bool = False
    infer_ms: float | None = None
    crop_jpeg_b64: str | None = None
    frame_jpeg_b64: str | None = None
    embedding: list[float] | None = None
    use_person_registry: bool = True
    video_name: str | None = None


class TrackLabelBody(BaseModel):
    session_id: str
    track_id: int
    person_verdict: str
    video: str | None = None
    global_person_id: str | None = None
    display_name: str | None = None
    action_notes: str | None = None
    reviewer: str = "supervisor"
    n_events: int | None = None
    dominant_action: str | None = None
    dominant_confidence: float | None = None


class HumanLabelBody(BaseModel):
    session_id: str | None = None
    event_id: str | None = None
    track_id: int | None = None
    crop_path: str | None = None
    predicted_label: str | None = None
    predicted_confidence: float | None = None
    person_verdict: str | None = None
    action_verdict: str | None = None
    correct_label: str | None = None
    usable_for_training: bool = False
    reviewer: str = "supervisor"
    notes: str | None = None


class PersonNameBody(BaseModel):
    display_name: str


class SplitTrackBody(BaseModel):
    session_id: str
    track_id: int


class MergePersonsBody(BaseModel):
    target_global_person_id: str
    source_global_person_ids: list[str] = Field(min_length=1)


class PreviewFrameBody(BaseModel):
    frame_idx: int
    jpeg_b64: str
    tracks: list[dict[str, Any]] = Field(default_factory=list)


class PreviewManifestBody(BaseModel):
    frames: int
    track_stats: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/sessions")
def api_create_session(body: SessionCreateBody, db: Session = Depends(get_db)) -> dict:
    row = create_audit_session(db, **body.model_dump())
    db.commit()
    return {"session_id": row.id, "status": "ok"}


@router.get("/sessions")
def api_list_sessions(
    db: Session = Depends(get_db),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    source: str | None = None,
    min_events: int = Query(1, ge=1, le=100),
) -> dict:
    rows, total = list_audit_sessions(db, limit=limit, offset=offset, source=source, min_events=min_events)
    db.commit()
    return {"total": total, "limit": limit, "offset": offset, "sessions": rows}


@router.get("/sessions/{session_id}")
def api_get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    data = get_audit_session(db, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@router.post("/sessions/{session_id}/finalize")
def api_finalize_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    if db.get(HarAuditSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    summary = finalize_audit_session(db, session_id)
    db.commit()
    if summary is None:
        return {"status": "discarded", "reason": "session had no inference events"}
    return {"status": "ok", "summary": summary}


@router.get("/sessions/{session_id}/events")
def api_session_events(
    session_id: str,
    db: Session = Depends(get_db),
    track_id: int | None = None,
    limit: int = Query(500, le=2000),
    offset: int = Query(0, ge=0),
) -> dict:
    events, total = list_session_events(db, session_id, track_id=track_id, limit=limit, offset=offset)
    return {"total": total, "events": events}


@router.get("/sessions/{session_id}/tracks")
def api_session_tracks(session_id: str, db: Session = Depends(get_db)) -> dict:
    return {"tracks": tracks_summary(db, session_id)}


@router.post("/sessions/{session_id}/events")
def api_ingest_event(session_id: str, body: EventIngestBody, db: Session = Depends(get_db)) -> dict:
    import base64

    crop_bytes = base64.b64decode(body.crop_jpeg_b64) if body.crop_jpeg_b64 else None
    frame_bytes = base64.b64decode(body.frame_jpeg_b64) if body.frame_jpeg_b64 else None
    try:
        row = record_session_event(
            db,
            session_id=session_id,
            track_id=body.track_id,
            frame_idx=body.frame_idx,
            bbox=body.bbox,
            prediction=body.prediction,
            label_changed=body.label_changed,
            uncertain=body.uncertain,
            infer_ms=body.infer_ms,
            crop_jpeg=crop_bytes,
            frame_jpeg=frame_bytes,
            embedding=body.embedding,
            use_person_registry=body.use_person_registry,
            video_name=body.video_name,
        )
        db.commit()
        return {"status": "ok", "event": row}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/artifacts/{artifact_path:path}")
def api_session_artifact(session_id: str, artifact_path: str) -> FileResponse:
    path = resolve_artifact_path(session_id, artifact_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "application/octet-stream"
    return FileResponse(path, media_type=media)


@router.get("/persons")
def api_list_persons(db: Session = Depends(get_db), limit: int = Query(100, le=500)) -> dict:
    return {"persons": list_persons(db, limit=limit)}


@router.get("/persons/{global_person_id}")
def api_get_person(global_person_id: str, db: Session = Depends(get_db)) -> dict:
    data = get_person(db, global_person_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Person not found")
    data["sessions"] = person_sessions_summary(db, global_person_id)
    return data


@router.patch("/persons/{global_person_id}")
def api_rename_person(global_person_id: str, body: PersonNameBody, db: Session = Depends(get_db)) -> dict:
    if not set_person_display_name(db, global_person_id, body.display_name):
        raise HTTPException(status_code=404, detail="Person not found")
    db.commit()
    return {"status": "ok", "global_person_id": global_person_id, "display_name": body.display_name.strip()}


@router.post("/persons/split-track")
def api_split_track(body: SplitTrackBody, db: Session = Depends(get_db)) -> dict:
    result = split_track_identity(db, session_id=body.session_id, track_id=body.track_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Track not linked to a registry person")
    db.commit()
    return result


@router.post("/persons/merge")
def api_merge_persons(body: MergePersonsBody, db: Session = Depends(get_db)) -> dict:
    result = merge_persons(
        db,
        target_global_person_id=body.target_global_person_id,
        source_global_person_ids=body.source_global_person_ids,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Persons not found or nothing to merge")
    db.commit()
    return result


@router.get("/persons/{global_person_id}/report")
def api_person_report(
    global_person_id: str,
    db: Session = Depends(get_db),
    snapshot_limit: int = Query(60, le=200),
    event_limit: int = Query(1000, le=5000),
) -> dict:
    data = person_report(
        db,
        global_person_id,
        snapshot_limit=snapshot_limit,
        event_limit=event_limit,
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return data


@router.get("/persons/{global_person_id}/history")
def api_person_history(
    global_person_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(200, le=1000),
) -> dict:
    return {"history": person_history(db, global_person_id, limit=limit)}


@router.post("/track-labels")
def api_save_track_label(body: TrackLabelBody, db: Session = Depends(get_db)) -> dict:
    row = save_track_label(db, body.model_dump())
    db.commit()
    return {"status": "ok", "label": row}


@router.get("/track-labels")
def api_list_track_labels(
    db: Session = Depends(get_db),
    session_id: str | None = None,
    limit: int = Query(100, le=500),
) -> dict:
    return {"labels": list_track_labels(db, session_id, limit=limit)}


@router.post("/human-labels")
def api_save_human_label(body: HumanLabelBody, db: Session = Depends(get_db)) -> dict:
    row = save_human_label(db, body.model_dump())
    db.commit()
    return {"status": "ok", "label": row}


@router.get("/action-labels")
def api_action_labels(db: Session = Depends(get_db)) -> dict:
    return list_action_labels(db)


@router.get("/review-queue")
def api_review_queue(db: Session = Depends(get_db), limit: int = Query(50, le=200)) -> dict:
    return {"queue": hitl_review_queue(db, limit=limit)}


@router.post("/sessions/{session_id}/preview-frames")
def api_save_preview_frame(session_id: str, body: PreviewFrameBody, db: Session = Depends(get_db)) -> dict:
    import base64

    try:
        row = save_preview_frame(
            db,
            session_id,
            frame_idx=body.frame_idx,
            jpeg=base64.b64decode(body.jpeg_b64),
            tracks=body.tracks,
        )
        db.commit()
        return {"status": "ok", **row}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/preview-manifest")
def api_save_preview_manifest(session_id: str, body: PreviewManifestBody) -> dict:
    save_preview_manifest(session_id, frames=body.frames, track_stats=body.track_stats)
    return {"status": "ok"}


@router.get("/sessions/{session_id}/preview-frames")
def api_list_preview_frames(
    session_id: str,
    limit: int = Query(2000, le=5000),
) -> dict:
    frames = list_preview_frames(session_id, limit=limit)
    summary = get_preview_summary(session_id)
    return {"frames": frames, "summary": summary, "total": len(frames)}

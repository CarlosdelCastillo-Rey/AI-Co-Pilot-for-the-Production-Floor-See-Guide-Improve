"""Full HAR session audit, person registry, and HITL — har-research parity."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import (
    HarAuditSession,
    HarHumanLabel,
    HarPerson,
    HarPersonAppearance,
    HarSessionEvent,
    HarTrackLabel,
    new_id,
    utcnow,
)

REID_THRESHOLD = settings.har_reid_match_threshold


def _utc_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _json_loads(raw: str | None, default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def session_artifacts_root(session_id: str) -> Path:
    root = settings.session_artifacts_path / session_id
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    return root


def artifact_public_path(session_id: str, rel: str) -> str:
    return f"/api/har/v2/sessions/{session_id}/artifacts/{rel}"


def _crop_url(session_id: str, crop_path: str | None) -> str | None:
    if not crop_path:
        return None
    return artifact_public_path(session_id, crop_path)


def _latest_person_crop_url(db: Session, global_person_id: str) -> str | None:
    ev = (
        db.query(HarSessionEvent)
        .filter(
            HarSessionEvent.global_person_id == global_person_id,
            HarSessionEvent.crop_path.isnot(None),
        )
        .order_by(desc(HarSessionEvent.occurred_at))
        .first()
    )
    if ev is None:
        return None
    return _crop_url(ev.session_id, ev.crop_path)


def _track_crop_url(
    db: Session,
    *,
    global_person_id: str,
    session_id: str,
    track_id: int,
) -> str | None:
    ev = (
        db.query(HarSessionEvent)
        .filter(
            HarSessionEvent.global_person_id == global_person_id,
            HarSessionEvent.session_id == session_id,
            HarSessionEvent.track_id == track_id,
            HarSessionEvent.crop_path.isnot(None),
        )
        .order_by(desc(HarSessionEvent.occurred_at))
        .first()
    )
    if ev is None:
        return None
    return _crop_url(ev.session_id, ev.crop_path)


def _delete_audit_session(db: Session, session_id: str) -> None:
    """Remove session and related rows (appearances/labels are not FK-cascaded)."""
    import shutil

    db.query(HarPersonAppearance).filter(HarPersonAppearance.session_id == session_id).delete(
        synchronize_session=False
    )
    db.query(HarTrackLabel).filter(HarTrackLabel.session_id == session_id).delete(synchronize_session=False)
    db.query(HarHumanLabel).filter(HarHumanLabel.session_id == session_id).delete(synchronize_session=False)
    row = db.get(HarAuditSession, session_id)
    if row is not None:
        db.delete(row)
    artifact_dir = settings.session_artifacts_path / session_id
    if artifact_dir.is_dir():
        shutil.rmtree(artifact_dir, ignore_errors=True)


def purge_trivial_audit_sessions(db: Session, *, max_events: int = 1) -> int:
    """Drop sessions with <= max_events inference rows (empty or single-event runs)."""
    counts = dict(
        db.query(HarSessionEvent.session_id, func.count(HarSessionEvent.id))
        .group_by(HarSessionEvent.session_id)
        .all()
    )
    stale_ids = [
        sid
        for (sid,) in db.query(HarAuditSession.id).all()
        if int(counts.get(sid, 0)) <= max_events
    ]
    for sid in stale_ids:
        _delete_audit_session(db, sid)
    return len(stale_ids)


def _sessions_with_min_events(db: Session, min_events: int = 2):
    """Session ids that have at least min_events recorded inference rows."""
    return (
        db.query(HarSessionEvent.session_id)
        .group_by(HarSessionEvent.session_id)
        .having(func.count(HarSessionEvent.id) >= min_events)
    )


def create_audit_session(
    db: Session,
    *,
    session_id: str | None = None,
    source: str = "eval",
    camera_id: str | None = None,
    video_name: str | None = None,
    model_id: str | None = None,
    model_tag: str | None = None,
    checkpoint_name: str | None = None,
    classifier_version: str | None = None,
    embedding_model: str | None = None,
    embedding_version: str | None = None,
    class_names: list[str] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> HarAuditSession:
    sid = session_id or new_id("har")
    row = HarAuditSession(
        id=sid,
        source=source,
        camera_id=camera_id,
        video_name=video_name,
        model_id=model_id,
        model_tag=model_tag,
        checkpoint_name=checkpoint_name,
        classifier_version=classifier_version,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        class_names_json=json.dumps(class_names or []),
        hyperparams_json=json.dumps(hyperparams or {}),
    )
    db.add(row)
    session_artifacts_root(sid)
    db.flush()
    return row


def finalize_audit_session(db: Session, session_id: str) -> dict[str, Any] | None:
    sess = db.get(HarAuditSession, session_id)
    if sess is None:
        return None
    events = (
        db.query(HarSessionEvent)
        .filter(HarSessionEvent.session_id == session_id)
        .all()
    )
    by_label: dict[str, int] = {}
    by_track: dict[int, int] = {}
    for ev in events:
        lbl = ev.action_label or ev.raw_label
        if lbl:
            by_label[lbl] = by_label.get(lbl, 0) + 1
        by_track[ev.track_id] = by_track.get(ev.track_id, 0) + 1
    n_confirmed = sum(1 for e in events if e.label_changed)
    if len(events) == 0:
        _delete_audit_session(db, session_id)
        db.flush()
        return None
    summary = {
        "session_id": session_id,
        "n_events": len(events),
        "n_confirmed": n_confirmed,
        "n_tracks": len(by_track),
        "events_by_label": by_label,
        "events_by_track": {str(k): v for k, v in sorted(by_track.items())},
    }
    sess.ended_at = utcnow()
    sess.n_events = len(events)
    sess.n_confirmed = n_confirmed
    sess.n_tracks = len(by_track)
    sess.summary_json = json.dumps(summary)
    db.flush()
    return summary


def _person_id_for_session_track(db: Session, session_id: str, track_id: int) -> str | None:
    row = (
        db.query(HarSessionEvent.global_person_id)
        .filter(
            HarSessionEvent.session_id == session_id,
            HarSessionEvent.track_id == track_id,
            HarSessionEvent.global_person_id.isnot(None),
        )
        .order_by(HarSessionEvent.event_seq)
        .first()
    )
    return row[0] if row else None


def _person_ids_for_video(db: Session, video_name: str | None) -> set[str]:
    if not video_name:
        return set()
    rows = (
        db.query(HarPersonAppearance.global_person_id)
        .filter(HarPersonAppearance.video_name == video_name)
        .distinct()
        .all()
    )
    return {r[0] for r in rows if r[0]}


def _resolve_person(
    db: Session,
    embedding: np.ndarray,
    *,
    session_id: str,
    track_id: int,
    event_id: str,
    video_name: str | None,
    frame_idx: int | None,
    action_label: str | None,
    confidence: float | None,
) -> tuple[str, float, str | None]:
    emb = _normalize(embedding)
    now = utcnow()

    pinned = _person_id_for_session_track(db, session_id, track_id)
    if pinned:
        gid = pinned
        best_score = 1.0
        person = db.get(HarPerson, gid)
        if person and person.centroid_blob:
            old = np.frombuffer(person.centroid_blob, dtype=np.float32)
            n = int(person.n_appearances or 0)
            new_c = _normalize((old * n + emb) / (n + 1))
            person.centroid_blob = new_c.tobytes()
            person.last_seen_at = now
    else:
        best_id: str | None = None
        best_score = -1.0
        if settings.har_reid_auto_link:
            candidate_ids = _person_ids_for_video(db, video_name) if video_name else None
            persons = db.query(HarPerson).all()
            for p in persons:
                if candidate_ids is not None and p.id not in candidate_ids:
                    continue
                if not p.centroid_blob:
                    continue
                centroid = np.frombuffer(p.centroid_blob, dtype=np.float32)
                score = _cosine(emb, centroid)
                if score > best_score:
                    best_score = score
                    best_id = p.id

        if settings.har_reid_auto_link and best_id is not None and best_score >= REID_THRESHOLD:
            gid = best_id
            person = db.get(HarPerson, gid)
            if person and person.centroid_blob:
                old = np.frombuffer(person.centroid_blob, dtype=np.float32)
                n = int(person.n_appearances or 0)
                new_c = _normalize((old * n + emb) / (n + 1))
                person.centroid_blob = new_c.tobytes()
                person.last_seen_at = now
        else:
            gid = f"person-{uuid.uuid4().hex[:10]}"
            best_score = 1.0
            db.add(
                HarPerson(
                    id=gid,
                    embedding_dim=int(emb.size),
                    centroid_blob=emb.tobytes(),
                    n_appearances=0,
                    first_seen_at=now,
                    last_seen_at=now,
                    meta_json=json.dumps({"source": "track_identity"}),
                )
            )

    db.add(
        HarPersonAppearance(
            id=new_id("app"),
            global_person_id=gid,
            session_id=session_id,
            track_id=track_id,
            event_id=event_id,
            video_name=video_name,
            frame_idx=frame_idx,
            action_label=action_label,
            confidence=confidence,
            match_score=round(best_score, 4),
            occurred_at=now,
        )
    )
    person = db.get(HarPerson, gid)
    if person:
        person.n_appearances = int(person.n_appearances or 0) + 1
        person.last_seen_at = now
    display_name = person.display_name.strip() if person and person.display_name else None
    return gid, best_score, display_name


def record_session_event(
    db: Session,
    *,
    session_id: str,
    track_id: int,
    frame_idx: int | None,
    bbox: list[int] | None,
    prediction: dict[str, Any],
    label_changed: bool = False,
    uncertain: bool = False,
    infer_ms: float | None = None,
    crop_jpeg: bytes | None = None,
    frame_jpeg: bytes | None = None,
    embedding: list[float] | np.ndarray | None = None,
    use_person_registry: bool = True,
    video_name: str | None = None,
) -> dict[str, Any]:
    sess = db.get(HarAuditSession, session_id)
    if sess is None:
        raise ValueError(f"Unknown session {session_id}")

    seq = (
        db.query(func.max(HarSessionEvent.event_seq))
        .filter(HarSessionEvent.session_id == session_id)
        .scalar()
        or 0
    ) + 1
    # IDs must be globally unique (PK); seq alone matches har-research filenames per session.
    event_id = f"{session_id}-evt_{seq:05d}"
    if len(event_id) > 64:
        event_id = new_id("evt")
    root = session_artifacts_root(session_id)

    crop_rel: str | None = None
    frame_rel: str | None = None
    emb_rel: str | None = None

    if crop_jpeg:
        crop_name = f"{event_id}_tid{track_id}_crop.jpg"
        (root / "frames" / crop_name).write_bytes(crop_jpeg)
        crop_rel = f"frames/{crop_name}"
    if frame_jpeg:
        slug = (prediction.get("label") or prediction.get("raw_label") or "unknown")
        slug = str(slug).replace(" ", "_").lower()[:32]
        frame_name = f"f{frame_idx or 0:05d}_tid{track_id}_{slug}.jpg"
        (root / "frames" / frame_name).write_bytes(frame_jpeg)
        frame_rel = f"frames/{frame_name}"
    if embedding is not None and (label_changed or uncertain):
        emb_arr = np.asarray(embedding, dtype=np.float32)
        emb_name = f"{event_id}_tid{track_id}.npy"
        np.save(root / "embeddings" / emb_name, emb_arr)
        emb_rel = f"embeddings/{emb_name}"

    global_person_id: str | None = None
    reid_score: float | None = None
    display_name: str | None = None
    if use_person_registry and embedding is not None:
        emb_arr = np.asarray(embedding, dtype=np.float32)
        global_person_id, reid_score, display_name = _resolve_person(
            db,
            emb_arr,
            session_id=session_id,
            track_id=track_id,
            event_id=event_id,
            video_name=video_name or sess.video_name,
            frame_idx=frame_idx,
            action_label=prediction.get("label") or prediction.get("raw_label"),
            confidence=prediction.get("confidence") or prediction.get("raw_confidence"),
        )

    ev = HarSessionEvent(
        id=event_id,
        session_id=session_id,
        event_seq=seq,
        track_id=track_id,
        global_person_id=global_person_id,
        reid_match_score=reid_score,
        frame_idx=frame_idx,
        bbox_json=json.dumps(bbox) if bbox else None,
        action_label=prediction.get("label"),
        raw_label=prediction.get("raw_label") or prediction.get("label"),
        confidence=prediction.get("confidence"),
        raw_confidence=prediction.get("raw_confidence") or prediction.get("confidence"),
        class_index=prediction.get("class_index"),
        top_k_json=json.dumps(prediction.get("top_k") or []),
        all_probs_json=json.dumps(prediction.get("all_probs") or {}),
        label_changed=label_changed,
        uncertain=uncertain,
        infer_ms=infer_ms,
        crop_path=crop_rel,
        frame_path=frame_rel,
        embedding_path=emb_rel,
    )
    db.add(ev)
    sess.n_events = int(sess.n_events or 0) + 1
    if label_changed:
        sess.n_confirmed = int(sess.n_confirmed or 0) + 1
    db.flush()

    return {
        "event_id": event_id,
        "session_id": session_id,
        "track_id": track_id,
        "global_person_id": global_person_id,
        "display_name": display_name,
        "reid_match_score": reid_score,
        "crop_url": artifact_public_path(session_id, crop_rel) if crop_rel else None,
        "frame_url": artifact_public_path(session_id, frame_rel) if frame_rel else None,
        "embedding_url": artifact_public_path(session_id, emb_rel) if emb_rel else None,
    }


def list_audit_sessions(
    db: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    source: str | None = None,
    min_events: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    purge_trivial_audit_sessions(db, max_events=0)
    eligible = _sessions_with_min_events(db, min_events=min_events)
    q = db.query(HarAuditSession).filter(HarAuditSession.id.in_(eligible))
    if source:
        q = q.filter(HarAuditSession.source == source)
    total = q.count()
    rows = q.order_by(desc(HarAuditSession.started_at)).offset(offset).limit(limit).all()
    return [_session_dict(r) for r in rows], total


def get_audit_session(db: Session, session_id: str) -> dict[str, Any] | None:
    row = db.get(HarAuditSession, session_id)
    if row is None:
        return None
    out = _session_dict(row)
    out["summary"] = _json_loads(row.summary_json, {})
    return out


def _session_dict(row: HarAuditSession) -> dict[str, Any]:
    return {
        "session_id": row.id,
        "source": row.source,
        "camera_id": row.camera_id,
        "video_name": row.video_name,
        "model_id": row.model_id,
        "model_tag": row.model_tag,
        "checkpoint_name": row.checkpoint_name,
        "classifier_version": row.classifier_version,
        "embedding_model": row.embedding_model,
        "started_at": _utc_iso(row.started_at),
        "ended_at": _utc_iso(row.ended_at),
        "n_events": row.n_events,
        "n_confirmed": row.n_confirmed,
        "n_tracks": row.n_tracks,
        "hyperparams": _json_loads(row.hyperparams_json, {}),
        "class_names": _json_loads(row.class_names_json, []),
    }


def list_session_events(
    db: Session,
    session_id: str,
    *,
    track_id: int | None = None,
    limit: int = 500,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    q = db.query(HarSessionEvent).filter(HarSessionEvent.session_id == session_id)
    if track_id is not None:
        q = q.filter(HarSessionEvent.track_id == track_id)
    total = q.count()
    rows = q.order_by(HarSessionEvent.event_seq).offset(offset).limit(limit).all()
    return [_event_dict(session_id, r) for r in rows], total


def _event_dict(session_id: str, ev: HarSessionEvent) -> dict[str, Any]:
    return {
        "event_id": ev.id,
        "session_id": session_id,
        "track_id": ev.track_id,
        "global_person_id": ev.global_person_id,
        "reid_match_score": ev.reid_match_score,
        "frame_idx": ev.frame_idx,
        "bbox": _json_loads(ev.bbox_json),
        "action_label": ev.action_label,
        "raw_label": ev.raw_label,
        "confidence": ev.confidence,
        "raw_confidence": ev.raw_confidence,
        "class_index": ev.class_index,
        "top_k": _json_loads(ev.top_k_json, []),
        "all_probs": _json_loads(ev.all_probs_json, {}),
        "label_changed": ev.label_changed,
        "uncertain": ev.uncertain,
        "infer_ms": ev.infer_ms,
        "occurred_at": _utc_iso(ev.occurred_at),
        "crop_url": artifact_public_path(session_id, ev.crop_path) if ev.crop_path else None,
        "frame_url": artifact_public_path(session_id, ev.frame_path) if ev.frame_path else None,
        "embedding_url": artifact_public_path(session_id, ev.embedding_path) if ev.embedding_path else None,
    }


def tracks_summary(db: Session, session_id: str) -> list[dict[str, Any]]:
    events, _ = list_session_events(db, session_id, limit=10000)
    if not events:
        return []
    by_track: dict[int, list[dict]] = {}
    for ev in events:
        by_track.setdefault(int(ev["track_id"]), []).append(ev)
    out: list[dict[str, Any]] = []
    for tid, grp in sorted(by_track.items()):
        grp.sort(key=lambda e: e.get("frame_idx") or 0)
        labels = [str(e.get("action_label") or e.get("raw_label") or "") for e in grp if e.get("action_label") or e.get("raw_label")]
        dominant = max(set(labels), key=labels.count) if labels else ""
        dom_confs = [float(e["confidence"] or 0) for e in grp if e.get("action_label") == dominant]
        gids = [e.get("global_person_id") for e in grp if e.get("global_person_id")]
        gid = max(set(gids), key=gids.count) if gids else None
        display_name: str | None = None
        if gid:
            person = db.get(HarPerson, gid)
            if person and person.display_name:
                display_name = person.display_name.strip()
        if not display_name:
            label_row = (
                db.query(HarTrackLabel)
                .filter(
                    HarTrackLabel.session_id == session_id,
                    HarTrackLabel.track_id == tid,
                    HarTrackLabel.display_name.isnot(None),
                )
                .order_by(desc(HarTrackLabel.reviewed_at))
                .first()
            )
            if label_row and label_row.display_name:
                display_name = label_row.display_name.strip()
        out.append(
            {
                "session_id": session_id,
                "track_id": tid,
                "global_person_id": gid,
                "display_name": display_name,
                "n_inferences": len(grp),
                "frame_first": grp[0].get("frame_idx"),
                "frame_last": grp[-1].get("frame_idx"),
                "dominant_action": dominant,
                "dominant_confidence": max(dom_confs) if dom_confs else 0,
                "label_changes": sum(1 for e in grp if e.get("label_changed")),
                "uncertain_count": sum(1 for e in grp if e.get("uncertain")),
                "sample_crop_url": next((e.get("crop_url") for e in grp if e.get("crop_url")), None),
            }
        )
    return out


def list_persons(db: Session, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = db.query(HarPerson).order_by(desc(HarPerson.last_seen_at)).limit(limit).all()
    out: list[dict[str, Any]] = []
    for p in rows:
        n_tracks = (
            db.query(HarPersonAppearance.session_id, HarPersonAppearance.track_id)
            .filter(HarPersonAppearance.global_person_id == p.id)
            .distinct()
            .count()
        )
        out.append(
            {
                "global_person_id": p.id,
                "display_name": p.display_name,
                "n_appearances": p.n_appearances,
                "n_tracks": int(n_tracks),
                "first_seen_at": _utc_iso(p.first_seen_at),
                "last_seen_at": _utc_iso(p.last_seen_at),
                "thumbnail_url": _latest_person_crop_url(db, p.id),
            }
        )
    return out


def get_person(db: Session, global_person_id: str) -> dict[str, Any] | None:
    p = db.get(HarPerson, global_person_id)
    if p is None:
        return None
    return {
        "global_person_id": p.id,
        "display_name": p.display_name,
        "n_appearances": p.n_appearances,
        "first_seen_at": _utc_iso(p.first_seen_at),
        "last_seen_at": _utc_iso(p.last_seen_at),
        "thumbnail_url": _latest_person_crop_url(db, global_person_id),
    }


def set_person_display_name(db: Session, global_person_id: str, display_name: str) -> bool:
    p = db.get(HarPerson, global_person_id)
    if p is None:
        return False
    p.display_name = display_name.strip()
    db.flush()
    return True


def split_track_identity(db: Session, *, session_id: str, track_id: int) -> dict[str, Any] | None:
    """Move one session track to a fresh registry person (undo false Re-ID merges)."""
    events = (
        db.query(HarSessionEvent)
        .filter(
            HarSessionEvent.session_id == session_id,
            HarSessionEvent.track_id == track_id,
            HarSessionEvent.global_person_id.isnot(None),
        )
        .order_by(HarSessionEvent.event_seq)
        .all()
    )
    if not events:
        return None

    old_gid = events[0].global_person_id
    if old_gid is None:
        return None

    now = utcnow()
    new_gid = f"person-{uuid.uuid4().hex[:10]}"
    emb: np.ndarray | None = None
    for ev in events:
        if not ev.embedding_path:
            continue
        path = resolve_artifact_path(session_id, ev.embedding_path)
        if path and path.is_file():
            emb = np.load(path)
            break

    db.add(
        HarPerson(
            id=new_gid,
            embedding_dim=int(emb.size) if emb is not None else 0,
            centroid_blob=_normalize(emb).tobytes() if emb is not None else None,
            n_appearances=0,
            first_seen_at=now,
            last_seen_at=now,
            meta_json=json.dumps({"source": "split_track", "from": old_gid}),
        )
    )

    moved = 0
    for ev in events:
        ev.global_person_id = new_gid
        moved += 1

    appearances = (
        db.query(HarPersonAppearance)
        .filter(
            HarPersonAppearance.session_id == session_id,
            HarPersonAppearance.track_id == track_id,
            HarPersonAppearance.global_person_id == old_gid,
        )
        .all()
    )
    for app in appearances:
        app.global_person_id = new_gid

    old_person = db.get(HarPerson, old_gid)
    new_person = db.get(HarPerson, new_gid)
    if new_person:
        new_person.n_appearances = len(appearances)
    if old_person:
        old_person.n_appearances = max(0, int(old_person.n_appearances or 0) - len(appearances))
        if old_person.n_appearances == 0 and not db.query(HarPersonAppearance).filter(
            HarPersonAppearance.global_person_id == old_gid
        ).count():
            db.delete(old_person)

    db.flush()
    return {
        "status": "ok",
        "session_id": session_id,
        "track_id": track_id,
        "old_global_person_id": old_gid,
        "new_global_person_id": new_gid,
        "events_moved": moved,
    }


def merge_persons(
    db: Session,
    *,
    target_global_person_id: str,
    source_global_person_ids: list[str],
) -> dict[str, Any] | None:
    """Combine duplicate registry persons into one stable identity."""
    sources = sorted({s for s in source_global_person_ids if s and s != target_global_person_id})
    if not sources:
        return None

    target = db.get(HarPerson, target_global_person_id)
    if target is None:
        return None

    source_persons: list[HarPerson] = []
    for sid in sources:
        person = db.get(HarPerson, sid)
        if person is not None:
            source_persons.append(person)
    if not source_persons:
        return None

    total_n = int(target.n_appearances or 0)
    combined_emb: np.ndarray | None = None
    if target.centroid_blob and total_n > 0:
        combined_emb = np.frombuffer(target.centroid_blob, dtype=np.float32) * total_n

    first_seen = target.first_seen_at
    last_seen = target.last_seen_at
    moved_appearances = 0
    moved_events = 0
    merged_ids: list[str] = []

    for src in source_persons:
        sid = src.id
        merged_ids.append(sid)
        n = int(src.n_appearances or 0)
        if src.centroid_blob and n > 0:
            emb = np.frombuffer(src.centroid_blob, dtype=np.float32)
            combined_emb = emb * n if combined_emb is None else combined_emb + emb * n
            total_n += n

        if src.first_seen_at and (first_seen is None or src.first_seen_at < first_seen):
            first_seen = src.first_seen_at
        if src.last_seen_at and (last_seen is None or src.last_seen_at > last_seen):
            last_seen = src.last_seen_at
        if not target.display_name and src.display_name:
            target.display_name = src.display_name

        for ev in db.query(HarSessionEvent).filter(HarSessionEvent.global_person_id == sid).all():
            ev.global_person_id = target_global_person_id
            moved_events += 1

        for app in db.query(HarPersonAppearance).filter(HarPersonAppearance.global_person_id == sid).all():
            app.global_person_id = target_global_person_id
            moved_appearances += 1

        for lbl in db.query(HarTrackLabel).filter(HarTrackLabel.global_person_id == sid).all():
            lbl.global_person_id = target_global_person_id

        db.delete(src)

    if combined_emb is not None and total_n > 0:
        new_centroid = _normalize(combined_emb / total_n)
        target.centroid_blob = new_centroid.tobytes()
        target.embedding_dim = int(new_centroid.size)

    target.n_appearances = (
        db.query(HarPersonAppearance)
        .filter(HarPersonAppearance.global_person_id == target_global_person_id)
        .count()
    )
    if first_seen is not None:
        target.first_seen_at = first_seen
    if last_seen is not None:
        target.last_seen_at = last_seen

    db.flush()
    return {
        "status": "ok",
        "target_global_person_id": target_global_person_id,
        "merged_global_person_ids": merged_ids,
        "appearances_moved": moved_appearances,
        "events_moved": moved_events,
    }


def person_history(db: Session, global_person_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = (
        db.query(HarPersonAppearance)
        .filter(HarPersonAppearance.global_person_id == global_person_id)
        .order_by(desc(HarPersonAppearance.occurred_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "session_id": r.session_id,
            "track_id": r.track_id,
            "event_id": r.event_id,
            "video_name": r.video_name,
            "frame_idx": r.frame_idx,
            "action_label": r.action_label,
            "confidence": r.confidence,
            "match_score": r.match_score,
            "occurred_at": _utc_iso(r.occurred_at),
            "crop_url": _crop_url_for_appearance(db, r),
        }
        for r in rows
    ]


def person_report(
    db: Session,
    global_person_id: str,
    *,
    snapshot_limit: int = 60,
    event_limit: int = 1000,
) -> dict[str, Any] | None:
    """Full person dossier — metrics, action mix, snapshots, and event log."""
    person = get_person(db, global_person_id)
    if person is None:
        return None

    ev_q = db.query(HarSessionEvent).filter(HarSessionEvent.global_person_id == global_person_id)
    n_events = int(ev_q.count())
    sessions = person_sessions_summary(db, global_person_id)

    agg = (
        ev_q.with_entities(
            func.avg(HarSessionEvent.confidence).label("avg_conf"),
            func.min(HarSessionEvent.confidence).label("min_conf"),
            func.max(HarSessionEvent.confidence).label("max_conf"),
            func.avg(HarSessionEvent.reid_match_score).label("avg_reid"),
            func.min(HarSessionEvent.reid_match_score).label("min_reid"),
            func.max(HarSessionEvent.reid_match_score).label("max_reid"),
            func.min(HarSessionEvent.occurred_at).label("first_seen"),
            func.max(HarSessionEvent.occurred_at).label("last_seen"),
            func.min(HarSessionEvent.frame_idx).label("frame_first"),
            func.max(HarSessionEvent.frame_idx).label("frame_last"),
        )
        .one()
    )
    n_uncertain = int(ev_q.filter(HarSessionEvent.uncertain.is_(True)).count())
    n_label_changes = int(ev_q.filter(HarSessionEvent.label_changed.is_(True)).count())
    n_sessions = len({r["session_id"] for r in sessions})
    n_tracks = len(sessions)
    videos = sorted({str(r.get("video") or "") for r in sessions if r.get("video")})

    action_rows = (
        db.query(
            HarSessionEvent.action_label,
            func.count(HarSessionEvent.id).label("n"),
            func.avg(HarSessionEvent.confidence).label("avg_conf"),
        )
        .filter(
            HarSessionEvent.global_person_id == global_person_id,
            HarSessionEvent.action_label.isnot(None),
            HarSessionEvent.action_label != "",
        )
        .group_by(HarSessionEvent.action_label)
        .order_by(desc("n"))
        .all()
    )
    action_breakdown: list[dict[str, Any]] = []
    dominant_action = ""
    for label, cnt, avg_conf in action_rows:
        action = str(label)
        if not dominant_action:
            dominant_action = action
        action_breakdown.append(
            {
                "action": action,
                "count": int(cnt),
                "share": round(int(cnt) / max(n_events, 1), 4),
                "avg_confidence": round(float(avg_conf), 4) if avg_conf is not None else None,
            }
        )

    snapshot_rows = (
        ev_q.filter(HarSessionEvent.crop_path.isnot(None))
        .order_by(desc(HarSessionEvent.occurred_at))
        .limit(snapshot_limit)
        .all()
    )
    snapshots = [
        {
            "event_id": ev.id,
            "session_id": ev.session_id,
            "track_id": ev.track_id,
            "frame_idx": ev.frame_idx,
            "action_label": ev.action_label,
            "confidence": ev.confidence,
            "occurred_at": _utc_iso(ev.occurred_at),
            "crop_url": _crop_url(ev.session_id, ev.crop_path),
            "frame_url": _crop_url(ev.session_id, ev.frame_path) if ev.frame_path else None,
            "uncertain": ev.uncertain,
            "label_changed": ev.label_changed,
        }
        for ev in snapshot_rows
    ]

    event_rows = ev_q.order_by(desc(HarSessionEvent.occurred_at)).limit(event_limit).all()
    events = [_event_dict(ev.session_id, ev) for ev in event_rows]

    human_label_rows = (
        db.query(HarHumanLabel)
        .join(HarSessionEvent, HarHumanLabel.event_id == HarSessionEvent.id)
        .filter(HarSessionEvent.global_person_id == global_person_id)
        .order_by(desc(HarHumanLabel.reviewed_at))
        .limit(200)
        .all()
    )
    human_labels = [
        {
            "label_id": row.id,
            "event_id": row.event_id,
            "session_id": row.session_id,
            "track_id": row.track_id,
            "predicted_label": row.predicted_label,
            "predicted_confidence": row.predicted_confidence,
            "person_verdict": row.person_verdict,
            "action_verdict": row.action_verdict,
            "correct_label": row.correct_label,
            "usable_for_training": row.usable_for_training,
            "reviewer": row.reviewer,
            "reviewed_at": _utc_iso(row.reviewed_at),
            "notes": row.notes,
        }
        for row in human_label_rows
    ]

    track_label_rows = (
        db.query(HarTrackLabel)
        .filter(HarTrackLabel.global_person_id == global_person_id)
        .order_by(desc(HarTrackLabel.reviewed_at))
        .limit(100)
        .all()
    )
    track_labels = [
        {
            "session_id": row.session_id,
            "track_id": row.track_id,
            "video": row.video,
            "person_verdict": row.person_verdict,
            "display_name": row.display_name,
            "dominant_action": row.dominant_action,
            "dominant_confidence": row.dominant_confidence,
            "n_events": row.n_events,
            "reviewer": row.reviewer,
            "reviewed_at": _utc_iso(row.reviewed_at),
            "action_notes": row.action_notes,
        }
        for row in track_label_rows
    ]

    return {
        **person,
        "metrics": {
            "n_events": n_events,
            "n_sessions": n_sessions,
            "n_tracks": n_tracks,
            "n_videos": len(videos),
            "videos": videos,
            "avg_confidence": round(float(agg.avg_conf), 4) if agg.avg_conf is not None else None,
            "min_confidence": round(float(agg.min_conf), 4) if agg.min_conf is not None else None,
            "max_confidence": round(float(agg.max_conf), 4) if agg.max_conf is not None else None,
            "avg_reid_match": round(float(agg.avg_reid), 4) if agg.avg_reid is not None else None,
            "min_reid_match": round(float(agg.min_reid), 4) if agg.min_reid is not None else None,
            "max_reid_match": round(float(agg.max_reid), 4) if agg.max_reid is not None else None,
            "n_uncertain": n_uncertain,
            "n_label_changes": n_label_changes,
            "dominant_action": dominant_action,
            "first_seen_at": _utc_iso(agg.first_seen),
            "last_seen_at": _utc_iso(agg.last_seen),
            "frame_first": agg.frame_first,
            "frame_last": agg.frame_last,
        },
        "action_breakdown": action_breakdown,
        "sessions": sessions,
        "snapshots": snapshots,
        "events": events,
        "human_labels": human_labels,
        "track_labels": track_labels,
        "events_shown": len(events),
        "events_truncated": n_events > event_limit,
        "snapshots_shown": len(snapshots),
        "snapshots_truncated": (
            ev_q.filter(HarSessionEvent.crop_path.isnot(None)).count() > snapshot_limit
        ),
    }


def _crop_url_for_appearance(db: Session, row: HarPersonAppearance) -> str | None:
    if row.event_id:
        ev = db.get(HarSessionEvent, row.event_id)
        if ev and ev.crop_path:
            return _crop_url(ev.session_id, ev.crop_path)
    return _track_crop_url(
        db,
        global_person_id=row.global_person_id,
        session_id=row.session_id,
        track_id=row.track_id,
    )


def person_sessions_summary(db: Session, global_person_id: str) -> list[dict[str, Any]]:
    rows = (
        db.query(
            HarPersonAppearance.session_id,
            HarPersonAppearance.track_id,
            HarPersonAppearance.video_name,
            func.count(HarPersonAppearance.id).label("n"),
            func.min(HarPersonAppearance.frame_idx).label("frame_first"),
            func.max(HarPersonAppearance.frame_idx).label("frame_last"),
            func.max(HarPersonAppearance.occurred_at).label("last_ts"),
            func.avg(HarPersonAppearance.match_score).label("match_score"),
        )
        .filter(HarPersonAppearance.global_person_id == global_person_id)
        .group_by(
            HarPersonAppearance.session_id,
            HarPersonAppearance.track_id,
            HarPersonAppearance.video_name,
        )
        .order_by(desc("last_ts"))
        .all()
    )
    out = []
    for r in rows:
        top_action = (
            db.query(HarPersonAppearance.action_label, func.count(HarPersonAppearance.id))
            .filter(
                HarPersonAppearance.global_person_id == global_person_id,
                HarPersonAppearance.session_id == r.session_id,
                HarPersonAppearance.track_id == r.track_id,
                HarPersonAppearance.action_label.isnot(None),
            )
            .group_by(HarPersonAppearance.action_label)
            .order_by(desc(func.count(HarPersonAppearance.id)))
            .first()
        )
        out.append(
            {
                "session_id": r.session_id,
                "track_id": r.track_id,
                "video": r.video_name,
                "n_appearances": r.n,
                "frame_first": r.frame_first,
                "frame_last": r.frame_last,
                "dominant_action": top_action[0] if top_action else "",
                "match_score": round(float(r.match_score), 4) if r.match_score is not None else None,
                "sample_crop_url": _track_crop_url(
                    db,
                    global_person_id=global_person_id,
                    session_id=r.session_id,
                    track_id=r.track_id,
                ),
            }
        )
    return out


def save_track_label(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    session_id = str(payload["session_id"])
    track_id = int(payload["track_id"])
    gid = payload.get("global_person_id") or _person_id_for_session_track(db, session_id, track_id)
    name = (payload.get("display_name") or "").strip() or None
    verdict = str(payload.get("person_verdict") or "unknown")

    row = HarTrackLabel(
        id=new_id("tl"),
        session_id=session_id,
        track_id=track_id,
        video=payload.get("video"),
        global_person_id=gid,
        person_verdict=verdict,
        display_name=name,
        action_notes=payload.get("action_notes"),
        reviewer=payload.get("reviewer"),
        n_events=payload.get("n_events"),
        dominant_action=payload.get("dominant_action"),
        dominant_confidence=payload.get("dominant_confidence"),
    )
    db.add(row)

    registry_updated = False
    if gid and name and verdict == "yes":
        registry_updated = set_person_display_name(db, gid, name)

    db.flush()
    return {
        "label_id": row.id,
        "session_id": row.session_id,
        "track_id": row.track_id,
        "person_verdict": row.person_verdict,
        "display_name": row.display_name,
        "global_person_id": gid,
        "registry_updated": registry_updated,
    }


def list_track_labels(db: Session, session_id: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    q = db.query(HarTrackLabel)
    if session_id:
        q = q.filter(HarTrackLabel.session_id == session_id)
    rows = q.order_by(desc(HarTrackLabel.reviewed_at)).limit(limit).all()
    return [
        {
            "label_id": r.id,
            "session_id": r.session_id,
            "track_id": r.track_id,
            "person_verdict": r.person_verdict,
            "display_name": r.display_name,
            "reviewed_at": _utc_iso(r.reviewed_at),
        }
        for r in rows
    ]


def save_human_label(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    from vision_ops_alerting.services.har_class_labels import register_custom_har_action_label

    action_verdict = str(payload.get("action_verdict") or "").lower()
    person_verdict = str(payload.get("person_verdict") or "").lower()
    correct_label = str(payload.get("correct_label") or "").strip()
    if action_verdict == "no" and correct_label:
        correct_label = register_custom_har_action_label(correct_label, db) or correct_label

    usable = bool(payload.get("usable_for_training"))
    if person_verdict == "yes" and action_verdict not in ("dont_know", "maybe"):
        if action_verdict == "yes":
            usable = True
        elif action_verdict == "no" and correct_label:
            usable = True

    row = HarHumanLabel(
        id=new_id("hl"),
        session_id=payload.get("session_id"),
        event_id=payload.get("event_id"),
        track_id=payload.get("track_id"),
        crop_path=payload.get("crop_path"),
        predicted_label=payload.get("predicted_label"),
        predicted_confidence=payload.get("predicted_confidence"),
        person_verdict=payload.get("person_verdict"),
        action_verdict=payload.get("action_verdict"),
        correct_label=correct_label or payload.get("correct_label"),
        usable_for_training=usable,
        reviewer=payload.get("reviewer"),
        notes=payload.get("notes"),
    )
    db.add(row)
    db.flush()
    return {"label_id": row.id, "usable_for_training": row.usable_for_training}


def hitl_review_queue(db: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    """Uncertain or low-confidence events not yet human-reviewed."""
    reviewed_ids = {
        r.event_id
        for r in db.query(HarHumanLabel.event_id).filter(HarHumanLabel.event_id.isnot(None)).all()
        if r.event_id
    }
    rows = (
        db.query(HarSessionEvent)
        .filter(
            (HarSessionEvent.uncertain.is_(True))
            | ((HarSessionEvent.confidence.isnot(None)) & (HarSessionEvent.confidence < 0.35))
        )
        .order_by(desc(HarSessionEvent.occurred_at))
        .limit(limit * 3)
        .all()
    )
    out: list[dict[str, Any]] = []
    for ev in rows:
        if ev.id in reviewed_ids:
            continue
        probs = _json_loads(ev.all_probs_json, {})
        entropy = 0.0
        if probs:
            for p in probs.values():
                if p and p > 0:
                    entropy -= float(p) * math.log(float(p))
        score = entropy * 2 + (1 - float(ev.confidence or 0))
        d = _event_dict(ev.session_id, ev)
        d["priority_score"] = round(score, 4)
        out.append(d)
        if len(out) >= limit:
            break
    out.sort(key=lambda x: -x["priority_score"])
    return out


def resolve_artifact_path(session_id: str, rel_path: str) -> Path | None:
    root = settings.session_artifacts_path / session_id
    target = (root / rel_path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return None
    return target if target.is_file() else None


def save_preview_frame(
    db: Session,
    session_id: str,
    *,
    frame_idx: int,
    jpeg: bytes,
    tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sess = db.get(HarAuditSession, session_id)
    if sess is None:
        raise ValueError(f"Unknown session {session_id}")
    root = session_artifacts_root(session_id)
    preview_dir = root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rel = f"preview/f{frame_idx:05d}.jpg"
    (root / rel).write_bytes(jpeg)
    manifest_path = root / "preview" / "index.jsonl"
    row = {"frame_idx": frame_idx, "path": rel, "n_tracks": len(tracks or []), "tracks": tracks or []}
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"frame_idx": frame_idx, "url": artifact_public_path(session_id, rel)}


def save_preview_manifest(session_id: str, *, frames: int, track_stats: list[dict[str, Any]]) -> None:
    root = session_artifacts_root(session_id)
    summary = {"frames": frames, "track_stats": track_stats}
    (root / "preview" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def list_preview_frames(session_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
    root = session_artifacts_root(session_id)
    manifest_path = root / "preview" / "index.jsonl"
    if not manifest_path.is_file():
        preview_dir = root / "preview"
        if preview_dir.is_dir():
            out = []
            for p in sorted(preview_dir.glob("f*.jpg")):
                try:
                    fi = int(p.stem[1:])
                except ValueError:
                    continue
                rel = f"preview/{p.name}"
                out.append({"frame_idx": fi, "url": artifact_public_path(session_id, rel), "n_tracks": 0, "tracks": []})
            return out[:limit]
        return []
    out: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = row.get("path") or f"preview/f{int(row['frame_idx']):05d}.jpg"
        row["url"] = artifact_public_path(session_id, rel)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def get_preview_summary(session_id: str) -> dict[str, Any]:
    path = session_artifacts_root(session_id) / "preview" / "summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

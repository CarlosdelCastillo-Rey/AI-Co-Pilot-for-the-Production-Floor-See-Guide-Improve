"""Persist and query HAR model inference runs (SQLite)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, joinedload

from vision_ops_alerting.db.models import HarInferenceResult, HarInferenceRun, new_id


def _parse_probe_time(probe: dict[str, Any]) -> datetime:
    raw = probe.get("updated_at") or probe.get("probed_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def record_har_run(db: Session, payload: dict[str, Any]) -> HarInferenceRun:
    """
    Store a probe-all or single-probe payload from vision-ops-backend.

    Expected keys: run_type, source, clip_path?, frame_count?, shared_clip,
    status, probes[], errors[]
    """
    run_type = payload.get("run_type") or ("batch" if len(payload.get("probes") or []) > 1 else "single")
    probes: list[dict[str, Any]] = list(payload.get("probes") or [])
    errors: list[dict[str, Any]] = list(payload.get("errors") or [])

    status = payload.get("status") or ("ok" if probes else "error")
    if probes and errors:
        status = "partial"

    run = HarInferenceRun(
        id=new_id("har-run"),
        run_type=run_type,
        clip_source=str(payload.get("source") or ""),
        clip_path=payload.get("clip_path"),
        frame_count=payload.get("frame_count"),
        shared_clip=bool(payload.get("shared_clip", True)),
        status=status,
        error_count=len(errors),
        meta_json=json.dumps({"errors": errors}, ensure_ascii=False) if errors else None,
    )
    db.add(run)
    db.flush()

    for probe in probes:
        pred = probe.get("prediction") or {}
        db.add(
            HarInferenceResult(
                id=new_id("har-res"),
                run_id=run.id,
                model_id=str(probe.get("model_id") or ""),
                camera_id=str(probe.get("camera_id") or ""),
                predicted_label=pred.get("label"),
                class_index=pred.get("class_index"),
                confidence=float(pred["confidence"]) if pred.get("confidence") is not None else None,
                backend=probe.get("backend"),
                device=probe.get("device"),
                top_k_json=json.dumps(pred.get("top_k") or [], ensure_ascii=False),
                overlay_json=json.dumps(probe.get("overlay") or {}, ensure_ascii=False),
                status=str(probe.get("status") or "ok"),
                probed_at=_parse_probe_time(probe),
            )
        )

    for err in errors:
        model_id = str(err.get("model_id") or "unknown")
        camera_id = err.get("camera_id") or _camera_id_for_model(model_id)

        db.add(
            HarInferenceResult(
                id=new_id("har-res"),
                run_id=run.id,
                model_id=model_id,
                camera_id=camera_id or "",
                predicted_label=None,
                class_index=None,
                confidence=None,
                backend=None,
                device=None,
                top_k_json=None,
                overlay_json=None,
                status="error",
                error_message=str(err.get("error") or "")[:4000],
                probed_at=datetime.now(timezone.utc),
            )
        )

    db.commit()
    db.refresh(run)
    return run


def _camera_id_for_model(model_id: str) -> str:
    mapping = {
        "v2-vjepa": "cam-har-01",
        "v2-dinov2": "cam-har-02",
    }
    return mapping.get(model_id, "")


def run_to_dict(run: HarInferenceRun, *, include_results: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": run.id,
        "runType": run.run_type,
        "clipSource": run.clip_source,
        "clipPath": run.clip_path,
        "frameCount": run.frame_count,
        "sharedClip": run.shared_clip,
        "status": run.status,
        "errorCount": run.error_count,
        "createdAt": run.created_at.isoformat() if run.created_at else None,
    }
    if include_results:
        out["results"] = [result_to_dict(r) for r in run.results]
    return out


def result_to_dict(row: HarInferenceResult) -> dict[str, Any]:
    top_k = []
    if row.top_k_json:
        try:
            top_k = json.loads(row.top_k_json)
        except json.JSONDecodeError:
            top_k = []
    overlay = None
    if row.overlay_json:
        try:
            overlay = json.loads(row.overlay_json)
        except json.JSONDecodeError:
            overlay = None
    return {
        "id": row.id,
        "runId": row.run_id,
        "modelId": row.model_id,
        "cameraId": row.camera_id,
        "predictedLabel": row.predicted_label,
        "classIndex": row.class_index,
        "confidence": row.confidence,
        "backend": row.backend,
        "device": row.device,
        "topK": top_k,
        "overlay": overlay,
        "status": row.status,
        "errorMessage": row.error_message,
        "probedAt": row.probed_at.isoformat() if row.probed_at else None,
    }


def list_har_runs(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    camera_id: str | None = None,
    model_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    q = db.query(HarInferenceRun).order_by(HarInferenceRun.created_at.desc())
    if camera_id or model_id:
        q = q.join(HarInferenceResult)
        if camera_id:
            q = q.filter(HarInferenceResult.camera_id == camera_id)
        if model_id:
            q = q.filter(HarInferenceResult.model_id == model_id)
        q = q.distinct(HarInferenceRun.id)
    total = q.count()
    runs = q.offset(offset).limit(limit).all()
    return [run_to_dict(r, include_results=False) for r in runs], total


def get_har_run(db: Session, run_id: str) -> dict[str, Any] | None:
    run = (
        db.query(HarInferenceRun)
        .options(joinedload(HarInferenceRun.results))
        .filter(HarInferenceRun.id == run_id)
        .first()
    )
    if run is None:
        return None
    data = run_to_dict(run, include_results=True)
    data["results"].sort(key=lambda r: r.get("modelId") or "")
    return data


def list_har_results(
    db: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    camera_id: str | None = None,
    model_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    q = db.query(HarInferenceResult).order_by(HarInferenceResult.probed_at.desc())
    if camera_id:
        q = q.filter(HarInferenceResult.camera_id == camera_id)
    if model_id:
        q = q.filter(HarInferenceResult.model_id == model_id)
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return [result_to_dict(r) for r in rows], total


def latest_results_by_model(db: Session) -> list[dict[str, Any]]:
    """Latest successful prediction per model_id (for watch / dashboard widgets)."""
    from sqlalchemy import func

    sub = (
        db.query(
            HarInferenceResult.model_id.label("model_id"),
            func.max(HarInferenceResult.probed_at).label("max_at"),
        )
        .filter(HarInferenceResult.status == "ok")
        .group_by(HarInferenceResult.model_id)
        .subquery()
    )
    rows = (
        db.query(HarInferenceResult)
        .join(
            sub,
            (HarInferenceResult.model_id == sub.c.model_id)
            & (HarInferenceResult.probed_at == sub.c.max_at),
        )
        .order_by(HarInferenceResult.model_id)
        .all()
    )
    return [result_to_dict(r) for r in rows]

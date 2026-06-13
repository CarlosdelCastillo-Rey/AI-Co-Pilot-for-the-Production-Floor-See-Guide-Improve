"""In-process HAR probe persistence (no HTTP)."""

from __future__ import annotations

import logging
from typing import Any

from vision_ops_backend.config import settings

logger = logging.getLogger(__name__)


def persist_har_run(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not settings.har_persist_enabled:
        return None
    try:
        from vision_ops_alerting.db.session import SessionLocal
        from vision_ops_alerting.services.har_inference_store import record_har_run, run_to_dict

        with SessionLocal() as db:
            run = record_har_run(db, payload)
            db.commit()
            return run_to_dict(run, include_results=True)
    except Exception as exc:
        logger.warning("HAR persist failed: %s", exc)
    return None


def build_persist_payload(
    *,
    run_type: str,
    source: str,
    probes: list[dict[str, Any]],
    errors: list[dict[str, str]],
    status: str,
    clip_path: str | None = None,
    frame_count: int | None = None,
    shared_clip: bool = True,
) -> dict[str, Any]:
    return {
        "run_type": run_type,
        "source": source,
        "clip_path": clip_path,
        "frame_count": frame_count,
        "shared_clip": shared_clip,
        "status": status,
        "probes": probes,
        "errors": errors,
    }

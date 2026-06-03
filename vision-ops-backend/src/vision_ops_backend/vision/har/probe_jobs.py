"""Background jobs for long-running HAR probe-all (avoids HTTP proxy timeouts)."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_probe_all_job(*, clip_path: str | None = None, reshuffle_videos: bool = True) -> str:
    job_id = f"har-{uuid.uuid4().hex[:12]}"
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "created_at": _now_iso(),
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def _run() -> None:
        try:
            from vision_ops_backend.vision.har.probe_runner import run_all_har_probes

            result = run_all_har_probes(clip_path=clip_path, reshuffle_videos=reshuffle_videos)
            with _lock:
                _jobs[job_id].update(
                    status=result.get("status", "ok"),
                    finished_at=_now_iso(),
                    result=result,
                )
        except Exception as exc:
            with _lock:
                _jobs[job_id].update(
                    status="error",
                    finished_at=_now_iso(),
                    error=str(exc),
                )

    threading.Thread(target=_run, daemon=True, name=f"har-probe-all-{job_id}").start()
    return job_id


def get_probe_all_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None

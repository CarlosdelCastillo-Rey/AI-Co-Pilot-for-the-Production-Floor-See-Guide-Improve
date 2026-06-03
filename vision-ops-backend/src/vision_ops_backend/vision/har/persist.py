"""Push HAR probe results to vision-ops-alerting SQLite."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from vision_ops_backend.config import settings

logger = logging.getLogger(__name__)


def persist_har_run(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST probe batch to alerting /api/har/runs. Returns run dict or None on failure."""
    if not settings.har_persist_enabled:
        return None
    base = settings.alerting_api_url.rstrip("/")
    url = f"{base}/api/har/runs"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("run")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        logger.warning("HAR persist HTTP %s: %s", exc.code, detail)
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

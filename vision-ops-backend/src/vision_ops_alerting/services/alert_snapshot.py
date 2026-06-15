"""Fetch and cache camera JPEGs for alert email / Telegram delivery."""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from vision_ops_alerting.config import PROJECT_ROOT, settings
from vision_ops_alerting.schemas import IndustrialContext
from vision_ops_backend.vision.paths import alert_snapshot_path, preview_path, still_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotAsset:
    local_path: Path
    public_url: str
    data_uri: str


def _snapshots_dir() -> Path:
    directory = PROJECT_ROOT / "data" / "alert_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cache_jpeg(data: bytes, *, prefix: str = "alert") -> SnapshotAsset:
    safe_prefix = "".join(c if c.isalnum() else "-" for c in prefix)[:24] or "alert"
    snapshot_id = f"{safe_prefix}-{uuid.uuid4().hex[:16]}"
    path = _snapshots_dir() / f"{snapshot_id}.jpg"
    path.write_bytes(data)
    public = f"{settings.public_base_url}/api/alerting/snapshots/{snapshot_id}.jpg"
    encoded = base64.standard_b64encode(data).decode("ascii")
    return SnapshotAsset(
        local_path=path,
        public_url=public,
        data_uri=f"data:image/jpeg;base64,{encoded}",
    )


def _read_local_artifact(camera_id: str) -> bytes | None:
    for path_fn in (still_path, preview_path):
        path = path_fn(camera_id)
        if path.is_file():
            return path.read_bytes()
    return None


def _fetch_camera_still(camera_id: str) -> bytes | None:
    return _read_local_artifact(camera_id)


def resolve_snapshot(ctx: IndustrialContext) -> SnapshotAsset | None:
    """Resolve a JPEG for the alert context (explicit path or cached still)."""
    if not settings.alert_snapshots_enabled:
        return None

    snap = (ctx.evidence.snapshot_path or "").strip()
    if snap:
        if snap.startswith("data:image"):
            return None
        if snap.startswith("/api/vision/alert-snapshots/"):
            # Actual trigger snapshot saved by capture_har_trigger_snapshot —
            # resolve the file directly instead of falling back to the camera still.
            filename = snap.rsplit("/", 1)[-1]
            sid = filename[:-4] if filename.endswith(".jpg") else filename
            snap_path = alert_snapshot_path(sid)
            if snap_path.is_file():
                return _cache_jpeg(snap_path.read_bytes(), prefix="snap")
            # file missing (e.g. restarted server) — fall through to camera still
        elif snap.startswith("/api/vision/"):
            camera_id = ctx.camera_id
            data = _read_local_artifact(camera_id)
            if data:
                return _cache_jpeg(data, prefix="ctx")
        else:
            path = Path(snap).expanduser()
            if path.is_file():
                return _cache_jpeg(path.read_bytes(), prefix="ctx")

    data = _fetch_camera_still(ctx.camera_id)
    if data:
        return _cache_jpeg(data, prefix=ctx.camera_id)
    return None


def enrich_context(ctx: IndustrialContext) -> tuple[IndustrialContext, SnapshotAsset | None]:
    """Attach a cached local snapshot path to the context for Telegram + timeline."""
    asset = resolve_snapshot(ctx)
    if asset is None:
        return ctx, None
    enriched = ctx.model_copy(
        update={
            "evidence": ctx.evidence.model_copy(update={"snapshot_path": str(asset.local_path)}),
        }
    )
    return enriched, asset


def snapshot_url_for_email(asset: SnapshotAsset | None, template_url: str = "") -> str:
    """Prefer inline data URI so MailerSend does not need a public image host."""
    if asset is not None:
        return asset.data_uri
    return template_url or ""


def cache_alert_jpeg(data: bytes, *, prefix: str = "alert") -> SnapshotAsset:
    """Public helper — persist JPEG bytes for notifications."""
    return _cache_jpeg(data, prefix=prefix)


def resolve_snapshot_path(snapshot_id: str) -> Path | None:
    if not snapshot_id or ".." in snapshot_id or "/" in snapshot_id:
        return None
    if not snapshot_id.endswith(".jpg"):
        snapshot_id = f"{snapshot_id}.jpg"
    stem = snapshot_id[: -len(".jpg")]
    if not stem.replace("-", "").isalnum():
        return None
    path = _snapshots_dir() / snapshot_id
    return path if path.is_file() else None

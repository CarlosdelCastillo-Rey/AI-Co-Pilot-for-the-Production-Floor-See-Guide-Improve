"""Serve cached alert snapshot JPEGs (email embed + Telegram source)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from vision_ops_alerting.services.alert_snapshot import resolve_snapshot_path

router = APIRouter(tags=["alert-snapshots"])


@router.get("/api/alerting/snapshots/{snapshot_file}")
def get_alert_snapshot(snapshot_file: str) -> FileResponse:
    path = resolve_snapshot_path(snapshot_file)
    if path is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")

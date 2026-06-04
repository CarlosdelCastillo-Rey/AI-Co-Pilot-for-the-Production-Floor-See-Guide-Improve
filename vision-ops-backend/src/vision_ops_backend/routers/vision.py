"""Vision foundation-model probes (DINO heatmap, V-JEPA) for industrial mock cameras."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from vision_ops_backend.config import settings
from vision_ops_backend.vision import CAM_ASSEMBLY, CAM_WAREHOUSE
from vision_ops_backend.vision.har.constants import HAR_CAMERA_IDS
from vision_ops_backend.vision.paths import (
    alert_snapshot_path,
    heatmap_overlay_path,
    heatmap_path,
    preview_path,
    repo_root,
    still_path,
)
from vision_ops_backend.vision.probe_runner import run_probe
from vision_ops_backend.vision.store import heatmap_available, load_last_probe

router = APIRouter(prefix="/api/vision", tags=["vision"])

def _allowed_cameras() -> set[str]:
    allowed = {CAM_ASSEMBLY, CAM_WAREHOUSE}
    if settings.har_enabled:
        allowed.update(HAR_CAMERA_IDS)
    return allowed


class ProbeRequest(BaseModel):
    camera_id: str = Field(..., description="cam-01 (assembly) or cam-02 (warehouse)")
    mode: str = Field(
        "auto",
        description="auto | dinov3_heatmap | vjepa_anomaly",
    )
    clip_path: str | None = Field(None, description="Optional .mp4 or path override")
    set_baseline: bool = Field(
        False,
        description="For cam-02: store embedding as new normal baseline",
    )


@router.get("/status")
def vision_status() -> dict:
    cam01 = load_last_probe(CAM_ASSEMBLY)
    cam02 = load_last_probe(CAM_WAREHOUSE)
    return {
        "ready": True,
        "cameras": {
            CAM_ASSEMBLY: {
                "label": "Assembly — DINO heatmap (D2)",
                "last_probe": cam01 is not None,
                "has_heatmap": heatmap_available(CAM_ASSEMBLY),
                "heatmapUrl": f"/api/vision/artifacts/{CAM_ASSEMBLY}/overlay"
                if heatmap_available(CAM_ASSEMBLY)
                else None,
            },
            CAM_WAREHOUSE: {
                "label": "Warehouse — V-JEPA anomaly (V1)",
                "last_probe": cam02 is not None,
                "last_severity": (cam02 or {}).get("severity"),
                "anomaly_score": (cam02 or {}).get("anomaly_score"),
            },
        },
    }


@router.get("/storage")
def vision_storage() -> dict:
    data = repo_root() / "vision-ops-backend" / "data" / "vision"
    return {
        "summary": (
            "Batch probes for mock industrial cameras. cam-01 stores DINO-style heatmaps; "
            "cam-02 stores V-JEPA (or motion) embeddings and anomaly scores. Not real-time."
        ),
        "paths": {
            "artifact_root": str(data),
            "dinov3_code": str(repo_root() / "models" / "dinov3-main"),
            "vjepa_code": str(repo_root() / "models" / "vjepa2-main"),
            "optional_deps": "torch, transformers (pip install for HF weights)",
        },
        "cameras": {
            CAM_ASSEMBLY: {"default_mode": "dinov3_heatmap"},
            CAM_WAREHOUSE: {"default_mode": "vjepa_anomaly"},
        },
        "gitignored": True,
    }


@router.post("/probe")
def vision_probe(body: ProbeRequest) -> dict:
    allowed = _allowed_cameras()
    if body.camera_id not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"camera_id must be one of {sorted(allowed)}",
        )

    mode = body.mode
    if mode == "auto":
        mode = "dinov3_heatmap" if body.camera_id == CAM_ASSEMBLY else "vjepa_anomaly"

    try:
        result = run_probe(
            body.camera_id,
            mode,
            clip_path=body.clip_path,
            set_baseline=body.set_baseline,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "probe": result}


@router.get("/artifacts/{camera_id}/heatmap")
def artifact_heatmap(camera_id: str) -> FileResponse:
    if camera_id not in _allowed_cameras():
        raise HTTPException(status_code=404, detail="Camera not found")
    path = heatmap_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not generated yet — POST /api/vision/probe")
    return FileResponse(path, media_type="image/png")


@router.get("/artifacts/{camera_id}/still")
def artifact_still(camera_id: str) -> FileResponse:
    if camera_id not in _allowed_cameras():
        raise HTTPException(status_code=404, detail="Camera not found")
    path = still_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Still frame not cached — open Live or run probe")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/artifacts/{camera_id}/preview")
def artifact_preview(camera_id: str) -> FileResponse:
    if camera_id not in _allowed_cameras():
        raise HTTPException(status_code=404, detail="Camera not found")
    path = preview_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Preview not generated yet — POST /api/vision/probe")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/artifacts/{camera_id}/overlay")
def artifact_overlay(camera_id: str) -> FileResponse:
    if camera_id not in _allowed_cameras():
        raise HTTPException(status_code=404, detail="Camera not found")
    path = heatmap_overlay_path(camera_id)
    if not path.is_file():
        path = heatmap_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not generated yet — POST /api/vision/probe")
    media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media)


@router.get("/alert-snapshots/{snapshot_file}")
def alert_snapshot(snapshot_file: str) -> FileResponse:
    """JPEG captured when a HAR activity log triggered an alert."""
    if not snapshot_file.endswith(".jpg") or ".." in snapshot_file or "/" in snapshot_file:
        raise HTTPException(status_code=400, detail="Invalid snapshot id")
    snapshot_id = snapshot_file[: -len(".jpg")]
    if not snapshot_id or not snapshot_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid snapshot id")
    path = alert_snapshot_path(snapshot_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Alert snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/cameras/{camera_id}/last")
def last_probe(camera_id: str) -> dict:
    if camera_id not in _allowed_cameras():
        raise HTTPException(status_code=404, detail="Camera not found")
    probe = load_last_probe(camera_id)
    if probe is None:
        raise HTTPException(status_code=404, detail="No probe run yet for this camera")
    base = settings.public_api_base.rstrip("/")
    if heatmap_available(camera_id):
        probe = {
            **probe,
            "heatmapUrl": f"{base}/api/vision/artifacts/{camera_id}/overlay",
        }
    return probe

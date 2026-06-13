from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Camera, new_id
from vision_ops_alerting.db.session import get_db
from vision_ops_alerting.services.cameras import (
    apply_config,
    camera_to_dict,
    list_cameras_merged,
    live_stats,
    parse_config,
)

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    location: str
    zone: str | None = None
    source_type: str = Field(default="rtsp", alias="sourceType")
    stream_url: str | None = Field(default=None, alias="streamUrl")
    coords: str | None = None
    inference_model: str | None = Field(default=None, alias="inferenceModel")
    inference_task: str | None = Field(default=None, alias="inferenceTask")
    image_url: str | None = Field(default=None, alias="imageUrl")
    backend_camera_id: str | None = Field(default=None, alias="backendCameraId")
    enabled: bool = True
    config: dict | None = None

    model_config = {"populate_by_name": True}


class CameraUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    zone: str | None = None
    source_type: str | None = Field(default=None, alias="sourceType")
    stream_url: str | None = Field(default=None, alias="streamUrl")
    coords: str | None = None
    inference_model: str | None = Field(default=None, alias="inferenceModel")
    inference_task: str | None = Field(default=None, alias="inferenceTask")
    image_url: str | None = Field(default=None, alias="imageUrl")
    backend_camera_id: str | None = Field(default=None, alias="backendCameraId")
    enabled: bool | None = None
    status: str | None = None
    config: dict | None = None

    model_config = {"populate_by_name": True}


@router.get("")
async def list_cameras(
    db: Session = Depends(get_db),
    status: str | None = Query(None),
    model: str | None = Query(None, alias="inferenceModel"),
    zone: str | None = Query(None),
    q: str | None = Query(None),
):
    cameras = await list_cameras_merged(db)
    if status:
        cameras = [c for c in cameras if c.get("status") == status.lower()]
    if model:
        cameras = [c for c in cameras if (c.get("inferenceModel") or "").lower() == model.lower()]
    if zone:
        cameras = [c for c in cameras if zone.lower() in (c.get("zone") or "").lower()]
    if q:
        needle = q.lower()
        cameras = [
            c
            for c in cameras
            if needle in c.get("name", "").lower()
            or needle in c.get("location", "").lower()
            or needle in (c.get("zone") or "").lower()
        ]
    return cameras


@router.get("/stats/live")
async def get_live_stats(db: Session = Depends(get_db)):
    cameras = await list_cameras_merged(db)
    return await live_stats(db, cameras)


@router.post("", status_code=201)
def create_camera(body: CameraCreate, db: Session = Depends(get_db)):
    cam_id = new_id("cam")
    sort_order = db.query(Camera).count()
    status = "live" if body.source_type == "webcam" else "offline"

    if body.inference_model and not body.inference_task:
        from vision_ops_alerting.services.cameras import MODEL_LABELS

        inference_task = MODEL_LABELS.get(body.inference_model.lower(), {}).get("task")
    else:
        inference_task = body.inference_task

    camera = Camera(
        id=cam_id,
        name=body.name,
        location=body.location,
        zone=body.zone,
        source_type=body.source_type,
        stream_url=body.stream_url,
        coords=body.coords,
        inference_model=body.inference_model,
        inference_task=inference_task,
        image_url=body.image_url,
        backend_camera_id=body.backend_camera_id,
        enabled=body.enabled,
        status=status,
        sort_order=sort_order,
    )
    if body.config:
        apply_config(camera, body.config)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera_to_dict(camera)


@router.get("/{camera_id}")
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    cameras = await list_cameras_merged(db)
    merged = next((c for c in cameras if c["id"] == camera_id), None)
    return merged or camera_to_dict(camera)


@router.patch("/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate, db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    for field, attr in [
        (body.name, "name"),
        (body.location, "location"),
        (body.zone, "zone"),
        (body.source_type, "source_type"),
        (body.stream_url, "stream_url"),
        (body.coords, "coords"),
        (body.inference_model, "inference_model"),
        (body.inference_task, "inference_task"),
        (body.image_url, "image_url"),
        (body.backend_camera_id, "backend_camera_id"),
        (body.enabled, "enabled"),
        (body.status, "status"),
    ]:
        if field is not None:
            setattr(camera, attr, field)

    if body.config is not None:
        existing = parse_config(camera)
        existing.update(body.config)
        apply_config(camera, existing)

    camera.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(camera)
    return camera_to_dict(camera)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(camera)
    db.commit()

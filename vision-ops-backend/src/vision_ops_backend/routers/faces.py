"""Face enrollment, storage info, and status (SFace)."""

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from vision_ops_backend.config import settings
from vision_ops_backend.face.paths import (
    enrollment_preview_path,
    faces_data_dir,
    owner_feature_path,
    repo_root,
    yunet_model_path,
    sface_model_path,
)
from vision_ops_backend.webcam import WebcamCapture

router = APIRouter(prefix="/api/faces", tags=["faces"])


class EnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Display name on live overlay")


def _capture(request: Request) -> WebcamCapture | None:
    return getattr(request.app.state, "webcam", None)


def _engine(request: Request):
    cap = _capture(request)
    return cap.face_engine if cap is not None else None


@router.get("/status")
def face_status(request: Request) -> dict:
    engine = _engine(request)
    preview = enrollment_preview_path()
    if engine is None:
        return {
            "enrolled": False,
            "ready": False,
            "error": "Face engine disabled",
            "has_preview": False,
        }
    return {
        "enrolled": engine.is_enrolled,
        "ready": engine.is_ready,
        "name": engine.owner_name,
        "error": engine.error,
        "has_preview": preview.is_file(),
        "previewUrl": "/api/faces/preview" if preview.is_file() else None,
    }


@router.get("/storage")
def storage_info() -> dict:
    """Explain where enrollment data and models live (optional face API)."""
    root = repo_root()
    data_dir = faces_data_dir()
    emb = owner_feature_path()
    preview = enrollment_preview_path()

    return {
        "summary": (
            "Live video is never saved. Enrollment stores a numeric face embedding "
            "(not a photo gallery) plus your display name. One optional preview JPG "
            "is saved for your reference."
        ),
        "paths": {
            "repo_root": str(root),
            "enrollment_dir": str(data_dir),
            "embedding_file": str(emb),
            "embedding_exists": emb.is_file(),
            "preview_image": str(preview),
            "preview_exists": preview.is_file(),
            "yunet_model": str(yunet_model_path()),
            "sface_model": str(sface_model_path()),
        },
        "stored_items": [
            {
                "id": "owner_embedding",
                "file": "vision-ops-backend/data/faces/owner.npz",
                "contents": "SFace 128-d vector + registered name",
                "used_for_recognition": True,
            },
            {
                "id": "enrollment_preview",
                "file": "vision-ops-backend/data/faces/enrollment_preview.jpg",
                "contents": "Single cropped snapshot from enrollment (optional reference)",
                "used_for_recognition": False,
            },
            {
                "id": "live_stream",
                "file": None,
                "contents": "Webcam frames processed in RAM only",
                "used_for_recognition": False,
            },
            {
                "id": "onnx_models",
                "file": "vision-ops-backend/weights/face_detection_yunet/, vision-ops-backend/weights/face_recognition_sface/",
                "contents": "Downloaded ONNX weights (shared, not personal data)",
                "used_for_recognition": True,
            },
        ],
        "gitignored": True,
    }


@router.get("/preview")
def enrollment_preview():
    path = enrollment_preview_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No enrollment preview saved yet")
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="image/jpeg")


@router.post("/enroll")
def enroll_face(request: Request, body: EnrollRequest) -> dict:
    cap = _capture(request)
    if cap is None:
        raise HTTPException(status_code=503, detail="Webcam disabled (WEBCAM_ENABLED=false)")

    engine = cap.face_engine

    if engine is None or not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=engine.error if engine else "Face engine not available",
        )

    if not cap.is_running:
        raise HTTPException(status_code=503, detail=cap.error or "Webcam not running")

    frames: list = []
    for _ in range(12):
        frame = cap.get_latest_frame()
        if frame is not None:
            frames.append(frame.copy())
        time.sleep(0.25)

    result = engine.enroll_from_frames(frames, name=body.name.strip())
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Enrollment failed"))

    return {
        "status": "enrolled",
        "name": result.get("name"),
        "samples": result.get("samples"),
        "storage": {
            "embedding": str(owner_feature_path()),
            "preview": str(enrollment_preview_path()),
        },
    }


@router.delete("/enroll")
def delete_enrollment(request: Request) -> dict:
    engine = _engine(request)
    removed: list[str] = []

    for path in (owner_feature_path(), enrollment_preview_path()):
        if path.is_file():
            path.unlink()
            removed.append(str(path))

    if engine is not None:
        engine.clear_enrollment()

    return {"status": "deleted", "removed": removed}

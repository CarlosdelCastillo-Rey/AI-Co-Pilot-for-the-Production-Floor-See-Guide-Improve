"""Camera list and MJPEG stream (webcam mock)."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vision_ops_backend.config import settings
from vision_ops_backend.industrial_cameras import list_industrial_cameras
from vision_ops_backend.webcam import WebcamCapture, mjpeg_generator

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


def _get_capture(request: Request) -> WebcamCapture:
    return request.app.state.webcam


@router.get("")
def list_cameras(request: Request) -> list[dict]:
    capture = _get_capture(request)
    base = settings.public_api_base.rstrip("/")
    is_live = capture.is_running and capture.error is None
    status = "live" if is_live else "offline"
    face = capture.face_engine
    face_status = ""
    if face and face.is_ready:
        face_status = " | FACE OK" if face.is_enrolled else " | ENROLL FACE"

    payload: dict = {
        "id": settings.camera_id,
        "name": "Camera 01 - Webcam (dev)",
        "location": "Local / MacBook",
        "status": status,
        "image": "",
        "coords": f"WEBCAM | DEV{face_status}",
        "overlays": capture.get_overlays() if is_live else [],
        "error": capture.error or (face.error if face else None),
    }
    if is_live:
        payload["streamUrl"] = f"{base}/api/cameras/{settings.camera_id}/stream"

    cameras = [payload]
    if settings.vision_enabled:
        cameras.extend(list_industrial_cameras())
    return cameras


@router.get("/{camera_id}/stream")
def camera_stream(camera_id: str, request: Request) -> StreamingResponse:
    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail="Camera not found")

    capture = _get_capture(request)
    if not capture.is_running:
        raise HTTPException(status_code=503, detail=capture.error or "Webcam not available")

    return StreamingResponse(
        mjpeg_generator(capture, fps=settings.mjpeg_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

"""Camera list and MJPEG stream (webcam mock)."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vision_ops_backend.config import settings
from vision_ops_backend.industrial_cameras import list_industrial_cameras
from vision_ops_backend.vision.har.constants import is_har_camera
from vision_ops_backend.vision.har.live_stream import get_har_live_manager
from vision_ops_backend.webcam import WebcamCapture, mjpeg_generator

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


def _get_capture(request: Request) -> WebcamCapture | None:
    return getattr(request.app.state, "webcam", None)


@router.get("")
def list_cameras(request: Request) -> list[dict]:
    cameras: list[dict] = []
    capture = _get_capture(request)
    if capture is not None:
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
        cameras.append(payload)

    if settings.vision_enabled:
        cameras.extend(list_industrial_cameras())
    return cameras


@router.get("/{camera_id}/stream")
def camera_stream(camera_id: str, request: Request) -> StreamingResponse:
    if is_har_camera(camera_id) and settings.har_live_enabled:
        stream = get_har_live_manager().get_stream(camera_id)
        if stream is None or not stream.is_running:
            raise HTTPException(status_code=503, detail="HAR live stream not available")
        return StreamingResponse(
            _har_mjpeg_generator(stream, fps=settings.har_live_stream_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    if camera_id != settings.camera_id:
        raise HTTPException(status_code=404, detail="Camera not found")

    capture = _get_capture(request)
    if capture is None:
        raise HTTPException(status_code=503, detail="Webcam disabled (WEBCAM_ENABLED=false)")
    if not capture.is_running:
        raise HTTPException(status_code=503, detail=capture.error or "Webcam not available")

    return StreamingResponse(
        mjpeg_generator(capture, fps=settings.mjpeg_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _har_mjpeg_generator(stream, fps: int):
    import time

    boundary = b"frame"
    interval = 1.0 / max(fps, 1)
    while stream.is_running:
        jpeg = stream.get_jpeg()
        if jpeg:
            yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)

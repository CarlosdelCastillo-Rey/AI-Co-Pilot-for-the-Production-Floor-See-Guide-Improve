"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vision_ops_backend.config import settings
from vision_ops_backend.face.sface_live import SFaceLiveEngine
from vision_ops_backend.routers import cameras, faces, har, health, vision
from vision_ops_backend.webcam import WebcamCapture


@asynccontextmanager
async def lifespan(app: FastAPI):
    webcam: WebcamCapture | None = None
    if settings.webcam_enabled:
        face_engine = None
        if settings.face_enabled:
            face_engine = SFaceLiveEngine(
                owner_name=settings.owner_name,
                match_threshold=settings.face_match_threshold,
            )
        webcam = WebcamCapture(
            camera_index=settings.camera_index,
            face_engine=face_engine,
        )
        webcam.start()
    app.state.webcam = webcam

    har_live = None
    if settings.har_enabled and settings.har_live_enabled:
        from vision_ops_backend.vision.har.live_stream import get_har_live_manager

        har_live = get_har_live_manager()
        har_live.start_all()
        app.state.har_live = har_live

    if settings.har_enabled and settings.har_auto_probe_on_start:
        import logging
        import threading

        def _auto_har_probe() -> None:
            try:
                from vision_ops_backend.vision.har.probe_runner import run_all_har_probes

                run_all_har_probes(reshuffle_videos=True)
            except Exception as exc:
                logging.getLogger(__name__).warning("HAR auto-probe on start failed: %s", exc)

        threading.Thread(target=_auto_har_probe, daemon=True).start()

    yield
    if har_live is not None:
        har_live.stop_all()
    if webcam is not None:
        webcam.stop()


app = FastAPI(
    title="VisionOps Backend",
    description="Webcam MJPEG + SFace + DINO/V-JEPA vision probes.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(faces.router)
app.include_router(vision.router)
app.include_router(har.router)

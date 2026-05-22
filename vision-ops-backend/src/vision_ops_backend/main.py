"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vision_ops_backend.config import settings
from vision_ops_backend.face.sface_live import SFaceLiveEngine
from vision_ops_backend.routers import cameras, faces, health
from vision_ops_backend.webcam import WebcamCapture


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
    webcam.stop()


app = FastAPI(
    title="VisionOps Backend",
    description="Webcam MJPEG + SFace live face recognition.",
    version="0.2.0",
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

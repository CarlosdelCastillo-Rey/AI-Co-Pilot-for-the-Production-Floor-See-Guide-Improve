"""Thread-safe webcam capture, optional SFace overlay, MJPEG streaming."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Generator

import cv2
from vision_ops_backend.features import LiveFeatureProcessor
import numpy as np

from vision_ops_backend.config import settings
from vision_ops_backend.face.sface_live import SFaceLiveEngine

logger = logging.getLogger(__name__)


class WebcamCapture:
    """Background reader with latest annotated JPEG and overlay metadata."""

    def __init__(
        self,
        camera_index: int = 0,
        jpeg_quality: int = 85,
        face_engine: SFaceLiveEngine | None = None,
    ) -> None:
        self._camera_index = camera_index
        self._jpeg_quality = jpeg_quality
        self._face_engine = face_engine
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._overlays: list[dict] = []
        self._frame_i = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._error: str | None = None
        self._live_processor = LiveFeatureProcessor()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def face_engine(self) -> SFaceLiveEngine | None:
        return self._face_engine

    def get_overlays(self) -> list[dict]:
        with self._lock:
            return list(self._overlays)

    def get_latest_frame(self) -> np.ndarray | None:
        """Decode latest JPEG for enrollment (best-effort)."""
        with self._lock:
            if not self._latest_jpeg:
                return None
            arr = np.frombuffer(self._latest_jpeg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _open_capture(self, index: int) -> cv2.VideoCapture | None:
        backends: list[int | None] = []
        if sys.platform == "darwin":
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.append(None)

        for backend in backends:
            cap = (
                cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            )
            if cap.isOpened():
                logger.info("Webcam opened: index=%s backend=%s", index, backend)
                return cap
            cap.release()
        return None

    def start(self) -> None:
        if self._running:
            return

        indices = [self._camera_index]
        if self._camera_index not in (1, 2):
            indices.extend((1, 2))

        for index in indices:
            cap = self._open_capture(index)
            if cap is not None:
                self._capture = cap
                self._camera_index = index
                self._running = True
                self._error = None
                self._thread = threading.Thread(
                    target=self._loop, name="webcam-capture", daemon=True
                )
                self._thread.start()
                return

        self._error = (
            f"Cannot open webcam (tried indices {indices}). "
            "Grant camera access to Terminal/Cursor in System Settings → Privacy → Camera."
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _loop(self) -> None:
        every_n = max(1, settings.face_detect_every_n_frames)
        while self._running and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                self._error = "Webcam read failed"
                time.sleep(0.1)
                continue

            overlays = self._overlays
            if self._face_engine and self._face_engine.is_ready:
                run_detection = self._frame_i % every_n == 0
                frame, overlays = self._face_engine.process(frame, run_detection=run_detection)

            if overlays and isinstance(overlays, list):
                for idx, face in enumerate(overlays):
                    box = face.get("box")
                    if box and len(box) == 4:
                        x, y, w, h = box
                        ltrb = [float(x), float(y), float(x + w), float(y + h)]

                        label = face.get("label", "Unknown")
                        subject_id = f"{label}_{idx}" if label == "Unknown" else label

                        live_features = self._live_processor.process_frame_track(
                            track_id=subject_id, bbox=ltrb, current_fps=12.0
                        )

                        face["telemetry"] = live_features
                        print(f"Face {subject_id} telemetry: {live_features}")

            ok_encode, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            )
            if ok_encode:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._overlays = overlays
            self._frame_i += 1
            time.sleep(0.001)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg


def mjpeg_generator(capture: WebcamCapture, fps: int = 12) -> Generator[bytes, None, None]:
    boundary = b"frame"
    interval = 1.0 / max(fps, 1)
    while capture.is_running:
        jpeg = capture.get_jpeg()
        if jpeg:
            yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(interval)

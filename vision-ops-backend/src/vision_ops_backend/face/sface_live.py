"""YuNet detection + SFace recognition for live webcam frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision_ops_backend.face.paths import (
    enrollment_preview_path,
    models_ready,
    owner_feature_path,
    sface_model_path,
    yunet_model_path,
)

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float
    is_owner: bool


class SFaceLiveEngine:
    """Detect faces and compare to enrolled owner embedding."""

    def __init__(
        self,
        owner_name: str = "You",
        match_threshold: float = 0.5,
        detect_threshold: float = 0.85,
    ) -> None:
        self.owner_name = owner_name
        self.match_threshold = match_threshold
        self.detect_threshold = detect_threshold
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: cv2.FaceRecognizerSF | None = None
        self._owner_feature: np.ndarray | None = None
        self._error: str | None = None
        self._frame_size: tuple[int, int] | None = None
        self._last_detections: list[FaceDetection] = []
        self._miss_frames = 0
        self._max_miss_frames = 8  # keep boxes briefly if detector skips a frame

        if not models_ready():
            self._error = (
                "Face models missing. Run: ./models/install_face_models.sh from repo root."
            )
            return

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(yunet_model_path()),
                "",
                (320, 320),
                self.detect_threshold,
                0.3,
                5000,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(str(sface_model_path()), "")
            self._load_owner()
            logger.info("SFace live engine ready (enrolled=%s)", self.is_enrolled)
        except cv2.error as exc:
            self._error = f"Failed to load face models: {exc}"

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_ready(self) -> bool:
        return self._detector is not None and self._recognizer is not None and self._error is None

    @property
    def is_enrolled(self) -> bool:
        return self._owner_feature is not None

    def _load_owner(self) -> None:
        path = owner_feature_path()
        if not path.is_file():
            return
        data = np.load(path)
        self._owner_feature = data["feature"]
        if "name" in data:
            self.owner_name = str(data["name"])

    def clear_enrollment(self) -> None:
        self._owner_feature = None
        self._last_detections = []
        self._miss_frames = 0

    def save_owner(self, feature: np.ndarray, name: str | None = None) -> None:
        path = owner_feature_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        label = name or self.owner_name
        np.savez(path, feature=feature, name=label)
        self._owner_feature = feature
        self.owner_name = label
        logger.info("Owner face enrolled → %s", path)

    def enroll_from_frames(self, frames: list[np.ndarray], name: str | None = None) -> dict:
        if not self.is_ready:
            return {"ok": False, "error": self._error}

        features: list[np.ndarray] = []
        for frame in frames:
            faces = self._detect_faces(frame)
            if len(faces) == 1:
                features.append(self._feature_for_face(frame, faces[0]))

        if len(features) < 3:
            return {
                "ok": False,
                "error": f"Only {len(features)} valid samples; need at least 3 with a single face.",
            }

        mean_feat = np.mean(np.stack(features, axis=0), axis=0)
        self.save_owner(mean_feat, name=name)
        self._save_enrollment_preview(frames[len(frames) // 2])
        return {"ok": True, "samples": len(features), "name": self.owner_name}

    def _save_enrollment_preview(self, frame: np.ndarray) -> None:
        """Optional reference crop saved locally (not used for matching)."""
        path = enrollment_preview_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        faces = self._detect_faces(frame)
        img = frame
        if len(faces) == 1:
            x, y, w, h = faces[0][:4].astype(int)
            pad = 20
            img = frame[
                max(0, y - pad) : y + h + pad,
                max(0, x - pad) : x + w + pad,
            ]
        cv2.imwrite(str(path), img)

    def _detect_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        assert self._detector is not None
        h, w = frame.shape[:2]
        if self._frame_size != (w, h):
            self._detector.setInputSize((w, h))
            self._frame_size = (w, h)

        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        return [faces[i] for i in range(faces.shape[0])]

    def _feature_for_face(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        assert self._recognizer is not None
        aligned = self._recognizer.alignCrop(frame, face)
        return self._recognizer.feature(aligned)

    def _label_face(self, frame: np.ndarray, face: np.ndarray) -> FaceDetection:
        x, y, w, h = face[:4].astype(int)
        det_score = float(face[-1]) if face.shape[0] > 14 else 1.0
        label = "Unknown"
        is_owner = False
        match_score = 0.0

        if self.is_enrolled and self._recognizer is not None and self._owner_feature is not None:
            feat = self._feature_for_face(frame, face)
            match_score = float(
                self._recognizer.match(
                    self._owner_feature,
                    feat,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
            if match_score >= self.match_threshold:
                label = self.owner_name
                is_owner = True
            else:
                label = f"Unknown ({match_score:.2f})"
        else:
            label = "Face (enroll me)"

        return FaceDetection(
            x=int(x),
            y=int(y),
            w=int(w),
            h=int(h),
            label=label,
            score=match_score if is_owner else det_score,
            is_owner=is_owner,
        )

    def _smooth_detections(
        self, previous: list[FaceDetection], current: list[FaceDetection]
    ) -> list[FaceDetection]:
        """Ease box movement between detection cycles (single-face webcam)."""
        if len(previous) != 1 or len(current) != 1:
            return current
        old, new = previous[0], current[0]
        alpha = 0.55
        return [
            FaceDetection(
                x=int(old.x * (1 - alpha) + new.x * alpha),
                y=int(old.y * (1 - alpha) + new.y * alpha),
                w=int(old.w * (1 - alpha) + new.w * alpha),
                h=int(old.h * (1 - alpha) + new.h * alpha),
                label=new.label,
                score=new.score,
                is_owner=new.is_owner,
            )
        ]

    def _update_detections(self, frame: np.ndarray) -> None:
        """Run YuNet + SFace (expensive); call every N frames only."""
        fresh: list[FaceDetection] = []
        for face in self._detect_faces(frame):
            fresh.append(self._label_face(frame, face))

        if fresh:
            self._last_detections = self._smooth_detections(self._last_detections, fresh)
            self._miss_frames = 0
        else:
            self._miss_frames += 1
            if self._miss_frames > self._max_miss_frames:
                self._last_detections = []

    def _draw_detections(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> tuple[np.ndarray, list[dict]]:
        """Draw cached boxes on every frame so overlays do not blink."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        overlays: list[dict] = []

        for det in detections:
            color = (76, 175, 80) if det.is_owner else (200, 200, 200)
            cv2.rectangle(annotated, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
            cv2.putText(
                annotated,
                det.label,
                (det.x, max(det.y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            overlays.append(
                {
                    "type": "person",
                    "label": det.label,
                    "top": f"{det.y / h * 100:.1f}%",
                    "left": f"{det.x / w * 100:.1f}%",
                    "width": f"{det.w / w * 100:.1f}%",
                    "height": f"{det.h / h * 100:.1f}%",
                    "variant": "primary" if det.is_owner else "tertiary",
                }
            )

        return annotated, overlays

    def process(self, frame: np.ndarray, *, run_detection: bool = True) -> tuple[np.ndarray, list[dict]]:
        """Update detections optionally, always draw last known boxes."""
        if not self.is_ready:
            return frame, []

        if run_detection:
            self._update_detections(frame)

        return self._draw_detections(frame, self._last_detections)

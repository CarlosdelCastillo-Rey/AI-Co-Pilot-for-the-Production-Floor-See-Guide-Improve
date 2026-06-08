"""Per-person HAR: ByteTrack + crop buffers + per-track inference (NB14-style)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_yolo_track_model: Any = None
_yolo_track_failed = False

CROP_SIZE = 224
DEFAULT_BBOX_PADDING = 0.15
DEFAULT_STALE_SEC = 2.0


@dataclass
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    det_conf: float
    action_label: str | None = None
    action_confidence: float | None = None
    inferring: bool = False
    n_frames: int = 0


@dataclass
class _TrackState:
    track_id: int
    crop_buffer: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=32))
    last_seen: float = field(default_factory=time.monotonic)
    display_label: str | None = None
    display_conf: float = 0.0
    candidate_label: str | None = None
    candidate_count: int = 0
    inferring: bool = False
    n_frames: int = 0
    last_bbox: tuple[int, int, int, int, float] = (0, 0, 0, 0, 0.0)


def _yolo_weights_path() -> str:
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / "yolov8n.pt",
        here.parents[4] / "vision-ops-backend" / "yolov8n.pt",
        Path("yolov8n.pt"),
    ):
        if candidate.is_file():
            return str(candidate)
    return "yolov8n.pt"


def _get_yolo_track_model() -> Any | None:
    global _yolo_track_model, _yolo_track_failed
    if _yolo_track_failed:
        return None
    if _yolo_track_model is None:
        try:
            from ultralytics import YOLO

            _yolo_track_model = YOLO(_yolo_weights_path())
        except Exception as exc:
            logger.info("YOLO track model unavailable: %s", exc)
            _yolo_track_failed = True
            return None
    return _yolo_track_model


def crop_person(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, padding: float) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    pw, ph = bw * padding, bh * padding
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


class PerPersonHarEngine:
    """Maintains track buffers and smoothed per-person action labels."""

    def __init__(
        self,
        *,
        buffer_frames: int = 32,
        bbox_padding: float = DEFAULT_BBOX_PADDING,
        dwell_windows: int = 2,
        stale_sec: float = DEFAULT_STALE_SEC,
    ) -> None:
        self.buffer_frames = max(8, buffer_frames)
        self.bbox_padding = bbox_padding
        self.dwell_windows = max(1, dwell_windows)
        self.stale_sec = stale_sec
        self._tracks: dict[int, _TrackState] = {}
        self._last_boxes: list[tuple[int, int, int, int, float, int]] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._last_boxes.clear()

    def reconfigure(
        self,
        *,
        buffer_frames: int | None = None,
        bbox_padding: float | None = None,
        dwell_windows: int | None = None,
    ) -> None:
        if buffer_frames is not None:
            self.buffer_frames = max(8, buffer_frames)
            for st in self._tracks.values():
                st.crop_buffer = deque(list(st.crop_buffer)[-self.buffer_frames :], maxlen=self.buffer_frames)
        if bbox_padding is not None:
            self.bbox_padding = bbox_padding
        if dwell_windows is not None:
            self.dwell_windows = max(1, dwell_windows)

    def update_frame(self, frame_bgr: np.ndarray) -> list[TrackedPerson]:
        model = _get_yolo_track_model()
        now = time.monotonic()
        detections: list[tuple[int, int, int, int, float, int]] = []

        if model is not None:
            try:
                res = model.track(frame_bgr, persist=True, classes=[0], verbose=False)[0]
                if res.boxes is not None:
                    for box in res.boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        detections.append((x1, y1, x2, y2, conf, tid))
            except Exception as exc:
                logger.debug("YOLO track failed: %s", exc)

        seen: set[int] = set()
        for x1, y1, x2, y2, conf, tid in detections:
            seen.add(tid)
            crop = crop_person(frame_bgr, x1, y1, x2, y2, padding=self.bbox_padding)
            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    crop_buffer=deque(maxlen=self.buffer_frames),
                )
            st = self._tracks[tid]
            st.last_seen = now
            st.last_bbox = (x1, y1, x2, y2, conf)
            st.n_frames += 1
            if crop is not None:
                st.crop_buffer.append(crop)

        stale = [tid for tid, st in self._tracks.items() if tid not in seen and now - st.last_seen > self.stale_sec]
        for tid in stale:
            del self._tracks[tid]

        self._last_boxes = detections
        return self.get_tracked_persons()

    def get_tracked_persons(self) -> list[TrackedPerson]:
        out: list[TrackedPerson] = []
        for tid, st in sorted(self._tracks.items()):
            x1, y1, x2, y2, conf = st.last_bbox
            out.append(
                TrackedPerson(
                    track_id=tid,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    det_conf=conf,
                    action_label=st.display_label,
                    action_confidence=st.display_conf if st.display_label else None,
                    inferring=st.inferring,
                    n_frames=st.n_frames,
                )
            )
        return out

    def min_buffer_for_infer(self) -> int:
        return max(8, self.buffer_frames // 2)

    def tracks_ready_for_infer(self) -> list[int]:
        min_buf = self.min_buffer_for_infer()
        return [
            tid
            for tid, st in self._tracks.items()
            if len(st.crop_buffer) >= min_buf and not st.inferring
        ]

    def get_crop_frames(self, track_id: int) -> list[np.ndarray]:
        st = self._tracks.get(track_id)
        if st is None:
            return []
        return list(st.crop_buffer)

    def set_track_inferring(self, track_id: int, inferring: bool) -> None:
        st = self._tracks.get(track_id)
        if st:
            st.inferring = inferring

    def apply_track_prediction(self, track_id: int, prediction: dict[str, Any]) -> None:
        st = self._tracks.get(track_id)
        if st is None:
            return
        label = str(prediction.get("label", ""))
        conf = float(prediction.get("confidence") or 0.0)
        if not label:
            return
        if st.display_label is None:
            st.display_label = label
            st.display_conf = conf
            st.candidate_label = None
            st.candidate_count = 0
            return
        if label == st.display_label:
            st.display_conf = conf
            st.candidate_label = None
            st.candidate_count = 0
            return
        if label == st.candidate_label:
            st.candidate_count += 1
        else:
            st.candidate_label = label
            st.candidate_count = 1
        if st.candidate_count >= self.dwell_windows:
            st.display_label = label
            st.display_conf = conf
            st.candidate_label = None
            st.candidate_count = 0

    def summary_prediction(self) -> dict[str, Any] | None:
        """Scene-level summary: highest-confidence track (for legacy UI)."""
        best: TrackedPerson | None = None
        for person in self.get_tracked_persons():
            if person.action_label is None or person.action_confidence is None:
                continue
            if best is None or person.action_confidence > (best.action_confidence or 0):
                best = person
        if best is None:
            return None
        return {
            "label": best.action_label,
            "confidence": best.action_confidence,
            "track_id": best.track_id,
        }

    def track_predictions_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": p.track_id,
                "bbox": [p.x1, p.y1, p.x2, p.y2],
                "det_conf": round(p.det_conf, 4),
                "action_label": p.action_label,
                "action_confidence": round(p.action_confidence, 4) if p.action_confidence is not None else None,
                "inferring": p.inferring,
                "n_frames": p.n_frames,
            }
            for p in self.get_tracked_persons()
        ]

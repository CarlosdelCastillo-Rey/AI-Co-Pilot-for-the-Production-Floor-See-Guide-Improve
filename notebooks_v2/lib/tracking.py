"""ByteTrack + per-person crop buffers + action memory (NB14-style)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lib.constants import (
    CROP_SIZE,
    DEFAULT_BBOX_PADDING,
    DEFAULT_BUFFER_FRAMES,
    DEFAULT_DWELL_WINDOWS,
    DEFAULT_STALE_SEC,
)
from lib.paths import yolo_weights_path

_yolo_model: Any = None
_yolo_failed = False


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
    crop_buffer: deque = field(default_factory=deque)
    last_seen: float = field(default_factory=time.monotonic)
    display_label: str | None = None
    display_conf: float = 0.0
    candidate_label: str | None = None
    candidate_count: int = 0
    inferring: bool = False
    n_frames: int = 0
    last_bbox: tuple[int, int, int, int, float] = (0, 0, 0, 0, 0.0)
    history: list[dict[str, Any]] = field(default_factory=list)


def _get_yolo():
    global _yolo_model, _yolo_failed
    if _yolo_failed:
        return None
    if _yolo_model is None:
        try:
            from ultralytics import YOLO

            w = yolo_weights_path()
            _yolo_model = YOLO(str(w) if w.is_file() else "yolov8n.pt")
        except Exception:
            _yolo_failed = True
            return None
    return _yolo_model


def crop_person(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    padding: float = DEFAULT_BBOX_PADDING,
) -> np.ndarray | None:
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


class PerPersonTracker:
    """One HAR stream per track_id — never shares labels across people."""

    def __init__(
        self,
        *,
        buffer_frames: int = DEFAULT_BUFFER_FRAMES,
        bbox_padding: float = DEFAULT_BBOX_PADDING,
        dwell_windows: int = DEFAULT_DWELL_WINDOWS,
        stale_sec: float = DEFAULT_STALE_SEC,
    ) -> None:
        self.buffer_frames = max(8, buffer_frames)
        self.bbox_padding = bbox_padding
        self.dwell_windows = max(1, dwell_windows)
        self.stale_sec = stale_sec
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update_frame(self, frame_bgr: np.ndarray) -> list[TrackedPerson]:
        model = _get_yolo()
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
            except Exception:
                pass

        seen: set[int] = set()
        for x1, y1, x2, y2, conf, tid in detections:
            seen.add(tid)
            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    crop_buffer=deque(maxlen=self.buffer_frames),
                )
            st = self._tracks[tid]
            st.last_seen = now
            st.last_bbox = (x1, y1, x2, y2, conf)
            st.n_frames += 1
            crop = crop_person(frame_bgr, x1, y1, x2, y2, padding=self.bbox_padding)
            if crop is not None:
                st.crop_buffer.append(crop)

        for tid in list(self._tracks):
            if tid not in seen and now - self._tracks[tid].last_seen > self.stale_sec:
                del self._tracks[tid]

        return self.list_persons()

    def list_persons(self) -> list[TrackedPerson]:
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

    def min_buffer(self) -> int:
        return max(8, self.buffer_frames // 2)

    def ready_track_ids(self) -> list[int]:
        m = self.min_buffer()
        return [tid for tid, st in self._tracks.items() if len(st.crop_buffer) >= m and not st.inferring]

    def get_crops(self, track_id: int) -> list[np.ndarray]:
        st = self._tracks.get(track_id)
        return list(st.crop_buffer) if st else []

    def set_inferring(self, track_id: int, v: bool) -> None:
        st = self._tracks.get(track_id)
        if st:
            st.inferring = v

    def apply_prediction(self, track_id: int, prediction: dict[str, Any]) -> dict[str, Any]:
        st = self._tracks.get(track_id)
        if st is None:
            return {"label_changed": False, "display_label": None, "display_conf": 0.0}
        label = str(prediction.get("label") or "")
        conf = float(prediction.get("confidence") or 0.0)
        if not label:
            return {"label_changed": False, "display_label": st.display_label, "display_conf": st.display_conf}

        prev_label = st.display_label

        if st.display_label is None:
            st.display_label = label
            st.display_conf = conf
        elif label == st.display_label:
            st.display_conf = conf
            st.candidate_label = None
            st.candidate_count = 0
        elif label == st.candidate_label:
            st.candidate_count += 1
            if st.candidate_count >= self.dwell_windows:
                st.display_label = label
                st.display_conf = conf
                st.candidate_label = None
                st.candidate_count = 0
        else:
            st.candidate_label = label
            st.candidate_count = 1

        st.history.append({"label": label, "confidence": conf, "t": time.time()})
        label_changed = st.display_label != prev_label
        return {
            "label_changed": label_changed,
            "display_label": st.display_label,
            "display_conf": st.display_conf,
        }

    def to_payload(self) -> list[dict[str, Any]]:
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
            for p in self.list_persons()
        ]

    def get_bbox(self, track_id: int) -> list[int]:
        st = self._tracks.get(track_id)
        if st is None:
            return [0, 0, 0, 0]
        x1, y1, x2, y2, _ = st.last_bbox
        return [x1, y1, x2, y2]

    def track_histories(self) -> dict[int, list[dict[str, Any]]]:
        return {tid: list(st.history) for tid, st in self._tracks.items()}

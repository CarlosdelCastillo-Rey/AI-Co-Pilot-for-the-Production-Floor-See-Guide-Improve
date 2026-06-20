"""Loop mock MP4s with sliding-window HAR inference (har-research live demo)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import HAR_MODELS, is_har_camera, spec_for_camera
from vision_ops_backend.vision.har.har_activity_client import (
    build_activity_entry,
    format_detections,
    format_tracked_detections,
    ingest_har_activity,
)
from vision_ops_backend.vision.har.har_v2_session_client import HarV2SessionClient, new_session_id
from vision_ops_backend.vision.har.person_name_lookup import hydrate_track_names_from_registry
from vision_ops_backend.vision.har.track_registry import absorb_v2_event, enrich_track_predictions
from vision_ops_backend.vision.har.overlay import (
    compose_live_frame,
    compose_per_person_live_frame,
    detect_person_boxes,
)
from vision_ops_backend.vision.har.per_person import PerPersonHarEngine
from vision_ops_backend.vision.har.probe_runner import build_activity_overlay
from vision_ops_backend.vision.har.trigger_snapshot import capture_har_trigger_snapshot
from vision_ops_backend.vision.mock_videos import assign_videos_to_har_cameras, public_url_for_video
from vision_ops_backend.vision.store import save_probe_result

logger = logging.getLogger(__name__)

# Only one HAR forward pass at a time (GPU + transformers load safety).
_har_infer_lock = threading.Lock()


class HarLiveStream:
    """Read one mock video in a loop; run HAR inference every N frames."""

    def __init__(self, camera_id: str, model_id: str, video_path: Path) -> None:
        self.camera_id = camera_id
        self.model_id = model_id
        self.video_path = video_path
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._infer_running = False  # used by non-per-person path only
        self._frame_idx = 0
        self._latest_frame_bgr: np.ndarray | None = None
        self._track_threads: dict[int, threading.Thread] = {}
        self._track_stop: dict[int, threading.Event] = {}
        self._playback_active = False
        self._session_id = f"har-sess-{uuid.uuid4().hex[:12]}"
        self._state: dict[str, Any] = {
            "camera_id": camera_id,
            "model_id": model_id,
            "video": video_path.name,
            "videoUrl": public_url_for_video(video_path),
            "inferring": False,
            "prediction": None,
            "error": None,
            "playback_active": False,
            "session_id": self._session_id,
            "per_person_mode": settings.har_live_per_person_mode,
            "track_predictions": [],
            "person_count": 0,
            "latest_snapshot_url": None,
        }
        spec = spec_for_camera(camera_id)
        self._model_label = spec.label if spec else model_id
        self._per_person_engine = PerPersonHarEngine(
            buffer_frames=settings.har_live_buffer_frames,
            dwell_windows=2,
        )
        self._v2_logger: HarV2SessionClient | None = None
        self._track_global: dict[int, str] = {}
        self._track_names: dict[int, str] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _open_v2_session(self) -> None:
        if not settings.har_v2_session_enabled:
            self._v2_logger = None
            return
        self._session_id = new_session_id()
        with self._lock:
            self._state["session_id"] = self._session_id
        self._v2_logger = HarV2SessionClient(
            session_id=self._session_id,
            source="live",
            camera_id=self.camera_id,
            video_name=self.video_path.name,
            model_id=self.model_id,
            model_tag=self.model_id,
        )

    def _close_v2_session(self) -> None:
        if self._v2_logger is not None:
            self._v2_logger.finalize()
            self._v2_logger = None

    def set_playback_active(self, active: bool) -> None:
        with self._lock:
            was_active = self._playback_active
            self._playback_active = active
            self._state["playback_active"] = active
            if not active:
                self._state["inferring"] = False
        if active and not was_active:
            self._per_person_engine.reset()
            self._track_global.clear()
            self._track_names.clear()
            self._open_v2_session()
        elif not active and was_active:
            self._stop_track_threads()
            self._close_v2_session()

    def is_playback_active(self) -> bool:
        with self._lock:
            return self._playback_active

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"har-live-{self.camera_id}")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _set_state(self, **kwargs: Any) -> None:
        with self._lock:
            self._state.update(kwargs)

    def _loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        buf: deque[np.ndarray] = deque(maxlen=settings.har_live_buffer_frames)
        buf_rgb: deque[np.ndarray] = deque(maxlen=settings.har_live_buffer_frames)
        infer_every = max(1, settings.har_live_infer_every)
        min_buf = max(8, settings.har_live_buffer_frames // 2)
        per_person = settings.har_live_per_person_mode

        while self._running:
            if not self.is_playback_active():
                time.sleep(0.08)
                continue

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(str(self.video_path))
                if not cap.isOpened():
                    self._set_state(error=f"cannot open {self.video_path.name}")
                    time.sleep(2.0)
                    continue
                self._set_state(error=None)

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                buf.clear()
                buf_rgb.clear()
                self._stop_track_threads()
                self._per_person_engine.reset()
                self._track_global.clear()
                self._track_names.clear()
                with self._lock:
                    self._state["track_predictions"] = []
                if settings.har_v2_session_enabled:
                    self._close_v2_session()
                    self._open_v2_session()
                else:
                    with self._lock:
                        self._session_id = f"har-sess-{uuid.uuid4().hex[:12]}"
                        self._state["session_id"] = self._session_id
                continue

            self._frame_idx += 1
            buf.append(frame.copy())
            buf_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with self._lock:
                pred = self._state.get("prediction")
                inferring = self._state.get("inferring", False)
                track_preds = list(self._state.get("track_predictions") or [])

            if per_person:
                persons = self._per_person_engine.update_frame(frame)
                self._latest_frame_bgr = frame
                hydrate_track_names_from_registry(self._track_global, self._track_names)
                track_preds = enrich_track_predictions(
                    self._per_person_engine.track_predictions_payload(),
                    self._track_global,
                    self._track_names,
                )
                summary = self._per_person_engine.summary_prediction()
                rendered = compose_per_person_live_frame(
                    frame,
                    frames_rgb=list(buf_rgb),
                    model_label=self._model_label,
                    track_predictions=track_preds,
                    model_id=self.model_id,
                    inferring=inferring,
                    show_heatmap=settings.har_live_show_heatmap,
                    show_boxes=settings.har_live_show_yolo_boxes,
                    summary_prediction=summary,
                )
                with self._lock:
                    self._state["track_predictions"] = track_preds
                    self._state["person_count"] = len(track_preds)
                    if summary:
                        self._state["prediction"] = summary
                # Start an inference thread for each new active track
                active_tids = {p.track_id for p in persons}
                for tid in active_tids:
                    if tid not in self._track_threads or not self._track_threads[tid].is_alive():
                        stop_ev = threading.Event()
                        self._track_stop[tid] = stop_ev
                        t = threading.Thread(
                            target=self._run_track_loop,
                            args=(tid, stop_ev),
                            daemon=True,
                            name=f"har-track-{self.camera_id}-{tid}",
                        )
                        self._track_threads[tid] = t
                        t.start()
                # Signal threads for tracks that disappeared
                for tid in list(self._track_threads):
                    if tid not in active_tids:
                        ev = self._track_stop.pop(tid, None)
                        if ev:
                            ev.set()
                        self._track_threads.pop(tid)
            else:
                rendered = compose_live_frame(
                    frame,
                    frames_rgb=list(buf_rgb),
                    model_label=self._model_label,
                    prediction=pred,
                    model_id=self.model_id,
                    inferring=inferring,
                    show_heatmap=settings.har_live_show_heatmap,
                    show_boxes=settings.har_live_show_yolo_boxes,
                )

            ok_enc, jpeg_buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok_enc:
                with self._lock:
                    self._latest_jpeg = jpeg_buf.tobytes()

            if not per_person:
                should_infer = (
                    self.is_playback_active()
                    and self._frame_idx % infer_every == 0
                    and not self._infer_running
                    and len(buf) >= min_buf
                )
                if should_infer:
                    frames_copy = list(buf)
                    threading.Thread(target=self._run_inference, args=(frames_copy,), daemon=True).start()

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            time.sleep(1.0 / max(native_fps, settings.har_live_stream_fps))

        if cap is not None:
            cap.release()

    def capture_current_snapshot(self) -> dict[str, Any]:
        """Annotate the current frame with all track predictions and return snapshot URL + results."""
        frame = self._latest_frame_bgr
        if frame is None:
            return {"error": "No frame available yet — start playback first"}

        hydrate_track_names_from_registry(self._track_global, self._track_names)
        track_payload = enrich_track_predictions(
            self._per_person_engine.track_predictions_payload(),
            self._track_global,
            self._track_names,
        )
        summary = self._per_person_engine.summary_prediction()

        prediction = {
            "label": summary.get("label") if summary else "—",
            "confidence": summary.get("confidence") if summary else 0.0,
            "track_id": summary.get("track_id") if summary else None,
            "top_k": [],
        }
        snapshot_url = capture_har_trigger_snapshot(
            [frame],
            model_id=self.model_id,
            model_label=self._model_label,
            prediction=prediction,
            show_heatmap=settings.har_live_show_heatmap,
            show_boxes=True,
            track_predictions=track_payload,
        )
        return {
            "snapshot_url": snapshot_url,
            "track_count": len(track_payload),
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "display_name": t.get("display_name"),
                    "action_label": t.get("action_label"),
                    "action_confidence": t.get("action_confidence"),
                    "inferring": t.get("inferring", False),
                }
                for t in track_payload
            ],
        }

    def _stop_track_threads(self) -> None:
        for ev in self._track_stop.values():
            ev.set()
        self._track_threads.clear()
        self._track_stop.clear()

    def _run_track_loop(self, tid: int, stop_event: threading.Event) -> None:
        """Continuous inference loop for a single tracked person."""
        from vision_ops_backend.vision.har.inference import run_har_inference

        min_buf = self._per_person_engine.min_buffer_for_infer()
        exclude = settings.har_exclude_label_list or None

        while not stop_event.is_set() and self._running:
            if not self.is_playback_active():
                time.sleep(0.1)
                continue
            crops = self._per_person_engine.get_crop_frames(tid)
            if len(crops) < min_buf:
                time.sleep(0.05)
                continue
            crops_snapshot = list(crops)
            self._per_person_engine.set_track_inferring(tid, True)
            t0 = time.perf_counter()
            try:
                with _har_infer_lock:
                    if stop_event.is_set():
                        return
                    infer = run_har_inference(
                        self.model_id,
                        crops_snapshot,
                        exclude_labels=exclude,
                        return_embedding=bool(self._v2_logger),
                        min_confidence=settings.har_v2_min_confidence,
                    )
                pred = infer["prediction"]
                label_changed = self._per_person_engine.apply_track_prediction(tid, pred)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                if self._v2_logger:
                    st = self._per_person_engine._tracks.get(tid)
                    bbox = list(st.last_bbox[:4]) if st else None
                    crop_jpeg = None
                    ok_j, jbuf = cv2.imencode(".jpg", crops_snapshot[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok_j:
                        crop_jpeg = jbuf.tobytes()
                    row = self._v2_logger.log_inference(
                        track_id=tid,
                        frame_idx=self._frame_idx,
                        bbox=bbox,
                        prediction=pred,
                        label_changed=label_changed,
                        uncertain=bool(pred.get("uncertain")),
                        infer_ms=infer_ms,
                        crop_jpeg=crop_jpeg,
                        embedding=infer.get("embedding"),
                    )
                    absorb_v2_event(self._track_global, self._track_names, tid, row)

                hydrate_track_names_from_registry(self._track_global, self._track_names)
                track_payload = enrich_track_predictions(
                    self._per_person_engine.track_predictions_payload(),
                    self._track_global,
                    self._track_names,
                )
                summary = self._per_person_engine.summary_prediction()

                snapshot_url: str | None = None
                if label_changed and summary:
                    frame_snap = self._latest_frame_bgr
                    if frame_snap is not None:
                        snapshot_url = capture_har_trigger_snapshot(
                            [frame_snap],
                            model_id=self.model_id,
                            model_label=self._model_label,
                            prediction={
                                "label": summary.get("label"),
                                "confidence": summary.get("confidence"),
                                "track_id": summary.get("track_id"),
                                "top_k": [],
                            },
                            show_heatmap=settings.har_live_show_heatmap,
                            show_boxes=settings.har_live_show_yolo_boxes,
                            track_predictions=track_payload,
                        )
                    if settings.har_activity_ingest_enabled:
                        clip_url = public_url_for_video(self.video_path)
                        detections = format_tracked_detections(track_payload)
                        ingest_har_activity(
                            build_activity_entry(
                                camera_id=self.camera_id,
                                model_id=self.model_id,
                                prediction={
                                    "label": summary.get("label"),
                                    "confidence": summary.get("confidence"),
                                    "track_id": summary.get("track_id"),
                                },
                                source="live:per_person",
                                detections=detections,
                                frame_index=self._frame_idx,
                                video_name=self.video_path.name,
                                clip_url=clip_url,
                                session_id=self._session_id,
                                infer_ms=round(infer_ms, 1),
                                snapshot_url=snapshot_url,
                                model_label=self._model_label,
                                hyperparams={
                                    "infer_every": settings.har_live_infer_every,
                                    "per_person_mode": True,
                                    "buffer_frames": settings.har_live_buffer_frames,
                                },
                            )
                        )

                state_update: dict[str, Any] = {
                    "inferring": False,
                    "prediction": summary,
                    "track_predictions": track_payload,
                    "person_count": len(track_payload),
                    "error": None,
                }
                if snapshot_url:
                    state_update["latest_snapshot_url"] = snapshot_url
                self._set_state(**state_update)
            except Exception as exc:
                logger.warning("HAR track %d %s: %s", tid, self.camera_id, exc)
            finally:
                self._per_person_engine.set_track_inferring(tid, False)
                # Yield briefly so the event loop and other threads can breathe
                time.sleep(0.05)

    def _run_per_person_inference(self, track_ids: list[int], frame_bgr: np.ndarray) -> None:
        if not self.is_playback_active() or not track_ids:
            return
        self._infer_running = True
        self._set_state(inferring=True)
        t0 = time.perf_counter()
        try:
            from vision_ops_backend.vision.har.inference import run_har_inference

            exclude = settings.har_exclude_label_list or None
            for tid in track_ids:
                if not self.is_playback_active():
                    return
                crops = self._per_person_engine.get_crop_frames(tid)
                if len(crops) < self._per_person_engine.min_buffer_for_infer():
                    continue
                self._per_person_engine.set_track_inferring(tid, True)
                try:
                    with _har_infer_lock:
                        infer = run_har_inference(
                            self.model_id,
                            crops,
                            exclude_labels=exclude,
                            return_embedding=bool(self._v2_logger),
                            min_confidence=settings.har_v2_min_confidence,
                        )
                    pred = infer["prediction"]
                    label_changed = self._per_person_engine.apply_track_prediction(tid, pred)
                    if self._v2_logger:
                        st = self._per_person_engine._tracks.get(tid)
                        bbox = list(st.last_bbox[:4]) if st else None
                        crop_jpeg = None
                        if crops:
                            ok_j, buf = cv2.imencode(".jpg", crops[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_j:
                                crop_jpeg = buf.tobytes()
                        row = self._v2_logger.log_inference(
                            track_id=tid,
                            frame_idx=self._frame_idx,
                            bbox=bbox,
                            prediction=pred,
                            label_changed=label_changed,
                            uncertain=bool(pred.get("uncertain")),
                            infer_ms=(time.perf_counter() - t0) * 1000,
                            crop_jpeg=crop_jpeg,
                            embedding=infer.get("embedding"),
                        )
                        absorb_v2_event(self._track_global, self._track_names, tid, row)
                finally:
                    self._per_person_engine.set_track_inferring(tid, False)

            hydrate_track_names_from_registry(self._track_global, self._track_names)
            track_payload = enrich_track_predictions(
                self._per_person_engine.track_predictions_payload(),
                self._track_global,
                self._track_names,
            )
            summary = self._per_person_engine.summary_prediction()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            clip_url = public_url_for_video(self.video_path)
            detections = format_tracked_detections(track_payload)
            snapshot_url = None
            if summary:
                snapshot_url = capture_har_trigger_snapshot(
                    [frame_bgr],
                    model_id=self.model_id,
                    model_label=self._model_label,
                    prediction={
                        "label": summary.get("label"),
                        "confidence": summary.get("confidence"),
                        "track_id": summary.get("track_id"),
                        "top_k": [],
                    },
                    show_heatmap=settings.har_live_show_heatmap,
                    show_boxes=settings.har_live_show_yolo_boxes,
                    track_predictions=track_payload,
                )
            if settings.har_activity_ingest_enabled and summary:
                ingest_har_activity(
                    build_activity_entry(
                        camera_id=self.camera_id,
                        model_id=self.model_id,
                        prediction={
                            "label": summary.get("label"),
                            "confidence": summary.get("confidence"),
                            "track_id": summary.get("track_id"),
                        },
                        source="live:per_person",
                        detections=detections,
                        frame_index=self._frame_idx,
                        video_name=self.video_path.name,
                        clip_url=clip_url,
                        session_id=self._session_id,
                        infer_ms=round(infer_ms, 1),
                        snapshot_url=snapshot_url,
                        model_label=self._model_label,
                        hyperparams={
                            "infer_every": settings.har_live_infer_every,
                            "per_person_mode": True,
                            "buffer_frames": settings.har_live_buffer_frames,
                        },
                    )
                )
            self._set_state(
                inferring=False,
                prediction=summary,
                track_predictions=track_payload,
                person_count=len(track_payload),
                error=None,
            )
        except Exception as exc:
            logger.warning("HAR live per-person %s: %s", self.camera_id, exc)
            self._set_state(inferring=False, error=str(exc))
        finally:
            self._infer_running = False

    def _run_inference(self, frames_bgr: list[np.ndarray]) -> None:
        if not self.is_playback_active():
            return
        self._infer_running = True
        self._set_state(inferring=True)
        t0 = time.perf_counter()
        try:
            if not self.is_playback_active():
                return
            from vision_ops_backend.vision.har.inference import run_har_inference

            with _har_infer_lock:
                infer = run_har_inference(
                    self.model_id,
                    frames_bgr,
                    exclude_labels=settings.har_exclude_label_list or None,
                )
            pred = infer["prediction"]
            overlay = build_activity_overlay(pred["label"], pred["confidence"])
            last_frame = frames_bgr[-1]
            boxes = detect_person_boxes(last_frame)
            detections = format_detections(
                boxes,
                action_label=pred.get("label"),
                action_confidence=pred.get("confidence"),
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            clip_url = public_url_for_video(self.video_path)
            snapshot_url = capture_har_trigger_snapshot(
                frames_bgr,
                model_id=self.model_id,
                model_label=self._model_label,
                prediction=pred,
                show_heatmap=settings.har_live_show_heatmap,
                show_boxes=settings.har_live_show_yolo_boxes,
            )
            with self._lock:
                session_id = self._session_id
                frame_idx = self._frame_idx
            payload = {
                "mode": "har_activity",
                "status": "ok",
                "model_id": self.model_id,
                "source": f"live:{self.video_path.name}",
                "shared_clip": False,
                "backend": infer["backend"],
                "device": infer.get("device"),
                "prediction": pred,
                "overlay": overlay,
                "videoUrl": clip_url,
            }
            save_probe_result(self.camera_id, payload)
            ingest_har_activity(
                build_activity_entry(
                    camera_id=self.camera_id,
                    model_id=self.model_id,
                    prediction=pred,
                    source="live",
                    backend=infer.get("backend"),
                    device=infer.get("device"),
                    detections=detections,
                    frame_index=frame_idx,
                    video_name=self.video_path.name,
                    clip_url=clip_url,
                    session_id=session_id,
                    infer_ms=round(infer_ms, 1),
                    snapshot_url=snapshot_url,
                    model_label=self._model_label,
                    hyperparams={
                        "infer_every": settings.har_live_infer_every,
                        "buffer_frames": settings.har_live_buffer_frames,
                        "stream_fps": settings.har_live_stream_fps,
                        "show_heatmap": settings.har_live_show_heatmap,
                        "show_yolo_boxes": settings.har_live_show_yolo_boxes,
                        "top_k": 3,
                        "ingest_logs": settings.har_activity_ingest_enabled,
                    },
                )
            )
            self._set_state(
                inferring=False,
                prediction=pred,
                backend=infer.get("backend"),
                device=infer.get("device"),
                error=None,
            )
        except Exception as exc:
            logger.warning("HAR live infer %s: %s", self.camera_id, exc)
            self._set_state(inferring=False, error=str(exc))
        finally:
            self._infer_running = False


class HarLiveManager:
    """One live stream per HAR mock camera."""

    def __init__(self) -> None:
        self._streams: dict[str, HarLiveStream] = {}
        self._lock = threading.Lock()

    def start_all(self) -> None:
        if not settings.har_enabled or not settings.har_live_enabled:
            return
        threading.Thread(target=self._start_all_worker, daemon=True, name="har-live-boot").start()

    def _start_all_worker(self) -> None:
        try:
            from vision_ops_backend.vision.har.extractors import preload_har_extractors

            logger.info("HAR live: preloading checkpoints and HF extractors…")
            preload_har_extractors()
            logger.info("HAR live: preload complete")
        except Exception as exc:
            logger.warning("HAR live preload failed (inference may retry): %s", exc)

        assignments = assign_videos_to_har_cameras()
        started = 0
        for spec in HAR_MODELS:
            path = assignments.get(spec.camera_id)
            if path is None or not path.is_file():
                logger.warning("HAR live: no video for %s", spec.camera_id)
                continue
            stream = HarLiveStream(spec.camera_id, spec.model_id, path)
            stream.start()
            with self._lock:
                self._streams[spec.camera_id] = stream
            started += 1
        logger.info("HAR live streams started (%d)", started)

    def stop_all(self) -> None:
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for s in streams:
            s.stop()

    def get_stream(self, camera_id: str) -> HarLiveStream | None:
        with self._lock:
            return self._streams.get(camera_id)

    def get_state(self, camera_id: str) -> dict[str, Any] | None:
        stream = self.get_stream(camera_id)
        return stream.get_state() if stream else None

    def set_playback(self, camera_id: str, *, playing: bool) -> bool:
        stream = self.get_stream(camera_id)
        if stream is None:
            return False
        stream.set_playback_active(playing)
        return True

    def set_playback_all(self, *, playing: bool) -> int:
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            stream.set_playback_active(playing)
        return len(streams)

    def all_states(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {cid: stream.get_state() for cid, stream in self._streams.items()}

    def runtime_for_camera(self, camera_id: str) -> dict[str, Any]:
        """Merge live state into camera list payloads."""
        stream = self.get_stream(camera_id)
        if stream is None:
            return {}
        state = stream.get_state()
        pred = state.get("prediction")
        overlays = []
        if pred:
            overlays = [
                build_activity_overlay(pred["label"], float(pred.get("confidence") or 0)),
            ]
        probe = None
        if pred:
            probe = {
                "mode": "har_activity",
                "model_id": state.get("model_id"),
                "prediction": pred,
                "videoUrl": state.get("videoUrl"),
                "source": f"live:{state.get('video')}",
            }
        return {
            "visionProbe": probe,
            "overlays": overlays,
            "videoUrl": state.get("videoUrl"),
            "harLive": {
                "inferring": state.get("inferring"),
                "error": state.get("error"),
            },
        }


_manager: HarLiveManager | None = None


def get_har_live_manager() -> HarLiveManager:
    global _manager
    if _manager is None:
        _manager = HarLiveManager()
    return _manager

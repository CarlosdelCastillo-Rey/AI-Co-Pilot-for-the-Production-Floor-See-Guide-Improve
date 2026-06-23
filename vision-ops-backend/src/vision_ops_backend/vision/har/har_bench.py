"""Configurable single-camera HAR bench for model/video/hyperparameter testing."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import (
    HAR_BENCH_CAMERA_ID,
    HAR_MODELS,
    spec_for_model,
)
from vision_ops_backend.vision.har.har_activity_client import (
    build_activity_entry,
    format_detections,
    format_tracked_detections,
    ingest_har_activity,
)
from vision_ops_backend.vision.har.live_stream import _har_infer_lock
from vision_ops_backend.vision.har.overlay import (
    compose_live_frame,
    compose_per_person_live_frame,
    detect_person_boxes,
)
from vision_ops_backend.vision.har.per_person import PerPersonHarEngine
from vision_ops_backend.vision.har.har_v2_session_client import HarV2SessionClient
from vision_ops_backend.vision.har.person_name_lookup import hydrate_track_names_from_registry
from vision_ops_backend.vision.har.track_registry import absorb_v2_event, enrich_track_predictions
from vision_ops_backend.vision.har.trigger_snapshot import capture_har_trigger_snapshot
from vision_ops_backend.vision.mock_videos import list_mock_video_files, public_url_for_video
from vision_ops_backend.vision.store import save_probe_result

logger = logging.getLogger(__name__)


@dataclass
class HarBenchConfig:
    infer_every: int = 16
    buffer_frames: int = 32
    stream_fps: int = 12
    show_heatmap: bool = False
    show_yolo_boxes: bool = True
    top_k: int = 5
    ingest_logs: bool = True
    per_person_mode: bool = True
    dwell_windows: int = 2
    bbox_padding: float = 0.15
    show_robo_hormiga: bool = False

    def clamp(self) -> None:
        self.infer_every = max(1, min(120, int(self.infer_every)))
        self.buffer_frames = max(8, min(128, int(self.buffer_frames)))
        self.stream_fps = max(1, min(60, int(self.stream_fps)))
        self.top_k = max(1, min(10, int(self.top_k)))
        self.dwell_windows = max(1, min(10, int(self.dwell_windows)))
        self.bbox_padding = max(0.0, min(0.5, float(self.bbox_padding)))


class HarBenchStream:
    """One sandbox stream — model, video, and hyperparameters are mutable at runtime."""

    def __init__(self, model_id: str, video_path: Path, camera_id: str = HAR_BENCH_CAMERA_ID) -> None:
        self.camera_id = camera_id
        self.model_id = model_id
        self.video_path = video_path
        self.config = HarBenchConfig(
            infer_every=settings.har_live_infer_every,
            buffer_frames=settings.har_live_buffer_frames,
            stream_fps=settings.har_live_stream_fps,
            show_heatmap=settings.har_live_show_heatmap,
            show_yolo_boxes=settings.har_live_show_yolo_boxes,
            show_robo_hormiga=settings.har_live_show_robo_hormiga,
        )
        self._lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame_bgr: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._infer_running = False
        self._frame_idx = 0
        self._playback_active = False
        self._session_id = f"har-bench-{uuid.uuid4().hex[:12]}"
        self._state: dict[str, Any] = self._build_state()
        self._per_person_engine = PerPersonHarEngine(
            buffer_frames=self.config.buffer_frames,
            bbox_padding=self.config.bbox_padding,
            dwell_windows=self.config.dwell_windows,
        )
        from vision_ops_backend.vision.har.robo_hormiga import RoboHormigaDetector
        self._robo_hormiga = RoboHormigaDetector()
        self._theft_state: dict[int, dict] = {}
        spec = spec_for_model(model_id)
        self._model_label = spec.label if spec else model_id
        self._v2_logger: HarV2SessionClient | None = None
        self._track_global: dict[int, str] = {}
        self._track_names: dict[int, str] = {}

    def _build_state(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "model_id": self.model_id,
            "video": self.video_path.name,
            "videoUrl": public_url_for_video(self.video_path),
            "inferring": False,
            "prediction": None,
            "error": None,
            "playback_active": False,
            "session_id": self._session_id,
            "backend": None,
            "device": None,
            "last_infer_ms": None,
            "per_person_mode": self.config.per_person_mode,
            "track_predictions": [],
            "person_count": 0,
            "latest_snapshot_url": None,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_state(self) -> dict[str, Any]:
        self._ensure_thread_alive()
        with self._lock:
            return {**dict(self._state), "config": asdict(self.config)}

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self.config)

    def _open_v2_session(self) -> None:
        if not settings.har_v2_session_enabled:
            self._v2_logger = None
            return
        with self._lock:
            self._session_id = f"har-bench-{uuid.uuid4().hex[:12]}"
            self._state["session_id"] = self._session_id
            model_id = self.model_id
            video_name = self.video_path.name
            cfg = asdict(self.config)
        self._v2_logger = HarV2SessionClient(
            session_id=self._session_id,
            source="bench",
            camera_id=self.camera_id,
            video_name=video_name,
            model_id=model_id,
            model_tag=model_id,
            hyperparams=cfg,
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
            self._robo_hormiga.reset_all()
            with self._lock:
                self._theft_state.clear()
            self._open_v2_session()
        elif not active and was_active:
            self._close_v2_session()

    def is_playback_active(self) -> bool:
        with self._lock:
            return self._playback_active

    def update_config(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key) and value is not None:
                    setattr(self.config, key, value)
            self.config.clamp()
            self._per_person_engine.reconfigure(
                buffer_frames=self.config.buffer_frames,
                bbox_padding=self.config.bbox_padding,
                dwell_windows=self.config.dwell_windows,
            )
            self._state["per_person_mode"] = self.config.per_person_mode
            if "per_person_mode" in kwargs:
                self._per_person_engine.reset()
                self._state["track_predictions"] = []
                self._state["prediction"] = None
            return asdict(self.config)

    def set_model(self, model_id: str) -> None:
        with self._lock:
            self.model_id = model_id
            self._state["model_id"] = model_id
            self._state["prediction"] = None
            self._state["track_predictions"] = []
            self._state["error"] = None
            spec = spec_for_model(model_id)
            self._model_label = spec.label if spec else model_id

    def set_video(self, video_path: Path) -> None:
        with self._lock:
            self.video_path = video_path
            self._state["video"] = video_path.name
            self._state["videoUrl"] = public_url_for_video(video_path)
            self._state["prediction"] = None
            self._state["error"] = None
            self._state["track_predictions"] = []
        self._per_person_engine.reset()
        self._track_global.clear()
        self._track_names.clear()
        self._robo_hormiga.reset_all()
        with self._lock:
            self._theft_state.clear()
        if self.is_playback_active():
            with self._lock:
                self._session_id = f"har-bench-{uuid.uuid4().hex[:12]}"
                self._state["session_id"] = self._session_id
            self._close_v2_session()
            self._open_v2_session()
        self._frame_idx = 0

    def reset_session(self) -> str:
        with self._lock:
            self._state["prediction"] = None
            self._state["track_predictions"] = []
            self._state["error"] = None
        self._per_person_engine.reset()
        self._track_global.clear()
        self._track_names.clear()
        self._robo_hormiga.reset_all()
        with self._lock:
            self._theft_state.clear()
        if self.is_playback_active():
            self._close_v2_session()
            self._open_v2_session()
        with self._lock:
            return self._session_id

    def start(self) -> None:
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"har-bench-{self.camera_id}",
        )
        self._thread.start()

    def _ensure_thread_alive(self) -> None:
        if not self._running:
            return
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            logger.warning("HAR bench thread stopped; restarting %s", self.camera_id)
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name=f"har-bench-{self.camera_id}",
            )
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
        buf: deque[np.ndarray] | None = None
        buf_rgb: deque[np.ndarray] | None = None
        buf_maxlen = 0
        current_video: Path | None = None

        def _ensure_buffers(target: int) -> tuple[deque[np.ndarray], deque[np.ndarray]]:
            nonlocal buf, buf_rgb, buf_maxlen
            if buf is not None and buf_rgb is not None and buf_maxlen == target:
                return buf, buf_rgb
            keep_bgr = list(buf)[-target:] if buf else []
            keep_rgb = list(buf_rgb)[-target:] if buf_rgb else []
            buf = deque(keep_bgr, maxlen=target)
            buf_rgb = deque(keep_rgb, maxlen=target)
            buf_maxlen = target
            return buf, buf_rgb

        while self._running:
            if not self.is_playback_active():
                time.sleep(0.08)
                continue

            with self._lock:
                cfg = HarBenchConfig(**asdict(self.config))
                cfg.clamp()
                video_path = self.video_path
                model_id = self.model_id
                model_label = self._model_label

            if cap is None or current_video != video_path or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(str(video_path))
                current_video = video_path
                self._frame_idx = 0
                buf = None
                buf_rgb = None
                buf_maxlen = 0
                self._per_person_engine.reset()
                self._robo_hormiga.reset_all()
                with self._lock:
                    self._theft_state.clear()
                if not cap.isOpened():
                    self._set_state(error=f"cannot open {video_path.name}")
                    time.sleep(2.0)
                    continue
                self._set_state(error=None)

            buf, buf_rgb = _ensure_buffers(cfg.buffer_frames)
            min_buf = max(8, cfg.buffer_frames // 2)
            infer_every = cfg.infer_every

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                if buf is not None:
                    buf.clear()
                if buf_rgb is not None:
                    buf_rgb.clear()
                self.reset_session()
                continue

            self._frame_idx += 1
            buf.append(frame.copy())
            buf_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with self._lock:
                pred = self._state.get("prediction")
                inferring = self._state.get("inferring", False)
                track_preds = list(self._state.get("track_predictions") or [])
                per_person = cfg.per_person_mode

            if per_person:
                self._latest_frame_bgr = frame
                tracked = self._per_person_engine.update_frame(frame)
                hydrate_track_names_from_registry(self._track_global, self._track_names)
                track_preds = enrich_track_predictions(
                    self._per_person_engine.track_predictions_payload(),
                    self._track_global,
                    self._track_names,
                )
                with self._lock:
                    self._state["track_predictions"] = track_preds
                    self._state["person_count"] = len(tracked)
                if cfg.show_robo_hormiga:
                    with self._lock:
                        theft_snap = dict(self._theft_state)
                    for tp in track_preds:
                        t_id = tp.get("track_id")
                        if t_id in theft_snap:
                            tp["theft_prob"] = theft_snap[t_id].get("theft_prob", 0.0)
                            tp["theft_alert"] = theft_snap[t_id].get("theft_alert", False)
                summary = self._per_person_engine.summary_prediction()
                rendered = compose_per_person_live_frame(
                    frame,
                    frames_rgb=list(buf_rgb),
                    model_label=model_label,
                    track_predictions=track_preds,
                    model_id=model_id,
                    inferring=inferring,
                    show_heatmap=cfg.show_heatmap,
                    show_boxes=cfg.show_yolo_boxes,
                    summary_prediction=summary,
                )
            else:
                rendered = compose_live_frame(
                    frame,
                    frames_rgb=list(buf_rgb),
                    model_label=model_label,
                    prediction=pred,
                    model_id=model_id,
                    inferring=inferring,
                    show_heatmap=cfg.show_heatmap,
                    show_boxes=cfg.show_yolo_boxes,
                )
            ok_enc, jpeg_buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok_enc:
                with self._lock:
                    self._latest_jpeg = jpeg_buf.tobytes()

            should_infer = (
                self.is_playback_active()
                and self._frame_idx % infer_every == 0
                and not self._infer_running
            )
            if per_person:
                should_infer = should_infer and bool(self._per_person_engine.tracks_ready_for_infer())
            else:
                should_infer = should_infer and len(buf) >= min_buf

            if should_infer:
                top_k = cfg.top_k
                ingest = cfg.ingest_logs
                if per_person:
                    track_ids = self._per_person_engine.tracks_ready_for_infer()
                    threading.Thread(
                        target=self._run_per_person_inference,
                        args=(track_ids, model_id, top_k, ingest, frame.copy(), cfg.show_robo_hormiga),
                        daemon=True,
                    ).start()
                else:
                    frames_copy = list(buf)
                    threading.Thread(
                        target=self._run_inference,
                        args=(frames_copy, model_id, top_k, ingest),
                        daemon=True,
                    ).start()

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            time.sleep(1.0 / max(native_fps, cfg.stream_fps))

        if cap is not None:
            cap.release()

    def _run_inference(
        self,
        frames_bgr: list[np.ndarray],
        model_id: str,
        top_k: int,
        ingest: bool,
    ) -> None:
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
                    model_id,
                    frames_bgr,
                    top_k=top_k,
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
            with self._lock:
                cfg = HarBenchConfig(**asdict(self.config))
                cfg.clamp()
            snapshot_url = capture_har_trigger_snapshot(
                frames_bgr,
                model_id=model_id,
                model_label=self._model_label,
                prediction=pred,
                show_heatmap=cfg.show_heatmap,
                show_boxes=cfg.show_yolo_boxes,
            )
            with self._lock:
                session_id = self._session_id
                frame_idx = self._frame_idx
                video_name = self.video_path.name
            payload = {
                "mode": "har_activity",
                "status": "ok",
                "model_id": model_id,
                "source": f"bench:{video_name}",
                "shared_clip": False,
                "backend": infer["backend"],
                "device": infer.get("device"),
                "prediction": pred,
                "overlay": overlay,
                "videoUrl": clip_url,
            }
            save_probe_result(self.camera_id, payload)
            if ingest and settings.har_activity_ingest_enabled:
                ingest_har_activity(
                    build_activity_entry(
                        camera_id=self.camera_id,
                        model_id=model_id,
                        prediction=pred,
                        source="bench",
                        backend=infer.get("backend"),
                        device=infer.get("device"),
                        detections=detections,
                        frame_index=frame_idx,
                        video_name=video_name,
                        clip_url=clip_url,
                        session_id=session_id,
                        infer_ms=round(infer_ms, 1),
                        snapshot_url=snapshot_url,
                        model_label=self._model_label,
                        hyperparams={
                            "infer_every": cfg.infer_every,
                            "buffer_frames": cfg.buffer_frames,
                            "stream_fps": cfg.stream_fps,
                            "show_heatmap": cfg.show_heatmap,
                            "show_yolo_boxes": cfg.show_yolo_boxes,
                            "top_k": cfg.top_k,
                            "ingest_logs": cfg.ingest_logs,
                        },
                    )
                )
            self._set_state(
                inferring=False,
                prediction=pred,
                backend=infer.get("backend"),
                device=infer.get("device"),
                error=None,
                last_infer_ms=round(infer_ms, 1),
            )
        except Exception as exc:
            logger.warning("HAR bench infer: %s", exc)
            self._set_state(inferring=False, error=str(exc))
        finally:
            self._infer_running = False

    def _run_per_person_inference(
        self,
        track_ids: list[int],
        model_id: str,
        top_k: int,
        ingest: bool,
        frame_bgr,
        show_robo_hormiga: bool = False,
    ) -> None:
        if not self.is_playback_active() or not track_ids:
            return
        self._infer_running = True
        self._set_state(inferring=True)
        t0 = time.perf_counter()
        try:
            from vision_ops_backend.vision.har.inference import run_har_inference

            exclude = settings.har_exclude_label_list or None
            last_backend: str | None = None
            last_device: str | None = None
            min_conf = settings.har_v2_min_confidence

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
                            model_id,
                            crops,
                            top_k=top_k,
                            exclude_labels=exclude,
                            return_embedding=True,
                            min_confidence=min_conf,
                        )
                    pred = infer["prediction"]
                    label_changed = self._per_person_engine.apply_track_prediction(tid, pred)
                    last_backend = infer.get("backend")
                    last_device = infer.get("device")

                    st = self._per_person_engine._tracks.get(tid)
                    bbox = list(st.last_bbox[:4]) if st else None
                    crop_jpeg = None
                    if crops:
                        ok, buf = cv2.imencode(".jpg", crops[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if ok:
                            crop_jpeg = buf.tobytes()
                    frame_jpeg = None
                    if label_changed:
                        with self._lock:
                            frame_idx = self._frame_idx
                        # frame snapshot handled on next loop iteration — skip full frame for now

                    if self._v2_logger:
                        v2_row = self._v2_logger.log_inference(
                            track_id=tid,
                            frame_idx=self._frame_idx,
                            bbox=bbox,
                            prediction=pred,
                            label_changed=label_changed,
                            uncertain=bool(pred.get("uncertain")),
                            infer_ms=None,
                            crop_jpeg=crop_jpeg,
                            frame_jpeg=frame_jpeg,
                            embedding=infer.get("embedding"),
                        )
                        absorb_v2_event(self._track_global, self._track_names, tid, v2_row)

                    if show_robo_hormiga:
                        theft_result = self._robo_hormiga.update(
                            tid, crops, embedding=infer.get("embedding")
                        )
                        with self._lock:
                            self._theft_state[tid] = theft_result
                finally:
                    self._per_person_engine.set_track_inferring(tid, False)

            hydrate_track_names_from_registry(self._track_global, self._track_names)
            track_payload = enrich_track_predictions(
                self._per_person_engine.track_predictions_payload(),
                self._track_global,
                self._track_names,
            )
            summary = self._per_person_engine.summary_prediction()
            pred = summary or {}
            infer_ms = (time.perf_counter() - t0) * 1000.0
            clip_url = public_url_for_video(self.video_path)
            with self._lock:
                cfg = HarBenchConfig(**asdict(self.config))
                cfg.clamp()
                session_id = self._session_id
                frame_idx = self._frame_idx
                video_name = self.video_path.name

            detections = format_tracked_detections(track_payload)
            snapshot_url = None
            if summary and frame_bgr is not None:
                snapshot_url = capture_har_trigger_snapshot(
                    [frame_bgr],
                    model_id=model_id,
                    model_label=self._model_label,
                    prediction={
                        "label": summary.get("label"),
                        "confidence": summary.get("confidence"),
                        "track_id": summary.get("track_id"),
                        "top_k": [],
                    },
                    show_heatmap=cfg.show_heatmap,
                    show_boxes=cfg.show_yolo_boxes,
                    track_predictions=track_payload,
                )
            if ingest and settings.har_activity_ingest_enabled and summary:
                ingest_har_activity(
                    build_activity_entry(
                        camera_id=self.camera_id,
                        model_id=model_id,
                        prediction={
                            "label": summary.get("label"),
                            "confidence": summary.get("confidence"),
                            "track_id": summary.get("track_id"),
                        },
                        source="bench:per_person",
                        backend=last_backend,
                        device=last_device,
                        detections=detections,
                        frame_index=frame_idx,
                        video_name=video_name,
                        clip_url=clip_url,
                        session_id=session_id,
                        infer_ms=round(infer_ms, 1),
                        snapshot_url=snapshot_url,
                        model_label=self._model_label,
                        hyperparams={
                            "infer_every": cfg.infer_every,
                            "buffer_frames": cfg.buffer_frames,
                            "per_person_mode": True,
                            "dwell_windows": cfg.dwell_windows,
                            "bbox_padding": cfg.bbox_padding,
                            "tracks_inferred": len(track_ids),
                        },
                    )
                )

            bench_update: dict[str, Any] = {
                "inferring": False,
                "prediction": pred if summary else None,
                "track_predictions": track_payload,
                "person_count": len(track_payload),
                "backend": last_backend,
                "device": last_device,
                "error": None,
                "last_infer_ms": round(infer_ms, 1),
                "per_person_mode": True,
            }
            if snapshot_url:
                bench_update["latest_snapshot_url"] = snapshot_url
            self._set_state(**bench_update)
        except Exception as exc:
            logger.warning("HAR bench per-person infer: %s", exc)
            self._set_state(inferring=False, error=str(exc))
        finally:
            self._infer_running = False

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
        cfg = self.config
        snapshot_url = capture_har_trigger_snapshot(
            [frame],
            model_id=self.model_id,
            model_label=self._model_label,
            prediction=prediction,
            show_heatmap=cfg.show_heatmap,
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


class HarBenchManager:
    def __init__(self) -> None:
        self._stream: HarBenchStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not settings.har_enabled or not settings.har_live_enabled:
            return
        threading.Thread(target=self._start_worker, daemon=True, name="har-bench-boot").start()

    def _start_worker(self) -> None:
        try:
            from vision_ops_backend.vision.har.checkpoints import get_registry
            from vision_ops_backend.vision.har.extractors import preload_har_extractors

            get_registry().ensure_loaded()
            preload_har_extractors()
        except Exception as exc:
            logger.warning("HAR bench preload failed: %s", exc)

        videos = list_mock_video_files()
        if not videos:
            logger.warning("HAR bench: no mock videos found")
            return
        default_model = HAR_MODELS[0].model_id
        stream = HarBenchStream(default_model, videos[0])
        stream.start()
        with self._lock:
            self._stream = stream
        logger.info("HAR bench stream started (%s on %s)", default_model, videos[0].name)

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream:
            stream.stop()

    def get_stream(self) -> HarBenchStream | None:
        with self._lock:
            return self._stream

    def get_state(self) -> dict[str, Any] | None:
        stream = self.get_stream()
        return stream.get_state() if stream else None

    def set_playback(self, *, playing: bool) -> bool:
        stream = self.get_stream()
        if stream is None:
            return False
        stream.set_playback_active(playing)
        return True

    def snapshot(self) -> dict[str, Any]:
        from vision_ops_backend.vision.har.checkpoints import get_registry

        registry = get_registry()
        videos = [
            {"name": p.name, "url": public_url_for_video(p)}
            for p in list_mock_video_files()
        ]
        models = [
            {
                "model_id": spec.model_id,
                "label": spec.label,
                "ready": registry._ready.get(spec.ckpt_key, False),
            }
            for spec in HAR_MODELS
        ]
        state = self.get_state()
        base = settings.public_api_base.rstrip("/")
        return {
            "enabled": settings.har_enabled and settings.har_live_enabled,
            "camera_id": HAR_BENCH_CAMERA_ID,
            "stream_url": f"{base}/api/cameras/{HAR_BENCH_CAMERA_ID}/stream",
            "state": state,
            "videos": videos,
            "models": models,
        }


_manager: HarBenchManager | None = None


def get_har_bench_manager() -> HarMockWallManager:
    from vision_ops_backend.vision.har.har_mock_wall import get_mock_wall_manager

    return get_mock_wall_manager()

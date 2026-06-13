"""Batch video eval with preview frames — har-research launch_eval_dashboard parity."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.har_v2_session_client import HarV2SessionClient, new_session_id
from vision_ops_backend.vision.har.live_stream import _har_infer_lock
from vision_ops_backend.vision.har.overlay import compose_per_person_live_frame
from vision_ops_backend.vision.har.per_person import PerPersonHarEngine

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


@dataclass
class EvalConfig:
    model_id: str
    video_path: Path
    max_frames: int = 600
    infer_every: int = 16
    buffer_frames: int = 32
    dwell_windows: int = 2
    bbox_padding: float = 0.15
    min_confidence: float = 0.25
    top_k: int = 5
    preview_width: int = 720


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_eval_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def start_eval_job(config: EvalConfig) -> str:
    job_id = f"eval-{uuid.uuid4().hex[:12]}"
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "frames_done": 0,
            "max_frames": config.max_frames,
            "session_id": None,
            "error": None,
            "model_id": config.model_id,
            "video": config.video_path.name,
        }
    threading.Thread(target=_run_eval, args=(job_id, config), daemon=True, name=f"har-eval-{job_id}").start()
    return job_id


def _run_eval(job_id: str, cfg: EvalConfig) -> None:
    from vision_ops_backend.vision.har.constants import spec_for_model
    from vision_ops_backend.vision.har.inference import run_har_inference

    spec = spec_for_model(cfg.model_id)
    model_label = spec.label if spec else cfg.model_id
    session_id = new_session_id()
    v2 = HarV2SessionClient(
        session_id=session_id,
        source="eval",
        video_name=cfg.video_path.name,
        model_id=cfg.model_id,
        model_tag=cfg.model_id,
        hyperparams={
            "infer_every": cfg.infer_every,
            "buffer_frames": cfg.buffer_frames,
            "dwell_windows": cfg.dwell_windows,
            "max_frames": cfg.max_frames,
        },
    )
    _update_job(job_id, session_id=session_id)

    engine = PerPersonHarEngine(
        buffer_frames=cfg.buffer_frames,
        bbox_padding=cfg.bbox_padding,
        dwell_windows=cfg.dwell_windows,
    )
    exclude = settings.har_exclude_label_list or None
    track_stats: dict[int, dict[str, Any]] = {}
    preview_manifest: list[dict[str, Any]] = []

    cap = cv2.VideoCapture(str(cfg.video_path))
    if not cap.isOpened():
        _update_job(job_id, status="error", error=f"Cannot open {cfg.video_path}")
        return

    frame_i = 0
    infer_every = max(1, cfg.infer_every)

    try:
        while frame_i < cfg.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            tracked = engine.update_frame(frame)
            track_payload = engine.track_predictions_payload()

            if frame_i % infer_every == 0:
                for tid in engine.tracks_ready_for_infer():
                    crops = engine.get_crop_frames(tid)
                    if len(crops) < engine.min_buffer_for_infer():
                        continue
                    engine.set_track_inferring(tid, True)
                    try:
                        with _har_infer_lock:
                            infer = run_har_inference(
                                cfg.model_id,
                                crops,
                                top_k=cfg.top_k,
                                exclude_labels=exclude,
                                return_embedding=True,
                                min_confidence=cfg.min_confidence,
                            )
                        pred = infer["prediction"]
                        label_changed = engine.apply_track_prediction(tid, pred)
                        st = engine._tracks.get(tid)
                        bbox = list(st.last_bbox[:4]) if st else None
                        crop_jpeg = None
                        if crops:
                            ok_j, buf = cv2.imencode(".jpg", crops[-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_j:
                                crop_jpeg = buf.tobytes()
                        v2.log_inference(
                            track_id=tid,
                            frame_idx=frame_i,
                            bbox=bbox,
                            prediction=pred,
                            label_changed=label_changed,
                            uncertain=bool(pred.get("uncertain")),
                            crop_jpeg=crop_jpeg,
                            embedding=infer.get("embedding"),
                        )
                    finally:
                        engine.set_track_inferring(tid, False)
                track_payload = engine.track_predictions_payload()

            for tr in track_payload:
                tid = int(tr["track_id"])
                st = track_stats.setdefault(
                    tid,
                    {"track_id": tid, "n_inferences": 0, "action_counts": {}, "first_frame": frame_i, "last_frame": frame_i},
                )
                st["last_frame"] = frame_i
                if tr.get("action_label"):
                    st["current_label"] = tr["action_label"]
                    st["current_conf"] = tr.get("action_confidence")

            h, w = frame.shape[:2]
            scale = cfg.preview_width / w if w > cfg.preview_width else 1.0
            nh = max(1, int(h * scale))
            nw = max(1, int(w * scale))
            small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            rgb_small = [cv2.cvtColor(small, cv2.COLOR_BGR2RGB)]
            rendered = compose_per_person_live_frame(
                small,
                frames_rgb=rgb_small,
                model_label=model_label,
                track_predictions=track_payload,
                model_id=cfg.model_id,
                inferring=False,
                show_heatmap=False,
                show_boxes=True,
                summary_prediction=engine.summary_prediction(),
            )
            ok_j, buf = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok_j:
                _post_preview_frame(session_id, frame_i, buf.tobytes(), track_payload)

            preview_manifest.append({"frame_idx": frame_i, "n_tracks": len(track_payload), "tracks": track_payload})
            _update_job(job_id, progress=round(100 * frame_i / cfg.max_frames, 1), frames_done=frame_i)

        v2.finalize()
        _post_preview_manifest(session_id, preview_manifest, track_stats)
        _update_job(
            job_id,
            status="ok",
            progress=100,
            frames_done=frame_i,
            n_tracks=len(track_stats),
            track_stats=list(track_stats.values()),
        )
    except Exception as exc:
        logger.exception("HAR eval job %s failed", job_id)
        _update_job(job_id, status="error", error=str(exc))
    finally:
        cap.release()


def _post_preview_frame(session_id: str, frame_idx: int, jpeg: bytes, tracks: list[dict]) -> None:
    try:
        from vision_ops_alerting.db.session import SessionLocal
        from vision_ops_alerting.services.har_session_store import save_preview_frame

        with SessionLocal() as db:
            save_preview_frame(db, session_id, frame_idx=frame_idx, jpeg=jpeg, tracks=tracks)
            db.commit()
    except Exception as exc:
        logger.debug("preview frame save failed: %s", exc)


def _post_preview_manifest(session_id: str, manifest: list[dict], track_stats: dict) -> None:
    try:
        from vision_ops_alerting.services.har_session_store import save_preview_manifest

        save_preview_manifest(
            session_id,
            frames=len(manifest),
            track_stats=list(track_stats.values()),
        )
    except Exception as exc:
        logger.debug("preview manifest save failed: %s", exc)

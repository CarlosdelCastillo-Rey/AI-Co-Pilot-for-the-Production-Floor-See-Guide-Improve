"""Export annotated per-person eval video with full session logging."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from lib.inference import HarPredictor
from lib.overlay import draw_hud, draw_tracks
from lib.session_log import HarSessionLogger, new_session_id
from lib.tracking import PerPersonTracker


def _open_video_writer(output_path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    """Prefer H.264 (browser-playable); fall back to mp4v."""
    for fourcc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (w, h),
        )
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot create video writer for {output_path}")


def iter_annotated_frames(
    *,
    video_path: Path,
    checkpoint: Path,
    max_frames: int = 600,
    infer_every: int = 16,
    buffer_frames: int = 32,
    dwell_windows: int = 2,
    min_confidence: float = 0.25,
    log_session: bool = True,
    save_crop_every_event: bool = True,
):
    """Yield (rendered_bgr_frame, meta) — same loop as render_eval_video."""
    predictor = HarPredictor(checkpoint, min_confidence=min_confidence)
    tracker = PerPersonTracker(buffer_frames=buffer_frames, dwell_windows=dwell_windows)
    session_id = new_session_id()
    logger = HarSessionLogger(
        checkpoint,
        predictor_info=predictor.info,
        source="eval",
        video_name=video_path.name,
        session_id=session_id,
        save_crop_every_event=save_crop_every_event,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    frame_i = 0
    infer_ms: float | None = None

    try:
        while frame_i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            tracker.update_frame(frame)

            frame_inferences: list[dict] = []
            if frame_i % infer_every == 0:
                for tid in tracker.ready_track_ids():
                    tracker.set_inferring(tid, True)
                    t0 = time.perf_counter()
                    pred = predictor.predict_from_crops(tracker.get_crops(tid))
                    ms = (time.perf_counter() - t0) * 1000.0
                    infer_ms = ms
                    change = {"label_changed": False, "display_label": None, "display_conf": 0.0}
                    if pred.get("label"):
                        change = tracker.apply_prediction(tid, pred)
                    row = None
                    if log_session:
                        row = logger.log_inference(
                            track_id=tid,
                            frame_idx=frame_i,
                            frame_bgr=frame,
                            bbox=tracker.get_bbox(tid),
                            prediction=pred,
                            label_changed=change["label_changed"],
                            infer_ms=ms,
                            save_for_review=pred.get("uncertain", False),
                        )
                    frame_inferences.append({
                        "track_id": tid,
                        "raw_label": pred.get("raw_label"),
                        "raw_confidence": pred.get("raw_confidence"),
                        "display_label": pred.get("label"),
                        "confidence": pred.get("confidence"),
                        "uncertain": pred.get("uncertain"),
                        "label_changed": change["label_changed"],
                        "infer_ms": round(ms, 1),
                        "global_person_id": (row or {}).get("global_person_id"),
                        "reid_match_score": (row or {}).get("reid_match_score"),
                    })
                    tracker.set_inferring(tid, False)

            payload = tracker.to_payload()
            rendered = draw_tracks(frame, payload)
            rendered = draw_hud(
                rendered,
                n_persons=len(payload),
                infer_ms=infer_ms,
                session_id=session_id,
                model_name=logger.model_tag,
            )
            yield rendered, {
                "frame_i": frame_i,
                "n_tracks": len(payload),
                "tracks": payload,
                "frame_inferences": frame_inferences,
                "infer_ms": infer_ms,
                "session_id": session_id,
                "model_tag": logger.model_tag,
            }
    finally:
        cap.release()

    summary = logger.finalize() if log_session else {}
    yield None, {
        "done": True,
        "frames": frame_i,
        "session_id": session_id,
        "session_dir": str(logger.session_dir),
        "log_summary": summary,
        "predictor": predictor,
    }


def render_eval_video(
    *,
    video_path: Path,
    checkpoint: Path,
    output_path: Path,
    max_frames: int = 600,
    infer_every: int = 16,
    buffer_frames: int = 32,
    dwell_windows: int = 2,
    min_confidence: float = 0.25,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(output_path, fps, w, h)

    frame_i = 0
    session_id = ""
    logger_meta: dict = {}

    for rendered, meta in iter_annotated_frames(
        video_path=video_path,
        checkpoint=checkpoint,
        max_frames=max_frames,
        infer_every=infer_every,
        buffer_frames=buffer_frames,
        dwell_windows=dwell_windows,
        min_confidence=min_confidence,
        log_session=True,
    ):
        if rendered is None:
            logger_meta = meta
            break
        writer.write(rendered)
        frame_i = meta["frame_i"]
        session_id = meta["session_id"]

    writer.release()

    from lib.video_preview import transcode_for_web
    web_path = transcode_for_web(output_path)
    summary = logger_meta.get("log_summary", {})
    return {
        "output": str(output_path),
        "web_output": str(web_path) if web_path != output_path else None,
        "frames": frame_i,
        "session_id": session_id,
        "video": video_path.name,
        "session_dir": logger_meta.get("session_dir", ""),
        "log_summary": summary,
    }

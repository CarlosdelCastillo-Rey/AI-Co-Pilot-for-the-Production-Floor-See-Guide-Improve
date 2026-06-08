"""Export annotated per-person eval video with full session logging."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from lib.inference import HarPredictor
from lib.overlay import draw_hud, draw_tracks
from lib.session_log import HarSessionLogger, new_session_id
from lib.tracking import PerPersonTracker


def render_eval_video(
    *,
    video_path: Path,
    checkpoint: Path,
    output_path: Path,
    max_frames: int = 600,
    infer_every: int = 16,
    buffer_frames: int = 32,
    dwell_windows: int = 2,
) -> dict:
    predictor = HarPredictor(checkpoint)
    tracker = PerPersonTracker(buffer_frames=buffer_frames, dwell_windows=dwell_windows)
    session_id = new_session_id()
    logger = HarSessionLogger(
        checkpoint,
        predictor_info=predictor.info,
        source="eval",
        video_name=video_path.name,
        session_id=session_id,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_i = 0
    infer_ms = None

    while frame_i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        tracker.update_frame(frame)

        if frame_i % infer_every == 0:
            for tid in tracker.ready_track_ids():
                tracker.set_inferring(tid, True)
                t0 = time.perf_counter()
                pred = predictor.predict_from_crops(tracker.get_crops(tid))
                ms = (time.perf_counter() - t0) * 1000.0
                infer_ms = ms
                if pred.get("label"):
                    change = tracker.apply_prediction(tid, pred)
                    logger.log_inference(
                        track_id=tid,
                        frame_idx=frame_i,
                        frame_bgr=frame,
                        bbox=tracker.get_bbox(tid),
                        prediction=pred,
                        label_changed=change["label_changed"],
                        infer_ms=ms,
                    )
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
        writer.write(rendered)

    cap.release()
    writer.release()
    summary = logger.finalize()
    return {
        "output": str(output_path),
        "frames": frame_i,
        "session_id": session_id,
        "video": video_path.name,
        "session_dir": str(logger.session_dir),
        "log_summary": summary,
    }

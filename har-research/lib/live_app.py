#!/usr/bin/env python3
"""
Live per-person HAR — OpenCV window.

Each ByteTrack id gets its own crop buffer, V-JEPA2 embedding, and MLP label.
Session logs: outputs/har_sessions/YYYY-MM-DD/<session>_<model_tag>/

Usage:
  cd notebooks
  python lib/live_app.py
  python lib/live_app.py --video ../data_sample/mock-videos/Industrial-One.mp4
  python lib/live_app.py --webcam 0

Keys: q quit · space pause · r reset tracks/memory
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

NB = Path(__file__).resolve().parents[1]
if str(NB) not in sys.path:
    sys.path.insert(0, str(NB))

from lib.constants import DEFAULT_BUFFER_FRAMES, DEFAULT_DWELL_WINDOWS, DEFAULT_INFER_EVERY
from lib.inference import HarPredictor
from lib.overlay import draw_hud, draw_tracks
from lib.paths import CHECKPOINTS_DIR, default_mock_video, ensure_notebook_paths
from lib.session_log import HarSessionLogger, list_sessions, new_session_id
from lib.tracking import PerPersonTracker


def run_live(
    *,
    checkpoint: Path,
    video: Path | None,
    webcam: int | None,
    infer_every: int,
    buffer_frames: int,
    dwell_windows: int,
    max_seconds: float | None,
    min_confidence: float = 0.25,
) -> int:
    ensure_notebook_paths()
    predictor = HarPredictor(checkpoint, min_confidence=min_confidence)
    tracker = PerPersonTracker(buffer_frames=buffer_frames, dwell_windows=dwell_windows)

    def _start_logger() -> HarSessionLogger:
        return HarSessionLogger(
            checkpoint,
            predictor_info=predictor.info,
            source="webcam" if webcam is not None else "live",
            video_name=video.name if video else None,
            session_id=new_session_id(),
        )

    logger = _start_logger()
    session_id = logger.session_id

    if webcam is not None:
        cap = cv2.VideoCapture(int(webcam))
    elif video is not None:
        cap = cv2.VideoCapture(str(video))
    else:
        dv = default_mock_video()
        if dv is None:
            print("No mock video in data_sample/mock-videos/ — use --video or --webcam")
            return 1
        cap = cv2.VideoCapture(str(dv))
        video = dv

    if not cap.isOpened():
        print("Could not open video source")
        return 1

    win = "VisionOps Per-Person HAR — q quit · space pause · r reset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False
    frame_i = 0
    infer_ms: float | None = None
    infer_lock = threading.Lock()
    infer_running = False
    t_start = time.time()
    last_frame = None

    def _infer(tid: int) -> None:
        nonlocal infer_ms, infer_running, last_frame, frame_i
        infer_running = True
        tracker.set_inferring(tid, True)
        t0 = time.perf_counter()
        try:
            with infer_lock:
                pred = predictor.predict_from_crops(tracker.get_crops(tid))
            if pred.get("label") or pred.get("uncertain"):
                change = (
                    tracker.apply_prediction(tid, pred)
                    if pred.get("label")
                    else {"label_changed": False, "display_label": None, "display_conf": 0.0}
                )
                ms = (time.perf_counter() - t0) * 1000.0
                infer_ms = ms
                logger.log_inference(
                    track_id=tid,
                    frame_idx=frame_i,
                    frame_bgr=last_frame,
                    bbox=tracker.get_bbox(tid),
                    prediction=pred,
                    label_changed=change["label_changed"],
                    infer_ms=ms,
                    save_for_review=pred.get("uncertain", False),
                )
        finally:
            tracker.set_inferring(tid, False)
            infer_running = False

    print(f"Session: {session_id}")
    print(f"Model tag: {logger.model_tag}")
    print(f"Log dir: {logger.session_dir}")
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Classes: {predictor.class_names}")
    if video:
        print(f"Video: {video.name}")

    while True:
        if max_seconds and time.time() - t_start > max_seconds:
            break

        if not paused:
            ok, frame = cap.read()
            if not ok:
                if webcam is not None:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_i = 0
                tracker.reset()
                continue
            frame_i += 1
            tracker.update_frame(frame)
            last_frame = frame

            if frame_i % infer_every == 0 and not infer_running:
                for tid in tracker.ready_track_ids():
                    threading.Thread(target=_infer, args=(tid,), daemon=True).start()

        if last_frame is None:
            time.sleep(0.05)
            continue
        display = last_frame

        payload = tracker.to_payload()
        rendered = draw_tracks(display, payload)
        rendered = draw_hud(
            rendered,
            n_persons=len(payload),
            infer_ms=infer_ms,
            session_id=session_id,
            model_name=logger.model_tag,
        )
        cv2.imshow(win, rendered)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            logger.finalize()
            tracker.reset()
            logger = _start_logger()
            session_id = logger.session_id
            print(f"Reset — new session {session_id}")

        if not paused:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            time.sleep(max(0.001, 1.0 / min(fps, 30.0)))

    cap.release()
    cv2.destroyAllWindows()
    deadline = time.time() + 30.0
    while infer_running and time.time() < deadline:
        time.sleep(0.05)
    summary = logger.finalize()
    print(f"Session log: {logger.session_dir}")
    print(f"Events: {summary['n_events']} ({summary['n_confirmed']} confirmed) · persons: {summary['n_persons']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-person HAR live demo")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "har_vjepa_12c_topdown_allavail.pt")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY)
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER_FRAMES)
    parser.add_argument("--dwell", type=int, default=DEFAULT_DWELL_WINDOWS)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--list-sessions", action="store_true", help="Show recent logged sessions")
    args = parser.parse_args()

    if args.list_sessions:
        rows = list_sessions(limit=20)
        if not rows:
            print("No sessions yet — run live or eval first.")
            return 0
        for r in rows:
            print(
                f"{r['date']}  {r['model_tag']:16}  {r['source']:6}  "
                f"events={r['n_events']} confirmed={r['n_confirmed']}  {r['session_dir']}"
            )
        return 0

    if not args.checkpoint.is_file():
        print(f"Checkpoint missing: {args.checkpoint}")
        print("Run 03_Train_HAR_Head.ipynb or 00_Pipeline_Run_All.ipynb first.")
        return 1

    return run_live(
        checkpoint=args.checkpoint,
        video=args.video,
        webcam=args.webcam,
        infer_every=args.infer_every,
        buffer_frames=args.buffer,
        dwell_windows=args.dwell,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())

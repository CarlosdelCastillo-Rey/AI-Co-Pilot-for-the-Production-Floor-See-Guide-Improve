"""Persist HAR probe results and shared preview artifacts."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vision_ops_backend.vision.har.constants import HAR_MODEL_IDS, spec_for_model
from vision_ops_backend.vision.har.har_activity_client import (
    build_activity_entry,
    format_detections,
    ingest_har_activity,
)
from vision_ops_backend.vision.har.inference import run_har_inference
from vision_ops_backend.vision.har.overlay import detect_person_boxes
from vision_ops_backend.vision.har.persist import build_persist_payload, persist_har_run
from vision_ops_backend.vision.har.shared_clip import load_har_frames_for_camera
from vision_ops_backend.vision.mock_videos import assign_videos_to_har_cameras, public_url_for_video
from vision_ops_backend.vision.preview import draw_overlay_on_frame
from vision_ops_backend.vision.store import save_preview, save_probe_result, save_still


def build_activity_overlay(label: str, confidence: float) -> dict[str, Any]:
    pct = int(round(confidence * 100))
    variant = "primary" if confidence >= 0.35 else "tertiary"
    return {
        "type": "machine",
        "label": f"{label} ({pct}%)",
        "top": "8%",
        "left": "8%",
        "width": "55%",
        "height": "18%",
        "variant": variant,
    }


def run_har_camera_probe(
    model_id: str,
    *,
    clip_path: str | None = None,
    frames_bgr: list[np.ndarray] | None = None,
    source: str | None = None,
    video_assignments: dict | None = None,
) -> dict[str, Any]:
    spec = spec_for_model(model_id)
    if spec is None:
        raise ValueError(f"unknown model_id: {model_id}")

    camera_id = spec.camera_id
    if frames_bgr is None:
        if clip_path is None and video_assignments:
            assigned = video_assignments.get(camera_id)
            if assigned is not None:
                clip_path = str(assigned)
        frames_bgr, source = load_har_frames_for_camera(camera_id, clip_path=clip_path)
    if not frames_bgr:
        raise FileNotFoundError("no frames available for HAR probe")

    infer = run_har_inference(model_id, frames_bgr)
    pred = infer["prediction"]
    overlay = build_activity_overlay(pred["label"], pred["confidence"])

    middle = frames_bgr[len(frames_bgr) // 2]
    preview_frame = draw_overlay_on_frame(middle, overlay)

    save_still(camera_id, middle)
    save_preview(camera_id, preview_frame)

    mock_path = (video_assignments or {}).get(camera_id)
    video_url = public_url_for_video(mock_path) if mock_path is not None else None

    payload: dict[str, Any] = {
        "mode": "har_activity",
        "status": "ok",
        "model_id": model_id,
        "source": source or "shared",
        "shared_clip": False,
        "backend": infer["backend"],
        "device": infer.get("device"),
        "prediction": pred,
        "overlay": overlay,
        "videoUrl": video_url,
        "artifacts": {
            "still": f"/api/vision/artifacts/{camera_id}/still",
            "preview": f"/api/vision/artifacts/{camera_id}/preview",
        },
    }
    saved = save_probe_result(camera_id, payload)
    middle = frames_bgr[len(frames_bgr) // 2]
    boxes = detect_person_boxes(middle)
    detections = format_detections(
        boxes,
        action_label=pred.get("label"),
        action_confidence=pred.get("confidence"),
    )
    ingest_har_activity(
        build_activity_entry(
            camera_id=camera_id,
            model_id=model_id,
            prediction=pred,
            source="probe",
            backend=infer.get("backend"),
            device=infer.get("device"),
            detections=detections,
            video_name=mock_path.name if mock_path is not None else None,
            clip_url=video_url,
            new_session=True,
            model_label=spec.label if spec else model_id,
            hyperparams={"source": "probe", "frame_sample": len(frames_bgr)},
        )
    )
    persist_har_run(
        build_persist_payload(
            run_type="single",
            source=source or "shared",
            probes=[saved],
            errors=[],
            status="ok",
            clip_path=clip_path,
            frame_count=len(frames_bgr),
            shared_clip=False,
        )
    )
    return saved


def run_all_har_probes(*, clip_path: str | None = None, reshuffle_videos: bool = True) -> dict[str, Any]:
    """Run each HAR model on its own randomly assigned mock video."""
    assignments = assign_videos_to_har_cameras(reshuffle=reshuffle_videos)
    if not assignments and clip_path is None:
        raise FileNotFoundError(
            "no mock videos in vision-ops-app/public/mock-videos and no clip_path override"
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    sources: list[str] = []

    for model_id in HAR_MODEL_IDS:
        spec = spec_for_model(model_id)
        if spec is None:
            continue
        try:
            cam_id = spec.camera_id
            path = assignments.get(cam_id)
            clip = clip_path or (str(path) if path else None)
            frames, source = load_har_frames_for_camera(cam_id, clip_path=clip)
            if not frames:
                raise FileNotFoundError(f"no frames for {cam_id}")
            sources.append(source)

            infer = run_har_inference(model_id, frames)
            pred = infer["prediction"]
            overlay = build_activity_overlay(pred["label"], pred["confidence"])
            middle = frames[len(frames) // 2]
            preview_frame = draw_overlay_on_frame(middle, overlay)
            save_still(cam_id, middle)
            save_preview(cam_id, preview_frame)

            video_url = public_url_for_video(path) if path is not None else None
            payload: dict[str, Any] = {
                "mode": "har_activity",
                "status": "ok",
                "model_id": model_id,
                "source": source,
                "shared_clip": False,
                "backend": infer["backend"],
                "device": infer.get("device"),
                "prediction": pred,
                "overlay": overlay,
                "videoUrl": video_url,
                "artifacts": {
                    "still": f"/api/vision/artifacts/{cam_id}/still",
                    "preview": f"/api/vision/artifacts/{cam_id}/preview",
                },
            }
            saved = save_probe_result(cam_id, payload)
            results.append(saved)
            mid = frames[len(frames) // 2]
            boxes = detect_person_boxes(mid)
            detections = format_detections(
                boxes,
                action_label=pred.get("label"),
                action_confidence=pred.get("confidence"),
            )
            ingest_har_activity(
                build_activity_entry(
                    camera_id=cam_id,
                    model_id=model_id,
                    prediction=pred,
                    source="probe",
                    backend=infer.get("backend"),
                    device=infer.get("device"),
                    detections=detections,
                    video_name=path.name if path is not None else None,
                    clip_url=video_url,
                    new_session=True,
                    model_label=spec.label,
                    hyperparams={"source": "probe", "frame_sample": len(frames)},
                )
            )
        except Exception as exc:
            errors.append({"model_id": model_id, "error": str(exc)})

    out = {
        "status": "ok" if results else "error",
        "source": sources[0] if sources else "mixed",
        "shared_clip": False,
        "video_assignments": {
            cid: public_url_for_video(p) for cid, p in assignments.items()
        },
        "probes": results,
        "errors": errors,
    }
    if results and errors:
        out["status"] = "partial"

    db_run = persist_har_run(
        build_persist_payload(
            run_type="batch",
            source=out["source"],
            probes=results,
            errors=errors,
            status=out["status"],
            clip_path=clip_path,
            frame_count=None,
            shared_clip=False,
        )
    )
    if db_run:
        out["db_run_id"] = db_run.get("id")
    return out

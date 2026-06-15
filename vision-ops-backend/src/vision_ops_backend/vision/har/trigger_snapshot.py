"""Save composed HAR frames when activity is logged (for timeline alert thumbnails)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.vision.har.overlay import compose_live_frame, compose_per_person_live_frame
from vision_ops_backend.vision.store import save_alert_snapshot

logger = logging.getLogger(__name__)

# Sentinel: dim background + highlight acting person when no ByteTrack id exists.
_SNAPSHOT_FOCUS_ANY = "__snapshot_focus__"


def capture_har_trigger_snapshot(
    frames_bgr: list[np.ndarray],
    *,
    model_id: str,
    model_label: str,
    prediction: dict[str, Any],
    show_heatmap: bool,
    show_boxes: bool,
    track_predictions: list[dict[str, Any]] | None = None,
) -> str | None:
    """Render the live overlay stack and persist a JPEG for timeline evidence."""
    if not frames_bgr:
        return None
    try:
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        last = frames_bgr[-1]
        if track_predictions:
            focus_track_id = prediction.get("track_id")
            if focus_track_id is None:
                acting = [
                    t
                    for t in track_predictions
                    if t.get("action_label") and t.get("action_confidence") is not None
                ]
                if acting:
                    focus_track_id = max(
                        acting,
                        key=lambda t: float(t.get("action_confidence") or 0),
                    ).get("track_id")
            summary = {
                "label": prediction.get("label"),
                "confidence": prediction.get("confidence"),
                "track_id": focus_track_id,
            }
            rendered = compose_per_person_live_frame(
                last,
                frames_rgb=frames_rgb,
                model_label=model_label,
                track_predictions=track_predictions,
                model_id=model_id,
                inferring=False,
                show_heatmap=show_heatmap,
                show_boxes=show_boxes,
                summary_prediction=summary if prediction.get("label") else None,
                focus_track_id=focus_track_id,
            )
        else:
            focus_track_id = prediction.get("track_id") or _SNAPSHOT_FOCUS_ANY
            rendered = compose_live_frame(
                last,
                frames_rgb=frames_rgb,
                model_label=model_label,
                prediction=prediction,
                model_id=model_id,
                inferring=False,
                show_heatmap=show_heatmap,
                show_boxes=show_boxes,
                focus_track_id=focus_track_id,
            )
        return save_alert_snapshot(rendered)
    except Exception as exc:
        logger.warning("HAR trigger snapshot failed: %s", exc)
        return None

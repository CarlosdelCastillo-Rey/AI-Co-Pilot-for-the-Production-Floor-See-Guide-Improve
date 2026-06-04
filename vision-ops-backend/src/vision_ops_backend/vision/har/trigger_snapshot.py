"""Save composed HAR frames when activity is logged (for timeline alert thumbnails)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.vision.har.overlay import compose_live_frame
from vision_ops_backend.vision.store import save_alert_snapshot

logger = logging.getLogger(__name__)


def capture_har_trigger_snapshot(
    frames_bgr: list[np.ndarray],
    *,
    model_id: str,
    model_label: str,
    prediction: dict[str, Any],
    show_heatmap: bool,
    show_boxes: bool,
) -> str | None:
    """Render the live overlay stack and persist a JPEG for timeline evidence."""
    if not frames_bgr:
        return None
    try:
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        rendered = compose_live_frame(
            frames_bgr[-1],
            frames_rgb=frames_rgb,
            model_label=model_label,
            prediction=prediction,
            model_id=model_id,
            inferring=False,
            show_heatmap=show_heatmap,
            show_boxes=show_boxes,
        )
        return save_alert_snapshot(rendered)
    except Exception as exc:
        logger.warning("HAR trigger snapshot failed: %s", exc)
        return None

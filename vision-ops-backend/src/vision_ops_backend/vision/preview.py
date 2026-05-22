"""Draw probe overlays on frames for Live tile previews."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _pct(value: str, total: int) -> int:
    s = str(value).strip().rstrip("%")
    try:
        return int(float(s) / 100.0 * total)
    except ValueError:
        return 0


def draw_overlay_on_frame(frame: np.ndarray, overlay: dict[str, Any]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x = _pct(overlay.get("left", "0%"), w)
    y = _pct(overlay.get("top", "0%"), h)
    bw = max(1, _pct(overlay.get("width", "10%"), w))
    bh = max(1, _pct(overlay.get("height", "10%"), h))

    variant = overlay.get("variant", "primary")
    if variant == "error":
        color = (68, 68, 239)  # BGR red-ish
    elif variant == "tertiary":
        color = (200, 200, 200)
    else:
        color = (76, 175, 80)

    cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
    label = str(overlay.get("label", ""))[:48]
    if label:
        cv2.putText(
            out,
            label,
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out

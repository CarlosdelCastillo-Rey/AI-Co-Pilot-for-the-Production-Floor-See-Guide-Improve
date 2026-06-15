"""Draw per-person boxes and action labels on BGR frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

TRACK_COLORS = [
    (76, 175, 80),
    (79, 195, 247),
    (255, 183, 77),
    (206, 147, 216),
    (240, 98, 146),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_tracks(frame_bgr: np.ndarray, tracks: list[dict[str, Any]]) -> np.ndarray:
    disp = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for tr in tracks:
        bbox = tr.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = map(int, bbox[:4])
        tid = int(tr.get("track_id", 0))
        color = _color(tid)
        inferring = bool(tr.get("inferring"))
        label = tr.get("action_label")
        conf = tr.get("action_confidence")
        det = float(tr.get("det_conf") or 0.0)

        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)
        if inferring:
            chip = f"#{tid} analyzing…"
        elif label and conf is not None:
            chip = f"#{tid} {str(label)[:24]} {float(conf):.0%}"
        else:
            chip = f"#{tid} Person {det:.0%}"

        (tw, th), _ = cv2.getTextSize(chip, font, 0.48, 1)
        y_top = max(0, y1 - th - 8)
        cv2.rectangle(disp, (x1, y_top), (x1 + tw + 10, y1), color, -1)
        cv2.putText(disp, chip, (x1 + 5, y1 - 5), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def draw_hud(
    frame_bgr: np.ndarray,
    *,
    model_name: str = "V-JEPA2+MLP",
    n_persons: int = 0,
    infer_ms: float | None = None,
    session_id: str = "",
) -> np.ndarray:
    disp = frame_bgr.copy()
    h, w = disp.shape[:2]
    lines = [f"Per-person HAR · {model_name}", f"Tracks: {n_persons}"]
    if infer_ms is not None:
        lines.append(f"Infer: {infer_ms:.0f} ms")
    if session_id:
        lines.append(f"Session: {session_id[:12]}")
    y = h - 12 - 18 * (len(lines) - 1)
    for line in lines:
        cv2.rectangle(disp, (0, y - 16), (min(w, 420), y + 4), (20, 20, 20), -1)
        cv2.putText(disp, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y += 18
    return disp

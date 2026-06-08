"""Draw HAR prediction overlays on video frames (Avance 4 notebook style)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from vision_ops_backend.vision.har.constants import HAR_MODEL_COLORS

logger = logging.getLogger(__name__)

_yolo_model: Any = None
_yolo_failed = False

# BGR — green activity box when action is assigned to a person
ACTION_BOX_BGR = (76, 175, 80)
PERSON_BOX_BGR = (79, 195, 247)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def build_motion_heatmap(frames_rgb: list[np.ndarray], out_h: int, out_w: int) -> np.ndarray | None:
    if len(frames_rgb) < 2:
        return None
    prev = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(frames_rgb[-2], (out_w, out_h)), cv2.COLOR_RGB2GRAY).astype(np.float32),
        (0, 0),
        3,
    )
    curr = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(frames_rgb[-1], (out_w, out_h)), cv2.COLOR_RGB2GRAY).astype(np.float32),
        (0, 0),
        3,
    )
    diff = cv2.GaussianBlur(cv2.absdiff(prev, curr), (0, 0), 9)
    mx = float(diff.max())
    if mx < 1e-6:
        return None
    return (diff / mx).astype(np.float32)


def detect_person_boxes(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    global _yolo_model, _yolo_failed
    if _yolo_failed:
        return []
    if _yolo_model is None:
        try:
            from ultralytics import YOLO

            _yolo_model = YOLO("yolov8n.pt")
        except Exception as exc:
            logger.info("YOLO not available for live overlay: %s", exc)
            _yolo_failed = True
            return []
    try:
        res = _yolo_model(frame_bgr, classes=[0], verbose=False)[0]
        out: list[tuple[int, int, int, int, float]] = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append((x1, y1, x2, y2, float(box.conf[0])))
        return out
    except Exception as exc:
        logger.debug("YOLO inference failed: %s", exc)
        return []


def _draw_label_chip(
    disp: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    bg_bgr: tuple[int, int, int],
    text_bgr: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.52,
) -> int:
    """Draw filled label above point (x, y); returns chip height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    y_top = max(0, y - th - 8)
    cv2.rectangle(disp, (x, y_top), (x + tw + 10, y), bg_bgr, -1)
    cv2.putText(disp, text, (x + 5, y - 5), font, font_scale, text_bgr, 1, cv2.LINE_AA)
    return th + 8


def draw_person_boxes_with_action(
    frame_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    *,
    action_label: str | None = None,
    action_confidence: float | None = None,
    model_id: str = "",
    per_person: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Person detections with optional HAR action label on each box (green when action present)."""
    if per_person:
        return draw_tracked_person_boxes(frame_bgr, per_person, model_id=model_id)

    disp = frame_bgr.copy()
    has_action = bool(action_label and action_confidence is not None)
    action_color = _hex_to_bgr(HAR_MODEL_COLORS.get(model_id, "#81C784")) if model_id else ACTION_BOX_BGR
    border_color = action_color if has_action else PERSON_BOX_BGR

    for x1, y1, x2, y2, det_conf in boxes:
        cv2.rectangle(disp, (x1, y1), (x2, y2), border_color, 3)

        if has_action:
            act_txt = f"{action_label[:26]}  {action_confidence:.0%}"
            _draw_label_chip(disp, act_txt, x1, y1, bg_bgr=action_color, text_bgr=(255, 255, 255))
            sub = f"person {det_conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(sub, font, 0.38, 1)
            y2_lbl = min(disp.shape[0] - 2, y2 + th + 6)
            cv2.rectangle(disp, (x1, y2), (x1 + tw + 8, y2_lbl), PERSON_BOX_BGR, -1)
            cv2.putText(disp, sub, (x1 + 4, y2 + th + 2), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            lbl = f"Person {det_conf:.0%}"
            _draw_label_chip(disp, lbl, x1, y1, bg_bgr=PERSON_BOX_BGR, text_bgr=(10, 10, 30))

    return disp


def draw_tracked_person_boxes(
    frame_bgr: np.ndarray,
    tracks: list[dict[str, Any]],
    *,
    model_id: str = "",
) -> np.ndarray:
    """One action label per ByteTrack id."""
    disp = frame_bgr.copy()
    action_color = _hex_to_bgr(HAR_MODEL_COLORS.get(model_id, "#81C784")) if model_id else ACTION_BOX_BGR
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tr in tracks:
        bbox = tr.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = map(int, bbox[:4])
        det_conf = float(tr.get("det_conf") or 0.0)
        tid = tr.get("track_id", "?")
        action_label = tr.get("action_label")
        action_conf = tr.get("action_confidence")
        inferring = bool(tr.get("inferring"))
        has_action = bool(action_label and action_conf is not None and not inferring)
        border_color = action_color if has_action else PERSON_BOX_BGR
        cv2.rectangle(disp, (x1, y1), (x2, y2), border_color, 3)

        if inferring:
            _draw_label_chip(disp, f"#{tid} analyzing…", x1, y1, bg_bgr=(60, 60, 60), text_bgr=(220, 220, 220))
        elif has_action:
            act_txt = f"#{tid} {str(action_label)[:22]}  {float(action_conf):.0%}"
            _draw_label_chip(disp, act_txt, x1, y1, bg_bgr=action_color, text_bgr=(255, 255, 255))
            sub = f"person {det_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(sub, font, 0.38, 1)
            y2_lbl = min(disp.shape[0] - 2, y2 + th + 6)
            cv2.rectangle(disp, (x1, y2), (x1 + tw + 8, y2_lbl), PERSON_BOX_BGR, -1)
            cv2.putText(disp, sub, (x1 + 4, y2 + th + 2), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            lbl = f"#{tid} Person {det_conf:.0%}"
            _draw_label_chip(disp, lbl, x1, y1, bg_bgr=PERSON_BOX_BGR, text_bgr=(10, 10, 30))

    return disp


def compose_per_person_live_frame(
    frame_bgr: np.ndarray,
    *,
    frames_rgb: list[np.ndarray],
    model_label: str,
    track_predictions: list[dict[str, Any]],
    model_id: str,
    inferring: bool = False,
    show_heatmap: bool = True,
    show_boxes: bool = True,
    summary_prediction: dict[str, Any] | None = None,
) -> np.ndarray:
    """Per-track HAR overlay (heatmap optional, one label per person)."""
    h, w = frame_bgr.shape[:2]
    heatmap = build_motion_heatmap(frames_rgb, h, w) if show_heatmap else None
    disp = apply_motion_heatmap(frame_bgr, heatmap)

    if show_boxes and track_predictions:
        disp = draw_tracked_person_boxes(disp, track_predictions, model_id=model_id)
        x1_p = min(int(t["bbox"][0]) for t in track_predictions)
        y1_p = min(int(t["bbox"][1]) for t in track_predictions)
        anchor = (max(8, x1_p), max(8, y1_p - 48))
    else:
        anchor = (8, 8)

    banner_pred = summary_prediction
    if banner_pred is None and track_predictions:
        best = max(
            (t for t in track_predictions if t.get("action_label")),
            key=lambda t: float(t.get("action_confidence") or 0),
            default=None,
        )
        if best:
            banner_pred = {
                "label": best.get("action_label"),
                "confidence": best.get("action_confidence"),
            }

    mode_label = f"{model_label} · per-person"
    return draw_activity_banner(
        disp,
        model_label=mode_label,
        prediction=banner_pred,
        model_id=model_id,
        inferring=inferring,
        anchor_xy=anchor,
    )


def apply_motion_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray | None, alpha: float = 0.38) -> np.ndarray:
    if heatmap is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_c = cv2.cvtColor(
        cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    hm_bgr = cv2.cvtColor(hm_c, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, hm_bgr, alpha, 0)


def draw_activity_banner(
    frame_bgr: np.ndarray,
    *,
    model_label: str,
    prediction: dict[str, Any] | None,
    model_id: str,
    inferring: bool = False,
    anchor_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """Model + activity chip anchored above person or top-left."""
    disp = frame_bgr.copy()
    h, w = disp.shape[:2]
    box_w = min(300, int(w * 0.62))
    box_h = 40
    bx = anchor_xy[0] if anchor_xy else 8
    by = anchor_xy[1] if anchor_xy else 8
    bx = max(8, min(bx, w - box_w - 8))
    by = max(8, min(by, h - box_h - 8))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if inferring:
        cv2.rectangle(disp, (bx, by), (bx + 220, by + 30), (26, 26, 13), -1)
        cv2.rectangle(disp, (bx, by), (bx + 220, by + 30), (120, 120, 80), 1)
        cv2.putText(disp, "Analyzing activity…", (bx + 8, by + 22), font, 0.48, (200, 200, 220), 1)
        return disp

    if not prediction:
        return disp

    label = str(prediction.get("label", "—"))[:32]
    conf = float(prediction.get("confidence") or 0.0)
    color = _hex_to_bgr(HAR_MODEL_COLORS.get(model_id, "#81C784"))

    overlay = disp.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (26, 26, 13), -1)
    cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)
    cv2.rectangle(disp, (bx, by), (bx + box_w, by + box_h), color, 2)

    bar_w = int((box_w - 4) * conf)
    cv2.rectangle(disp, (bx + 2, by + box_h - 5), (bx + 2 + bar_w, by + box_h - 2), color, -1)

    short = model_label.replace("DINOv2 →", "DINO→").replace("V-JEPA2 + MCJEPA", "VJEPA2")[:22]
    cv2.putText(disp, f"ACTION · {short}", (bx + 6, by + 14), font, 0.38, (180, 180, 180), 1)
    cv2.putText(disp, f"{label}  {conf:.0%}", (bx + 6, by + 32), font, 0.46, color, 1)
    return disp


def compose_live_frame(
    frame_bgr: np.ndarray,
    *,
    frames_rgb: list[np.ndarray],
    model_label: str,
    prediction: dict[str, Any] | None,
    model_id: str,
    inferring: bool = False,
    show_heatmap: bool = True,
    show_boxes: bool = True,
) -> np.ndarray:
    """Heatmap + person boxes with action labels + summary banner."""
    h, w = frame_bgr.shape[:2]
    heatmap = build_motion_heatmap(frames_rgb, h, w) if show_heatmap else None
    disp = apply_motion_heatmap(frame_bgr, heatmap)

    boxes = detect_person_boxes(frame_bgr) if show_boxes else []
    action_label = None
    action_conf: float | None = None
    if prediction and not inferring:
        action_label = str(prediction.get("label", ""))
        action_conf = float(prediction.get("confidence") or 0.0)

    if show_boxes and boxes:
        disp = draw_person_boxes_with_action(
            disp,
            boxes,
            action_label=action_label,
            action_confidence=action_conf,
            model_id=model_id,
        )
        x1_p = min(b[0] for b in boxes)
        y1_p = min(b[1] for b in boxes)
        anchor = (max(8, x1_p), max(8, y1_p - 48))
    else:
        anchor = (8, 8)

    return draw_activity_banner(
        disp,
        model_label=model_label,
        prediction=prediction,
        model_id=model_id,
        inferring=inferring,
        anchor_xy=anchor,
    )

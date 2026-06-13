"""v2 crop extraction — 3-view InHARD sub-frame splitting + YOLO person crops."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lib.constants import (
    CROP_SIZE,
    DEFAULT_BBOX_PADDING,
    DEFAULT_BUFFER_FRAMES,
    DEFAULT_INHARD_VIEW,
    INHARD_VIEW_FRONT,
    INHARD_VIEW_FULL,
    INHARD_VIEW_SIDE,
    INHARD_VIEW_TOPDOWN,
)
from lib.embeddings import extract_clip_from_file
from lib.tracking import crop_person, _get_yolo


# ── InHARD 3-view extraction ───────────────────────────────────────────────────

def extract_inhard_view(frame_bgr: np.ndarray, view: str = DEFAULT_INHARD_VIEW) -> np.ndarray:
    """Extract a single sub-view from an InHARD 1280×720 3-view composite frame.

    Layout:
        top-left  [  0:360,    0:640] → overhead / top-down (best for direction)
        top-right [  0:360,  640:   ] → side / eye-level
        bot-right [360:   ,  640:   ] → front / low-angle
        full                          → pass through unchanged
    """
    if view == INHARD_VIEW_FULL:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    half_h, half_w = h // 2, w // 2
    if view == INHARD_VIEW_TOPDOWN:
        return frame_bgr[:half_h, :half_w]
    if view == INHARD_VIEW_SIDE:
        return frame_bgr[:half_h, half_w:]
    if view == INHARD_VIEW_FRONT:
        return frame_bgr[half_h:, half_w:]
    return frame_bgr


def extract_all_views(frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        INHARD_VIEW_TOPDOWN: extract_inhard_view(frame_bgr, INHARD_VIEW_TOPDOWN),
        INHARD_VIEW_SIDE:    extract_inhard_view(frame_bgr, INHARD_VIEW_SIDE),
        INHARD_VIEW_FRONT:   extract_inhard_view(frame_bgr, INHARD_VIEW_FRONT),
    }


# ── YOLO person crop ──────────────────────────────────────────────────────────

def _largest_person_box(frame_bgr: np.ndarray):
    model = _get_yolo()
    if model is None:
        return None
    try:
        res = model(frame_bgr, classes=[0], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        best, best_area = None, 0
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, float(box.conf[0]))
        return best
    except Exception:
        return None


def _center_crop(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return cv2.resize(frame_bgr[y0:y0+side, x0:x0+side], (CROP_SIZE, CROP_SIZE))


def frame_to_person_crop(
    frame_bgr: np.ndarray,
    *,
    view: str = DEFAULT_INHARD_VIEW,
    padding: float = DEFAULT_BBOX_PADDING,
) -> tuple[np.ndarray, tuple | None]:
    """Extract sub-view then YOLO-crop the person. Returns (crop, bbox_or_None)."""
    sub = extract_inhard_view(frame_bgr, view)
    det = _largest_person_box(sub)
    if det is None:
        return _center_crop(sub), None
    x1, y1, x2, y2, conf = det
    crop = crop_person(sub, x1, y1, x2, y2, padding=padding)
    if crop is None:
        return _center_crop(sub), None
    return crop, (x1, y1, x2, y2, conf)


# ── Crop buffer builders ──────────────────────────────────────────────────────

def build_crop_buffer(
    frames_bgr: list[np.ndarray],
    *,
    view: str = DEFAULT_INHARD_VIEW,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    padding: float = DEFAULT_BBOX_PADDING,
) -> tuple[list[np.ndarray], list[tuple | None]]:
    """Build frame crop buffer + per-frame bbox list (for visualization)."""
    buf_crops: list[np.ndarray] = []
    buf_bboxes: list[tuple | None] = []
    for frame in frames_bgr:
        crop, bbox = frame_to_person_crop(frame, view=view, padding=padding)
        buf_crops.append(crop)
        buf_bboxes.append(bbox)
        if len(buf_crops) > buffer_frames:
            buf_crops  = buf_crops[-buffer_frames:]
            buf_bboxes = buf_bboxes[-buffer_frames:]
    return buf_crops, buf_bboxes


def crops_for_embedding(
    clip_path: Path | str,
    *,
    mode: str = "yolo_crop",
    view: str = DEFAULT_INHARD_VIEW,
    max_read: int = 64,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
) -> list[np.ndarray]:
    """Return frame list for backbone embedding."""
    frames = extract_clip_from_file(clip_path, max_frames=max_read)
    if not frames:
        return []
    if mode == "full_clip":
        return frames
    crops, _ = build_crop_buffer(frames, view=view, buffer_frames=buffer_frames)
    return crops


def crops_for_embedding_with_meta(
    clip_path: Path | str,
    *,
    mode: str = "yolo_crop",
    view: str = DEFAULT_INHARD_VIEW,
    max_read: int = 64,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
) -> dict:
    """Like crops_for_embedding but also returns raw frames and bbox info for viz."""
    frames = extract_clip_from_file(clip_path, max_frames=max_read)
    if not frames:
        return {"crops": [], "raw_frames": [], "bboxes": [], "subviews": []}
    subviews = [extract_inhard_view(f, view) for f in frames]
    if mode == "full_clip":
        return {"crops": frames, "raw_frames": frames, "bboxes": [], "subviews": subviews}
    crops, bboxes = build_crop_buffer(frames, view=view, buffer_frames=buffer_frames)
    return {"crops": crops, "raw_frames": frames, "bboxes": bboxes, "subviews": subviews}


def crops_from_jpeg(path: Path | str, view: str = DEFAULT_INHARD_VIEW) -> list[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return []
    crop, _ = frame_to_person_crop(img, view=view)
    return [crop] * DEFAULT_BUFFER_FRAMES


# ── Debug visualization ───────────────────────────────────────────────────────

def _draw_bbox(subview_bgr: np.ndarray, bbox: tuple | None) -> np.ndarray:
    img = subview_bgr.copy()
    if bbox is None:
        return img
    x1, y1, x2, y2, conf = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_crop_extraction(
    clip_path: Path | str,
    *,
    view: str = DEFAULT_INHARD_VIEW,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    max_read: int = 64,
    frame_idx: int | None = None,
    save_path: Path | str | None = None,
    title: str | None = None,
):
    """Show mosaic → sub-view (+ YOLO box) → 16 sampled person crops fed to backbones.

    Returns matplotlib Figure. Saves PNG when save_path is set.
    """
    import matplotlib.pyplot as plt
    from lib.constants import NUM_FRAMES
    from lib.embeddings import sample_frames

    meta = crops_for_embedding_with_meta(
        clip_path, view=view, max_read=max_read, buffer_frames=buffer_frames,
    )
    raw = meta["raw_frames"]
    crops = meta["crops"]
    if not raw or not crops:
        raise ValueError(f"No frames read from {clip_path}")

    idx = frame_idx if frame_idx is not None else len(raw) // 2
    idx = max(0, min(idx, len(raw) - 1))
    bbox = meta["bboxes"][idx] if idx < len(meta["bboxes"]) else None
    subview = meta["subviews"][idx]
    sampled = sample_frames(crops, n=NUM_FRAMES)

    clip_name = Path(clip_path).name
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.28)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.imshow(_bgr_to_rgb(raw[idx]))
    ax_top.set_title(f"Raw InHARD mosaic  (frame {idx}/{len(raw) - 1})", fontsize=10)
    ax_top.axis("off")

    ax_mid = ax_top.inset_axes([0.02, 0.52, 0.38, 0.46])
    ax_mid.imshow(_bgr_to_rgb(_draw_bbox(subview, bbox)))
    det = "YOLO" if bbox else "center fallback"
    ax_mid.set_title(f"{view} sub-view + {det}", fontsize=8)
    ax_mid.axis("off")

    ax_bot = fig.add_subplot(gs[1, 0])
    strip = np.concatenate([_bgr_to_rgb(c) for c in sampled], axis=1)
    ax_bot.imshow(strip)
    ax_bot.set_title(
        f"16 sampled person crops ({NUM_FRAMES}×{CROP_SIZE}px) → backbone input  "
        f"[buffer={len(crops)} frames]",
        fontsize=10,
    )
    ax_bot.axis("off")

    fig.suptitle(title or clip_name, fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def save_crop_debug_samples(
    clips,
    out_dir: Path | str,
    *,
    n_samples: int = 12,
    one_per_class: bool = True,
    view: str = DEFAULT_INHARD_VIEW,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    seed: int = 42,
) -> list[Path]:
    """Save crop-pipeline PNGs for inspection. Returns paths written."""
    import random

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = list(clips)
    rng.shuffle(pool)

    picked = []
    seen: set[str] = set()
    for rec in pool:
        if one_per_class and rec.label in seen:
            continue
        picked.append(rec)
        seen.add(rec.label)
        if len(picked) >= n_samples:
            break

    paths: list[Path] = []
    for rec in picked:
        safe_label = rec.label.replace(" ", "_").replace("/", "-")
        path = out / f"{safe_label}__{rec.path.stem}.png"
        visualize_crop_extraction(
            rec.path,
            view=view,
            buffer_frames=buffer_frames,
            save_path=path,
            title=f"{rec.label}  ·  {rec.path.name}",
        )
        import matplotlib.pyplot as plt
        plt.close("all")
        paths.append(path)
    return paths

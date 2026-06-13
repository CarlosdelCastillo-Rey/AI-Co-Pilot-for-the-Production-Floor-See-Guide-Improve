"""Notebook-friendly video playback (OpenCV decode → matplotlib JS player)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def transcode_for_web(src: Path, *, force: bool = False) -> Path:
    """H.264 + yuv420p for browser <video> tags. Returns src if ffmpeg missing."""
    src = Path(src)
    dst = src.with_name(f"{src.stem}_web.mp4")
    if (
        not force
        and dst.is_file()
        and dst.stat().st_mtime >= src.stat().st_mtime
        and dst.stat().st_size > 0
    ):
        return dst
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return src
    return dst if dst.is_file() and dst.stat().st_size > 0 else src


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_bgr(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    nh = max(1, int(h * max_width / w))
    return cv2.resize(frame_bgr, (max_width, nh), interpolation=cv2.INTER_AREA)


def _bgr_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _html_track_list(tracks: list[dict], frame_i: int) -> str:
    if not tracks:
        return f"<p><i>Frame {frame_i}: no YOLO person detections</i></p>"
    rows = []
    for tr in sorted(tracks, key=lambda t: int(t.get("track_id", 0))):
        tid = tr.get("track_id")
        det = float(tr.get("det_conf") or 0)
        if tr.get("inferring"):
            action = "<span style='color:#FF9800'>analyzing…</span>"
        elif tr.get("action_label"):
            conf = float(tr.get("action_confidence") or 0)
            action = f"<b>{tr['action_label']}</b> {conf:.0%}"
        else:
            action = f"<span style='color:#888'>Person detected {det:.0%}</span>"
        bb = tr.get("bbox") or [0, 0, 0, 0]
        rows.append(
            f"<tr><td><b>#{tid}</b></td>"
            f"<td>{action}</td>"
            f"<td>{det:.0%}</td>"
            f"<td style='font-size:10px'>{bb}</td></tr>"
        )
    return (
        f"<p><b>Frame {frame_i}</b> · {len(tracks)} tracked person(s)</p>"
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<tr style='background:#333;color:#fff'>"
        "<th>Track</th><th>Registered action</th><th>YOLO</th><th>BBox</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _html_track_stats(stats: dict[int, dict]) -> str:
    if not stats:
        return "<p><i>No track statistics yet</i></p>"
    blocks = []
    for tid in sorted(stats):
        s = stats[tid]
        counts = s.get("action_counts") or {}
        total = sum(counts.values()) or 1
        bars = []
        for lbl, n in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * n / total
            bars.append(
                f"<tr><td>{lbl}</td><td>{n}</td>"
                f"<td><div style='background:#4CAF50;height:10px;width:{pct:.0f}%'></div></td></tr>"
            )
        cur = s.get("current_label") or "—"
        cur_c = s.get("current_conf") or 0
        gid = s.get("global_person_id")
        gid_line = (
            f"<br>Person ID: <code>{gid}</code>"
            + (f" (match {float(s.get('reid_match_score') or 0):.0%})" if gid else "")
        )
        blocks.append(
            f"<div style='margin-bottom:12px;padding:8px;border:1px solid #ddd;border-radius:6px'>"
            f"<b>Track #{tid}</b> · {s.get('n_inferences', 0)} inferences · "
            f"frames {s.get('first_frame', '?')}–{s.get('last_frame', '?')}"
            f"{gid_line}<br>"
            f"Current: <code>{cur}</code> ({cur_c:.0%})"
            "<table style='width:100%;font-size:12px;margin-top:6px'>"
            "<tr><th>Action</th><th>Count</th><th>Share</th></tr>"
            + "".join(bars)
            + "</table></div>"
        )
    return "<h4>Per-track action statistics (all inferences)</h4>" + "".join(blocks)


def launch_eval_dashboard(
    video_path: Path | str,
    checkpoint: Path | str,
    *,
    width: int = 720,
    max_frames: int = 600,
    infer_every: int = 16,
    buffer_frames: int = 32,
    dwell_windows: int = 2,
    min_confidence: float = 0.25,
    log_session: bool = True,
):
    """Interactive UI: YOLO detection view + tracked persons + action stats per ID."""
    import time
    from collections import Counter, defaultdict

    import ipywidgets as widgets
    from IPython.display import clear_output, display
    from tqdm.auto import tqdm

    from lib.eval_video import iter_annotated_frames

    video_path = Path(video_path)
    checkpoint = Path(checkpoint)

    frames_jpeg: list[bytes] = []
    frame_tracks: list[list[dict]] = []
    frame_indices: list[int] = []
    track_stats: dict[int, dict] = defaultdict(
        lambda: {
            "action_counts": Counter(),
            "n_inferences": 0,
            "first_frame": None,
            "last_frame": None,
            "current_label": None,
            "current_conf": 0.0,
        }
    )
    session_dir = ""

    print(f"Building eval dashboard for {video_path.name} …")
    for rendered, meta in tqdm(
        iter_annotated_frames(
            video_path=video_path,
            checkpoint=checkpoint,
            max_frames=max_frames,
            infer_every=infer_every,
            buffer_frames=buffer_frames,
            dwell_windows=dwell_windows,
            min_confidence=min_confidence,
            log_session=log_session,
        ),
        desc="Processing",
    ):
        if rendered is None:
            session_dir = meta.get("session_dir", "")
            break
        fi = meta["frame_i"]
        tracks = meta.get("tracks") or []
        small = _resize_bgr(rendered, width)
        frames_jpeg.append(_bgr_to_jpeg_bytes(small))
        frame_tracks.append(tracks)
        frame_indices.append(fi)

        for tr in tracks:
            tid = int(tr["track_id"])
            st = track_stats[tid]
            if st["first_frame"] is None:
                st["first_frame"] = fi
            st["last_frame"] = fi
            if tr.get("action_label"):
                st["current_label"] = tr["action_label"]
                st["current_conf"] = float(tr.get("action_confidence") or 0)

        for inf in meta.get("frame_inferences") or []:
            tid = int(inf["track_id"])
            st = track_stats[tid]
            st["n_inferences"] += 1
            lbl = str(inf.get("raw_label") or "uncertain")
            st["action_counts"][lbl] += 1
            if inf.get("global_person_id"):
                st["global_person_id"] = inf["global_person_id"]
                st["reid_match_score"] = inf.get("reid_match_score")

    if not frames_jpeg:
        print("No frames processed.")
        return None

    print(f"  {len(frames_jpeg)} frames · {len(track_stats)} unique tracks")
    if session_dir:
        print(f"  Session log: {session_dir}")

    img = widgets.Image(format="jpg", width=width)
    track_panel = widgets.HTML()
    stats_panel = widgets.HTML(value=_html_track_stats(dict(track_stats)))
    frame_slider = widgets.IntSlider(
        value=0, min=0, max=len(frames_jpeg) - 1, step=1,
        description="Frame", continuous_update=True,
        layout=widgets.Layout(width=f"{width + 40}px"),
    )
    frame_label = widgets.HTML()
    play_btn = widgets.ToggleButton(value=False, description="▶ Play", icon="play")
    speed = widgets.Dropdown(
        options=[("0.25×", 4.0), ("0.5×", 2.0), ("1×", 1.0), ("2×", 0.5)],
        value=1.0, description="Speed",
    )

    def _show(idx: int) -> None:
        idx = max(0, min(idx, len(frames_jpeg) - 1))
        img.value = frames_jpeg[idx]
        fi = frame_indices[idx]
        track_panel.value = _html_track_list(frame_tracks[idx], fi)
        frame_label.value = f"<b>Frame {fi}</b> / {frame_indices[-1]} · idx {idx + 1}/{len(frames_jpeg)}"

    def _on_slider(change) -> None:
        _show(change["new"])

    frame_slider.observe(_on_slider, names="value")

    import threading
    _stop = threading.Event()

    def _play_loop() -> None:
        cap_fps = 25.0
        while not _stop.is_set():
            if not play_btn.value:
                break
            nxt = frame_slider.value + 1
            if nxt > frame_slider.max:
                nxt = 0
            frame_slider.value = nxt
            time.sleep(max(0.02, (1.0 / cap_fps) * float(speed.value)))

    def _on_play(change) -> None:
        if change["new"]:
            _stop.clear()
            threading.Thread(target=_play_loop, daemon=True).start()
        else:
            _stop.set()

    play_btn.observe(_on_play, names="value")

    _show(0)
    ui = widgets.VBox([
        widgets.HTML(
            "<h3>Live eval — YOLO detection + per-person actions</h3>"
            "<p>Green boxes = YOLO person tracks. Label chip = registered HAR action after dwell.</p>"
        ),
        widgets.HBox([
            widgets.VBox([img, widgets.HBox([play_btn, speed]), frame_slider, frame_label]),
            widgets.VBox([widgets.HTML("<b>Tracked persons (this frame)</b>"), track_panel], layout=widgets.Layout(width="420px")),
        ]),
        stats_panel,
    ])
    clear_output(wait=True)
    display(ui)
    return None


def play_inference_notebook(
    video_path: Path | str,
    checkpoint: Path | str,
    *,
    width: int = 960,
    max_frames: int = 600,
    infer_every: int = 16,
    buffer_frames: int = 32,
    dwell_windows: int = 2,
    min_confidence: float = 0.25,
    log_session: bool = True,
    save_crop_every_event: bool = True,
):
    """Run YOLO + HAR inference and play annotated frames inline (boxes + action labels)."""
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from tqdm.auto import tqdm

    from lib.eval_video import iter_annotated_frames

    video_path = Path(video_path)
    checkpoint = Path(checkpoint)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    rgb_frames: list[np.ndarray] = []
    max_tracks = 0
    print(f"Running inference on {video_path.name} (YOLO tracks + HAR labels) …")

    for rendered, meta in tqdm(
        iter_annotated_frames(
            video_path=video_path,
            checkpoint=checkpoint,
            max_frames=max_frames,
            infer_every=infer_every,
            buffer_frames=buffer_frames,
            dwell_windows=dwell_windows,
            min_confidence=min_confidence,
            log_session=log_session,
            save_crop_every_event=save_crop_every_event,
        ),
        desc="Annotating frames",
    ):
        if rendered is None:
            break
        max_tracks = max(max_tracks, meta.get("n_tracks", 0))
        small = _resize_bgr(rendered, width)
        rgb_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    if not rgb_frames:
        raise RuntimeError(f"No frames produced from {video_path}")

    print(f"  {len(rgb_frames)} frames · max concurrent tracks: {max_tracks}")
    if max_tracks == 0:
        print("  ⚠ No persons detected — check YOLO weights / video content.")

    vh, vw = rgb_frames[0].shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, max(4.0, vh / 100)))
    ax.axis("off")
    im = ax.imshow(rgb_frames[0])

    def _update(n: int):
        im.set_data(rgb_frames[n])
        return [im]

    interval_ms = 1000.0 / max(fps, 1.0)
    anim = animation.FuncAnimation(
        fig, _update, frames=len(rgb_frames), interval=interval_ms, blit=True, repeat=True,
    )
    html = anim.to_jshtml()
    plt.close(fig)
    return display(HTML(html))


def play_video_notebook(
    path: Path | str,
    *,
    width: int = 960,
    max_frames: int | None = None,
):
    """Inline JS video player — decodes with OpenCV (works with mp4v eval exports).

    Returns IPython.display.HTML display object.
    """
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        fc = min(fc, max_frames)
    if fc < 1:
        cap.release()
        raise RuntimeError(f"No frames in {path}")

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = vh / max(vw, 1)
    fig_h = max(4.0, width * aspect / 100)

    fig, ax = plt.subplots(figsize=(width / 100, fig_h))
    ax.axis("off")
    frame0 = _read_frame(cap, 0)
    im = ax.imshow(frame0 if frame0 is not None else np.zeros((vh, vw, 3), dtype=np.uint8))

    def _update(n: int):
        frame = _read_frame(cap, n)
        if frame is not None:
            im.set_data(frame)
        return [im]

    interval_ms = 1000.0 / max(fps, 1.0)
    anim = animation.FuncAnimation(
        fig, _update, frames=fc, interval=interval_ms, blit=True, repeat=True,
    )
    html = anim.to_jshtml()
    cap.release()
    plt.close(fig)
    return display(HTML(html))


def embed_video_html5(path: Path | str, *, width: int = 960):
    """Browser <video> tag — needs H.264 (auto-transcodes via ffmpeg when available)."""
    from IPython.display import Video, display

    path = Path(path)
    web_path = transcode_for_web(path)
    if web_path == path:
        print(
            "Note: source uses mp4v — browsers often cannot play it. "
            "Using matplotlib player instead, or install ffmpeg for H.264 transcode."
        )
        return play_video_notebook(path, width=width)
    return display(Video(str(web_path), embed=True, width=width, html_attributes="controls"))

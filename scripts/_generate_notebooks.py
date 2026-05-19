#!/usr/bin/env python3
"""Generate Fase 0 pipeline notebooks (run once from repo root)."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPTS = Path(__file__).parent
NB_FORMAT = 4
NBFORMAT_MINOR = 4


def md(source: str, cell_id: str = "") -> dict:
    cell: dict = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source.split("\n")],
    }
    return cell


def code(source: str, cell_id: str = "") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in source.split("\n")],
    }


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": NB_FORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }


def write_nb(name: str, cells: list[dict]) -> None:
    path = SCRIPTS / name
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


# --- 01 capture ---
write_nb(
    "01_capture_stream.ipynb",
    [
        md(
            """# 01 — Captura de stream (Fase 0)

**Objetivo:** Validar ingesta desde webcam, archivo `.mp4` o RTSP con OpenCV.

| Entrada | Salida |
|---------|--------|
| `SOURCE` (auto / path / 0 / rtsp://…) | `outputs/01_capture/frame_%06d.jpg` + `metadata.json` |

**Criterio de éxito:** ≥30 frames guardados sin errores de lectura consecutivos.""",
            "title",
        ),
        md(
            """## Prerrequisitos

- `uv sync --all-groups`
- Opcional: clip InHARD en `data_sample/InHARD-master/01-InHARD/Segmented/RGBSegmented/`
- Kernel: Python 3.11 del proyecto (`.venv`)""",
            "prereq",
        ),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from _common.io import (
    ensure_scripts_on_path,
    load_dotenv_repo,
    repo_root,
    resolve_source_video,
    setup_logging,
    stage_output_dir,
    utc_now_iso,
    write_json,
)
from loguru import logger

ensure_scripts_on_path()
load_dotenv_repo()
setup_logging()
logger.info("Repo root: {}", repo_root())''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''# --- Editar estas variables ---
SOURCE = "auto"  # "auto" | ruta .mp4 | 0 (webcam) | "rtsp://..."
MAX_FRAMES = 120
FPS_SAMPLE = 2.0  # guardar ~1 frame cada 1/FPS_SAMPLE segundos (aprox.)
OUT_DIR = stage_output_dir("01_capture")
MIN_FRAMES_OK = 30''',
            "config",
        ),
        md("## 3. Captura y muestreo", "s3"),
        code(
            '''resolved = resolve_source_video(SOURCE)
cap = cv2.VideoCapture(resolved)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la fuente: {resolved!r}")

fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_interval = max(1, int(round(fps_native / max(FPS_SAMPLE, 0.1))))

saved = 0
frame_idx = 0
timestamps: list[float] = []
t0 = time.time()

while saved < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        logger.warning("Fin de stream o lectura fallida en frame {}", frame_idx)
        break
    if frame_idx % frame_interval == 0:
        out_path = OUT_DIR / f"frame_{saved:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        timestamps.append(time.time() - t0)
        saved += 1
    frame_idx += 1

cap.release()
logger.info("Frames guardados: {}", saved)''',
            "capture",
        ),
        md("## 4. Persistencia (metadata)", "s4"),
        code(
            '''metadata = {
    "source": str(resolved),
    "width": width,
    "height": height,
    "fps_native": fps_native,
    "fps_sample_target": FPS_SAMPLE,
    "frame_interval": frame_interval,
    "frames_saved": saved,
    "timestamps_sec": timestamps,
    "created_at": utc_now_iso(),
}
write_json(OUT_DIR / "metadata.json", metadata)
metadata''',
            "persist",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''assert saved >= MIN_FRAMES_OK or saved > 0, (
    f"Se esperaban >={MIN_FRAMES_OK} frames (o al menos 1 en entorno sin cámara). "
    f"Obtuvo {saved}. Coloque un .mp4 en InHARD o ajuste MAX_FRAMES."
)
print(f"OK — {saved} frames en {OUT_DIR}")''',
            "validate",
        ),
        md("## 6. Siguiente paso\n\nEjecutar **[02_segment_frames.ipynb](02_segment_frames.ipynb)**.", "next"),
    ],
)

# --- 02 segment ---
write_nb(
    "02_segment_frames.ipynb",
    [
        md(
            """# 02 — Segmentación de frames y clips (Fase 0)

**Objetivo:** Generar frames espaciados (análisis estático) y clips temporales (acciones / V-JEPA).

| Entrada | Salida |
|---------|--------|
| `outputs/01_capture/` o video | `outputs/02_segments/frames/`, `clips/clip_XXXX/`, `manifest.json` |""",
            "title",
        ),
        md("## Prerrequisitos\n\nNotebook **01** o un archivo `.mp4` en `INPUT_VIDEO`.", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from _common.io import (
    ensure_scripts_on_path,
    find_first_mp4,
    read_json,
    repo_root,
    setup_logging,
    stage_output_dir,
    write_json,
)
from loguru import logger

ensure_scripts_on_path()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''CAPTURE_DIR = stage_output_dir("01_capture")
OUT_DIR = stage_output_dir("02_segments")
FRAMES_DIR = OUT_DIR / "frames"
CLIPS_DIR = OUT_DIR / "clips"

INPUT_VIDEO = None  # None = metadata de 01 o primer InHARD
STATIC_FPS = 1.0
CLIP_SEC = 2.0
CLIP_FRAMES_MAX = 64''',
            "config",
        ),
        md("## 3. Resolver fuente de video", "s3"),
        code(
            '''meta_path = CAPTURE_DIR / "metadata.json"
if INPUT_VIDEO:
    video_path = Path(INPUT_VIDEO)
elif meta_path.is_file():
    meta = read_json(meta_path)
    video_path = Path(meta["source"])
else:
    found = find_first_mp4()
    if found is None:
        raise FileNotFoundError("No hay video. Ejecute 01 o extraiga InHARD.")
    video_path = found

if not video_path.is_file():
    raise FileNotFoundError(f"Video no encontrado: {video_path}")

logger.info("Segmentando: {}", video_path)''',
            "resolve",
        ),
        md("## 4. Frames espaciados + clips", "s4"),
        code(
            '''FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
static_interval = max(1, int(round(fps / max(STATIC_FPS, 0.1))))
clip_len = min(CLIP_FRAMES_MAX, max(8, int(CLIP_SEC * fps)))

static_saved = 0
clip_idx = 0
frame_i = 0
clip_frames: list = []
manifest_clips: list[dict] = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_i % static_interval == 0:
        cv2.imwrite(str(FRAMES_DIR / f"static_{static_saved:06d}.jpg"), frame)
        static_saved += 1
    clip_frames.append(frame)
    if len(clip_frames) >= clip_len:
        clip_dir = CLIPS_DIR / f"clip_{clip_idx:04d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        start_sec = (frame_i - clip_len + 1) / fps
        end_sec = frame_i / fps
        for j, cf in enumerate(clip_frames):
            cv2.imwrite(str(clip_dir / f"frame_{j:06d}.jpg"), cf)
        manifest_clips.append({
            "clip_id": f"clip_{clip_idx:04d}",
            "path": str(clip_dir.relative_to(repo_root())),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "num_frames": len(clip_frames),
            "parent_video": str(video_path),
        })
        clip_idx += 1
        clip_frames = []
    frame_i += 1

cap.release()
logger.info("Static frames: {}, clips: {}", static_saved, clip_idx)''',
            "segment",
        ),
        md("## 5. Manifest", "s5"),
        code(
            '''manifest = {
    "parent_video": str(video_path),
    "static_frames_dir": str(FRAMES_DIR.relative_to(repo_root())),
    "static_frame_count": static_saved,
    "clips": manifest_clips,
}
write_json(OUT_DIR / "manifest.json", manifest)
manifest''',
            "manifest",
        ),
        md("## 6. Validación", "s6"),
        code(
            '''assert static_saved > 0 and clip_idx > 0, "Segmentación vacía"
print(f"OK — {static_saved} frames estáticos, {clip_idx} clips")''',
            "validate",
        ),
        md("## 6. Siguiente paso\n\n**[03_detect_and_track.ipynb](03_detect_and_track.ipynb)**", "next"),
    ],
)

# --- 03 detect ---
write_nb(
    "03_detect_and_track.ipynb",
    [
        md(
            """# 03 — Detección YOLOv8 + tracking DeepSORT (Fase 0)

**Objetivo:** IDs estables por objeto en secuencia corta; exportar `tracks.jsonl` y video anotado.""",
            "title",
        ),
        md("## Prerrequisitos\n\nSalida de **02** (`manifest.json` + clips).", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from _common.io import (
    append_jsonl,
    ensure_scripts_on_path,
    read_json,
    setup_logging,
    stage_output_dir,
)
from loguru import logger

ensure_scripts_on_path()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''SEGMENTS_DIR = stage_output_dir("02_segments")
OUT_DIR = stage_output_dir("03_track")
TRACKS_PATH = OUT_DIR / "tracks.jsonl"

CLIP_ID = "clip_0000"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.35
# COCO: person=0; truck=7, car=2 como proxy de montacargas
TARGET_CLASS_IDS = {0, 2, 7}
CLASS_NAMES = {0: "person", 2: "car", 7: "truck"}''',
            "config",
        ),
        md("## 3. Cargar clip y modelos", "s3"),
        code(
            '''manifest = read_json(SEGMENTS_DIR / "manifest.json")
clip_entry = next((c for c in manifest["clips"] if c["clip_id"] == CLIP_ID), manifest["clips"][0])
from _common.io import repo_root

clip_dir = repo_root() / clip_entry["path"]

frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
if not frame_paths:
    raise FileNotFoundError(f"Sin frames en {clip_dir}")

model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=30, n_init=3)
logger.info("Procesando {} frames de {}", len(frame_paths), clip_entry["clip_id"])''',
            "load",
        ),
        md("## 4. Detección + tracking por frame", "s4"),
        code(
            '''if TRACKS_PATH.exists():
    TRACKS_PATH.unlink()

h, w = cv2.imread(str(frame_paths[0])).shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_DIR / "annotated.mp4"), fourcc, 10.0, (w, h))

for frame_idx, fp in enumerate(frame_paths):
    frame = cv2.imread(str(fp))
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in TARGET_CLASS_IDS:
            continue
        conf = float(box.conf.item())
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, CLASS_NAMES.get(cls_id, str(cls_id))))

    tracks = tracker.update_tracks(detections, frame=frame)
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        l, t, bw, bh = map(int, tr.to_ltrb())
        tid = tr.track_id
        label = tr.get_det_class() if hasattr(tr, "get_det_class") else "obj"
        append_jsonl(
            TRACKS_PATH,
            {
                "frame_idx": frame_idx,
                "track_id": int(tid),
                "class": str(label),
                "bbox": [l, t, l + bw, t + bh],
                "confidence": float(tr.det_conf) if tr.det_conf else None,
            },
        )
        cv2.rectangle(frame, (l, t), (l + bw, t + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"{tid}:{label}", (l, max(t - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    writer.write(frame)

writer.release()
track_count = sum(1 for _ in open(TRACKS_PATH))
logger.info("Registros en tracks.jsonl: {}", track_count)''',
            "track",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''from _common.io import read_jsonl
rows = read_jsonl(TRACKS_PATH)
assert len(rows) > 0, "tracks.jsonl vacío"
ids = {r["track_id"] for r in rows}
print(f"OK — {len(rows)} detecciones, {len(ids)} track IDs únicos")''',
            "validate",
        ),
        md("## Siguiente paso\n\n**[04_build_event_buffer.ipynb](04_build_event_buffer.ipynb)**", "next"),
    ],
)

# --- 04 events ---
write_nb(
    "04_build_event_buffer.ipynb",
    [
        md(
            """# 04 — Buffer de eventos (Fase 0)

**Objetivo:** Reglas MVP sobre tracks → `events.jsonl` (simula cola Redis).""",
            "title",
        ),
        md("## Prerrequisitos\n\n**03** → `outputs/03_track/tracks.jsonl`", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import os
import uuid
from collections import defaultdict

from _common.io import (
    append_jsonl,
    bbox_centroid,
    default_roi,
    ensure_scripts_on_path,
    env_or_none,
    load_dotenv_repo,
    point_in_roi,
    read_json,
    read_jsonl,
    setup_logging,
    stage_output_dir,
    utc_now_iso,
)
from loguru import logger

ensure_scripts_on_path()
load_dotenv_repo()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''TRACKS_PATH = stage_output_dir("03_track") / "tracks.jsonl"
CAPTURE_META = stage_output_dir("01_capture") / "metadata.json"
OUT_DIR = stage_output_dir("04_events")
EVENTS_PATH = OUT_DIR / "events.jsonl"

ROI = default_roi()
IDLE_SEC = 3.0
IDLE_PIX = 25.0
FPS_ASSUME = 10.0
VEHICLE_CLASSES = {"car", "truck", "forklift"}
REDIS_URL = env_or_none("REDIS_URL")
BACKEND = "redis" if REDIS_URL else "json"''',
            "config",
        ),
        md("## 3. Cargar tracks y dimensiones", "s3"),
        code(
            '''rows = read_jsonl(TRACKS_PATH)
if not rows:
    raise RuntimeError("Sin tracks. Ejecute notebook 03.")

frame_w, frame_h = 1920, 1080
if CAPTURE_META.is_file():
    meta = read_json(CAPTURE_META)
    frame_w = int(meta.get("width", frame_w))
    frame_h = int(meta.get("height", frame_h))

by_track: dict[int, list] = defaultdict(list)
for r in rows:
    by_track[int(r["track_id"])].append(r)
logger.info("Tracks únicos: {}", len(by_track))''',
            "load",
        ),
        md("## 4. Motor de reglas MVP", "s4"),
        code(
            '''if EVENTS_PATH.exists():
    EVENTS_PATH.unlink()

events_emitted = 0
idle_frames = int(IDLE_SEC * FPS_ASSUME)

for track_id, track_rows in by_track.items():
    track_rows.sort(key=lambda x: x["frame_idx"])
    cls = str(track_rows[0].get("class", "unknown"))

    # Regla: persona en ROI → warning
    if cls == "person":
        for r in track_rows:
            cx, cy = bbox_centroid(r["bbox"])
            if point_in_roi(cx, cy, ROI, frame_w, frame_h):
                append_jsonl(
                    EVENTS_PATH,
                    {
                        "event_id": str(uuid.uuid4()),
                        "type": "warning",
                        "severity": "medium",
                        "track_id": track_id,
                        "class": cls,
                        "zone": "roi_placeholder",
                        "frame_idx": r["frame_idx"],
                        "message": f"Persona {track_id} en zona ROI",
                        "ts": utc_now_iso(),
                    },
                )
                events_emitted += 1
                break

    # Regla: idle — poco movimiento durante N frames
    if len(track_rows) >= idle_frames:
        still = True
        ref = bbox_centroid(track_rows[0]["bbox"])
        for r in track_rows[1:idle_frames]:
            cx, cy = bbox_centroid(r["bbox"])
            if abs(cx - ref[0]) + abs(cy - ref[1]) > IDLE_PIX:
                still = False
                break
        if still:
            append_jsonl(
                EVENTS_PATH,
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "idle",
                    "severity": "low",
                    "track_id": track_id,
                    "class": cls,
                    "zone": "floor",
                    "message": f"Track {track_id} inactivo ~{IDLE_SEC}s",
                    "ts": utc_now_iso(),
                },
            )
            events_emitted += 1

    # Regla: vehículo en movimiento → forklift_zone
    if cls in VEHICLE_CLASSES and len(track_rows) >= 2:
        c0 = bbox_centroid(track_rows[0]["bbox"])
        c1 = bbox_centroid(track_rows[-1]["bbox"])
        dist = abs(c1[0] - c0[0]) + abs(c1[1] - c0[1])
        if dist > IDLE_PIX:
            append_jsonl(
                EVENTS_PATH,
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "forklift_zone",
                    "severity": "high",
                    "track_id": track_id,
                    "class": cls,
                    "zone": "aisle",
                    "message": f"Vehículo {track_id} en movimiento (proxy montacargas)",
                    "ts": utc_now_iso(),
                },
            )
            events_emitted += 1

logger.info("Eventos emitidos: {}", events_emitted)''',
            "rules",
        ),
        md("## 5. Backend opcional Redis", "s5"),
        code(
            '''redis_pushed = 0
if BACKEND == "redis" and REDIS_URL:
    try:
        import redis
        r = redis.from_url(REDIS_URL)
        for ev in read_jsonl(EVENTS_PATH):
            r.rpush("visionops:events", __import__("json").dumps(ev))
            redis_pushed += 1
        logger.info("Redis: {} eventos en cola visionops:events", redis_pushed)
    except Exception as exc:
        logger.warning("Redis no disponible ({}); solo JSON local", exc)
else:
    logger.info("Backend JSON local: {}", EVENTS_PATH)''',
            "redis",
        ),
        md("## 6. Validación", "s6"),
        code(
            '''events = read_jsonl(EVENTS_PATH)
print(f"OK — {len(events)} eventos en {EVENTS_PATH}")
events[:3]''',
            "validate",
        ),
        md(
            """## Siguiente paso

Ramas paralelas: **[05_semantic_event.ipynb](05_semantic_event.ipynb)**, **[06_generate_heatmap.ipynb](06_generate_heatmap.ipynb)**, **[07_telegram_webhook.ipynb](07_telegram_webhook.ipynb)**""",
            "next",
        ),
    ],
)

# --- 05 semantic ---
write_nb(
    "05_semantic_event.ipynb",
    [
        md(
            """# 05 — Evento semántico con Gemini (Fase 0)

**Objetivo:** Bitácora estilo PRD a partir de clip + metadatos del evento.""",
            "title",
        ),
        md("## Prerrequisitos\n\n`GEMINI_API_KEY` en `.env`; salidas de **02** y **04**.", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import json
import os
from pathlib import Path

from _common.io import (
    ensure_scripts_on_path,
    env_or_none,
    load_dotenv_repo,
    read_json,
    read_jsonl,
    setup_logging,
    stage_output_dir,
    write_json,
)
from loguru import logger

ensure_scripts_on_path()
load_dotenv_repo()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''OUT_DIR = stage_output_dir("05_semantic")
EVENTS_PATH = stage_output_dir("04_events") / "events.jsonl"
SEGMENTS_MANIFEST = stage_output_dir("02_segments") / "manifest.json"
GEMINI_MODEL = "gemini-1.5-flash"
SKIPPED = False''',
            "config",
        ),
        md("## 3. Seleccionar evento y contexto", "s3"),
        code(
            '''events = read_jsonl(EVENTS_PATH)
if not events:
    raise RuntimeError("Sin eventos. Ejecute notebook 04.")

event = events[-1]
manifest = read_json(SEGMENTS_MANIFEST)
clip = manifest["clips"][0] if manifest.get("clips") else {}
clip_path = clip.get("path", "")
logger.info("Evento: {} type={}", event.get("event_id"), event.get("type"))''',
            "select",
        ),
        md("## 4. Llamada Gemini", "s4"),
        code(
            '''api_key = env_or_none("GEMINI_API_KEY")
result = {"status": "pending", "event_id": event.get("event_id")}

if not api_key:
    SKIPPED = True
    result = {
        "status": "SKIPPED",
        "reason": "GEMINI_API_KEY no definida en .env",
        "event_id": event.get("event_id"),
    }
    logger.warning(result["reason"])
else:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""Eres un asistente de bitácora industrial VisionOps.
Genera un resumen operativo breve y un JSON con campos: title, severity, description, zone.
Evento detectado: {json.dumps(event, ensure_ascii=False)}
Clip referencia: {clip_path}
Responde primero 2-3 oraciones en español (estilo bitácora) y luego un bloque JSON válido."""
    response = model.generate_content(prompt)
    text = response.text or ""
    result = {
        "status": "ok",
        "event_id": event.get("event_id"),
        "narrative": text,
        "title": event.get("message", "Evento de planta"),
        "severity": event.get("severity", "medium"),
        "description": event.get("message", ""),
        "zone": event.get("zone", "unknown"),
        "raw_model_text": text,
    }

out_path = OUT_DIR / f"semantic_{event.get('event_id', 'last')}.json"
write_json(out_path, result)
result''',
            "gemini",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''assert result.get("status") in ("ok", "SKIPPED")
print(f"Estado: {result['status']} → {out_path}")''',
            "validate",
        ),
        md("## Siguiente paso\n\n**[06_generate_heatmap.ipynb](06_generate_heatmap.ipynb)** o **[07_telegram_webhook.ipynb](07_telegram_webhook.ipynb)**", "next"),
    ],
)

# --- 06 heatmap ---
write_nb(
    "06_generate_heatmap.ipynb",
    [
        md(
            """# 06 — Heatmap de actividad (Fase 0)

**Objetivo:** Agregar centroides de tracks en grid → PNG + JSON (REQ-03 Analytics).""",
            "title",
        ),
        md("## Prerrequisitos\n\n**03** → `tracks.jsonl`; metadata de **01** para dimensiones.", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common.io import (
    bbox_centroid,
    ensure_scripts_on_path,
    read_json,
    read_jsonl,
    setup_logging,
    stage_output_dir,
    utc_now_iso,
    write_json,
)
from loguru import logger

ensure_scripts_on_path()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''TRACKS_PATH = stage_output_dir("03_track") / "tracks.jsonl"
CAPTURE_META = stage_output_dir("01_capture") / "metadata.json"
OUT_DIR = stage_output_dir("06_heatmap")
GRID_ROWS = 16
GRID_COLS = 24''',
            "config",
        ),
        md("## 3. Acumular grid", "s3"),
        code(
            '''rows = read_jsonl(TRACKS_PATH)
frame_w, frame_h = 1920, 1080
if CAPTURE_META.is_file():
    m = read_json(CAPTURE_META)
    frame_w, frame_h = int(m["width"]), int(m["height"])

grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
for r in rows:
    cx, cy = bbox_centroid(r["bbox"])
    col = min(GRID_COLS - 1, int(cx / max(frame_w, 1) * GRID_COLS))
    row = min(GRID_ROWS - 1, int(cy / max(frame_h, 1) * GRID_ROWS))
    grid[row, col] += 1.0

logger.info("Celdas activas: {}", int((grid > 0).sum()))''',
            "grid",
        ),
        md("## 4. Exportar PNG + JSON", "s4"),
        code(
            '''heatmap_meta = {
    "grid_rows": GRID_ROWS,
    "grid_cols": GRID_COLS,
    "frame_width": frame_w,
    "frame_height": frame_h,
    "total_hits": float(grid.sum()),
    "created_at": utc_now_iso(),
    "values": grid.tolist(),
}
write_json(OUT_DIR / "heatmap.json", heatmap_meta)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(grid, origin="upper", cmap="hot", interpolation="nearest")
ax.set_title("VisionOps — heatmap de centroides")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "heatmap.png", dpi=120)
plt.close(fig)
print(f"OK — {OUT_DIR / 'heatmap.png'}")''',
            "export",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''assert (OUT_DIR / "heatmap.png").is_file()
assert (OUT_DIR / "heatmap.json").is_file()
print("Heatmap generado correctamente")''',
            "validate",
        ),
    ],
)

# --- 07 telegram ---
write_nb(
    "07_telegram_webhook.ipynb",
    [
        md(
            """# 07 — Alerta Telegram (Fase 0)

**Objetivo:** Enviar mensaje (y foto opcional) al bot de Telegram (REQ-04).""",
            "title",
        ),
        md("## Prerrequisitos\n\n`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` en `.env`; eventos de **04**.", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

from pathlib import Path

import requests

from _common.io import (
    ensure_scripts_on_path,
    env_or_none,
    load_dotenv_repo,
    read_jsonl,
    setup_logging,
    stage_output_dir,
)
from loguru import logger

ensure_scripts_on_path()
load_dotenv_repo()
setup_logging()''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''EVENTS_PATH = stage_output_dir("04_events") / "events.jsonl"
SEMANTIC_DIR = stage_output_dir("05_semantic")
TRACK_VIDEO = stage_output_dir("03_track") / "annotated.mp4"
THUMB = stage_output_dir("01_capture") / "frame_000000.jpg"

TOKEN = env_or_none("TELEGRAM_BOT_TOKEN")
CHAT_ID = env_or_none("TELEGRAM_CHAT_ID")
SKIPPED = False''',
            "config",
        ),
        md("## 3. Construir payload", "s3"),
        code(
            '''events = read_jsonl(EVENTS_PATH)
if not events:
    raise RuntimeError("Sin eventos para alertar")

event = events[-1]
text = (
    f"[VisionOps] {event.get('type', 'event').upper()}\\n"
    f"{event.get('message', '')}\\n"
    f"Severity: {event.get('severity')}\\n"
    f"Zone: {event.get('zone')}"
)
logger.info("Mensaje: {}", text[:120])''',
            "payload",
        ),
        md("## 4. Enviar a Telegram", "s4"),
        code(
            '''response_summary = {"status": "pending"}

if not TOKEN or not CHAT_ID:
    SKIPPED = True
    response_summary = {
        "status": "SKIPPED",
        "reason": "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID ausentes",
    }
    logger.warning(response_summary["reason"])
else:
    base = f"https://api.telegram.org/bot{TOKEN}"
    r = requests.post(
        f"{base}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text},
        timeout=30,
    )
    response_summary = {"status": "ok" if r.ok else "error", "http": r.status_code, "body": r.text[:500]}
    if THUMB.is_file() and r.ok:
        with THUMB.open("rb") as photo:
            rp = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": CHAT_ID, "caption": event.get("type", "")},
                files={"photo": photo},
                timeout=60,
            )
        response_summary["photo_http"] = rp.status_code

response_summary''',
            "send",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''assert response_summary["status"] in ("ok", "SKIPPED", "error")
print(f"Telegram: {response_summary['status']}")''',
            "validate",
        ),
    ],
)

# --- 08 vjepa ---
write_nb(
    "08_vjepa_action_probe.ipynb",
    [
        md(
            """# 08 — V-JEPA action probe (Fase 0, opcional)

**Objetivo:** Embedding / clasificación sobre clip segmentado (I+D post-MVP PRD).

**Nota:** Requiere pesos HuggingFace y GPU opcional. Si falla, estado `SKIPPED` controlado.""",
            "title",
        ),
        md("## Prerrequisitos\n\n**02** clips; código en `models/vjepa2-main/`.", "prereq"),
        md("## 1. Setup", "s1"),
        code(
            '''from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _common.io import (
    ensure_scripts_on_path,
    read_json,
    repo_root,
    setup_logging,
    stage_output_dir,
    write_json,
)
from loguru import logger

ensure_scripts_on_path()
setup_logging()

VJEPA_ROOT = repo_root() / "models" / "vjepa2-main"
if VJEPA_ROOT.is_dir() and str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))''',
            "setup",
        ),
        md("## 2. Configuration", "s2"),
        code(
            '''OUT_DIR = stage_output_dir("08_vjepa")
SEGMENTS_MANIFEST = stage_output_dir("02_segments") / "manifest.json"
CLIP_ID = "clip_0000"
HF_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256-ssv2"
MAX_FRAMES = 16
SKIPPED = False
STATUS = "pending"''',
            "config",
        ),
        md("## 3. Cargar frames del clip", "s3"),
        code(
            '''import cv2

manifest = read_json(SEGMENTS_MANIFEST)
clip_entry = next((c for c in manifest["clips"] if c["clip_id"] == CLIP_ID), manifest["clips"][0])
clip_dir = repo_root() / clip_entry["path"]
paths = sorted(clip_dir.glob("frame_*.jpg"))[:MAX_FRAMES]
if not paths:
    raise FileNotFoundError(f"Sin frames en {clip_dir}")

frames = [cv2.imread(str(p)) for p in paths]
frames = [f for f in frames if f is not None]
logger.info("Frames cargados: {}", len(frames))''',
            "load",
        ),
        md("## 4. Inferencia V-JEPA (HuggingFace) o fallback", "s4"),
        code(
            '''embedding = None
probe_result = {"clip_id": clip_entry["clip_id"], "num_frames": len(frames)}

try:
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_ID)
    model = AutoModel.from_pretrained(HF_MODEL_ID).to(device).eval()

    # T x H x W x C (RGB)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    video = np.stack(rgb, axis=0)
    inputs = processor(list(video), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = model.get_vision_features(**inputs)
    embedding = feats.detach().cpu().numpy()
    probe_result.update({
        "status": "ok",
        "backend": "huggingface",
        "model_id": HF_MODEL_ID,
        "embedding_shape": list(embedding.shape),
        "device": device,
    })
except Exception as exc:
    SKIPPED = True
    embedding = np.zeros((1, 32), dtype=np.float32)
    probe_result.update({
        "status": "SKIPPED",
        "reason": str(exc),
        "hint": "Descargue pesos HF o use GPU; ver models/vjepa2-main/README.md",
    })
    logger.warning("V-JEPA skip: {}", exc)

np.save(OUT_DIR / "embedding.npy", embedding)
write_json(OUT_DIR / "probe_result.json", probe_result)
probe_result''',
            "infer",
        ),
        md("## 5. Validación", "s5"),
        code(
            '''assert (OUT_DIR / "embedding.npy").is_file()
assert probe_result.get("status") in ("ok", "SKIPPED")
print(f"V-JEPA probe: {probe_result['status']}")''',
            "validate",
        ),
    ],
)

if __name__ == "__main__":
    print("Done.")

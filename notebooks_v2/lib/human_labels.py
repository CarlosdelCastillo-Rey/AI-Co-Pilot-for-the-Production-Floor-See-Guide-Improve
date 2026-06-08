"""Human-in-the-loop labels — CSV store + active-learning queue."""

from __future__ import annotations

import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from lib.constants import HITL_ACTION_VERDICTS, HITL_PERSON_VERDICTS, TRAINABLE_ACTIONS
from lib.paths import HUMAN_LABELS_DIR, SESSIONS_DIR, V1_SESSIONS_DIR

LABELS_CSV = HUMAN_LABELS_DIR / "labels.csv"
QUEUE_JSON = HUMAN_LABELS_DIR / "review_queue.json"

CSV_COLUMNS = [
    "label_id",
    "event_id",
    "session_dir",
    "crop_path",
    "frame_path",
    "embedding_path",
    "predicted_label",
    "confidence",
    "entropy",
    "priority_score",
    "action_verdict",
    "correct_label",
    "person_verdict",
    "usable_for_training",
    "reviewer",
    "reviewed_at",
    "source",
    "video",
    "track_id",
    "frame_idx",
    "notes",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entropy(probs: dict[str, float] | None) -> float:
    if not probs:
        return 0.0
    values = [max(0.0, float(v)) for v in probs.values()]
    total = sum(values)
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in values:
        p = v / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def priority_score(*, confidence: float, entropy: float) -> float:
    """Higher = review sooner (uncertain predictions)."""
    return float(entropy * 2.0 + (1.0 - confidence))


def ensure_labels_store() -> Path:
    HUMAN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if not LABELS_CSV.is_file():
        with LABELS_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
    return LABELS_CSV


def load_labels_df():
    import pandas as pd

    ensure_labels_store()
    return pd.read_csv(LABELS_CSV)


def reviewed_event_ids() -> set[str]:
    if not LABELS_CSV.is_file():
        return set()
    out: set[str] = set()
    with LABELS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row.get("event_id", "")
            if eid:
                out.add(eid)
    return out


def _session_roots(extra: list[Path] | None = None) -> list[Path]:
    roots = [SESSIONS_DIR]
    if V1_SESSIONS_DIR.is_dir():
        roots.append(V1_SESSIONS_DIR)
    if extra:
        roots.extend(extra)
    return roots


def iter_session_events(
    *,
    session_roots: list[Path] | None = None,
    include_unconfirmed: bool = True,
) -> Iterator[dict[str, Any]]:
    for root in _session_roots(session_roots):
        if not root.is_dir():
            continue
        for events_path in sorted(root.rglob("events.jsonl")):
            session_dir = events_path.parent
            manifest_path = session_dir / "manifest.json"
            manifest: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if not include_unconfirmed and not ev.get("confirmed_detection"):
                    continue
                ev["_session_dir"] = str(session_dir)
                ev["_sessions_root"] = str(root)
                ev["_manifest"] = manifest
                yield ev


def build_review_queue(
    *,
    limit: int = 200,
    session_roots: list[Path] | None = None,
    prefer_low_confidence: bool = True,
) -> list[dict[str, Any]]:
    seen = reviewed_event_ids()
    candidates: list[dict[str, Any]] = []

    for ev in iter_session_events(session_roots=session_roots, include_unconfirmed=True):
        eid = str(ev.get("event_id") or "")
        if not eid or eid in seen:
            continue
        crop_rel = ev.get("crop_path")
        frame_rel = ev.get("frame_path")
        if not crop_rel and not frame_rel:
            continue
        root = Path(ev["_sessions_root"])
        crop_path = root / crop_rel if crop_rel else None
        frame_path = root / frame_rel if frame_rel else None
        img_path = crop_path if crop_path and crop_path.is_file() else frame_path
        if img_path is None or not img_path.is_file():
            continue

        conf = float(ev.get("confidence") or 0.0)
        probs = ev.get("all_probs") or {}
        ent = _entropy(probs if isinstance(probs, dict) else None)
        score = priority_score(confidence=conf, entropy=ent)

        manifest = ev.get("_manifest") or {}
        candidates.append(
            {
                "event_id": eid,
                "session_dir": ev["_session_dir"],
                "sessions_root": ev["_sessions_root"],
                "image_path": str(img_path),
                "crop_path": str(crop_path) if crop_path else "",
                "frame_path": str(frame_path) if frame_path else "",
                "embedding_path": str(root / ev["embedding_path"])
                if ev.get("embedding_path")
                else "",
                "predicted_label": ev.get("action_label"),
                "confidence": conf,
                "entropy": ent,
                "priority_score": score,
                "top_k": ev.get("top_k"),
                "all_probs": probs,
                "track_id": ev.get("track_id"),
                "frame_idx": ev.get("frame_idx"),
                "video": manifest.get("video") or ev.get("video"),
                "model_tag": ev.get("model_tag"),
            }
        )

    if prefer_low_confidence:
        candidates.sort(key=lambda r: (-r["priority_score"], r.get("event_id", "")))
    return candidates[:limit]


def save_review_queue(path: Path | None = None, **kwargs: Any) -> Path:
    path = path or QUEUE_JSON
    queue = build_review_queue(**kwargs)
    payload = {"created_at": _utc_now(), "n_items": len(queue), "items": queue}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_label(row: dict[str, Any]) -> dict[str, Any]:
    ensure_labels_store()
    record = {col: row.get(col, "") for col in CSV_COLUMNS}
    if not record["label_id"]:
        record["label_id"] = f"hl-{uuid.uuid4().hex[:10]}"
    if not record["reviewed_at"]:
        record["reviewed_at"] = _utc_now()

    action_verdict = str(record.get("action_verdict") or "").lower()
    correct_label = str(record.get("correct_label") or "").strip()
    person_verdict = str(record.get("person_verdict") or "").lower()
    usable = record.get("usable_for_training")

    if usable in ("", None):
        trainable = action_verdict == "yes" or (
            action_verdict == "no" and correct_label in TRAINABLE_ACTIONS
        )
        is_person = person_verdict == "yes"
        record["usable_for_training"] = bool(
            trainable
            and is_person
            and action_verdict not in ("dont_know", "maybe")
        )

    with LABELS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(record)
    return record


def trainable_human_rows() -> list[dict[str, Any]]:
    if not LABELS_CSV.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with LABELS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            usable = str(row.get("usable_for_training", "")).lower()
            if usable not in ("true", "1", "yes"):
                continue
            verdict = str(row.get("action_verdict", "")).lower()
            label = str(row.get("correct_label") or "").strip()
            if verdict == "yes":
                label = str(row.get("predicted_label") or "").strip()
            if label not in TRAINABLE_ACTIONS:
                continue
            rows.append(dict(row))
    return rows


def load_human_embeddings_bundle(class_names: list[str]) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    """Load embeddings for human-verified rows (from .npy or re-embed later in pipeline)."""
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for row in trainable_human_rows():
        label = row.get("correct_label") or row.get("predicted_label")
        label = str(label or "").strip()
        if label not in label_to_idx:
            continue
        emb_path = str(row.get("embedding_path") or "").strip()
        if emb_path and Path(emb_path).is_file():
            emb = np.load(emb_path).astype(np.float32)
        else:
            meta_rows.append({**row, "_needs_reembed": True})
            continue
        X_list.append(emb)
        y_list.append(label_to_idx[label])
        meta_rows.append(dict(row))
    return X_list, y_list, meta_rows

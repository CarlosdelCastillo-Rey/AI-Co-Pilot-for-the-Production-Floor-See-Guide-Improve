"""Versioned local session logs — model, person, actions, frames, embeddings, confidence."""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lib.constants import VJEPA_MODEL_ID
from lib.paths import SESSIONS_DIR


def new_session_id() -> str:
    return f"har-{uuid.uuid4().hex[:12]}"


def model_tag_from_checkpoint(checkpoint: Path) -> str:
    """e.g. har_vjepa_all14_5each.pt → all14_5each"""
    stem = checkpoint.stem
    if stem.startswith("har_vjepa_"):
        return stem[len("har_vjepa_") :]
    if stem.startswith("har_"):
        return stem[len("har_") :]
    return stem


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class HarSessionLogger:
    """
    Persists a full inference audit trail under:

        outputs/har_sessions/YYYY-MM-DD/<run_id>_<model_tag>/
          manifest.json
          events.jsonl
          summary.json          (written on finalize)
          frames/               (JPEG on confirmed detections)
          embeddings/           (.npy per saved event)
    Also appends one row to outputs/har_sessions/index.csv for cross-run comparison.
    """

    INDEX_COLUMNS = [
        "run_id",
        "date",
        "model_tag",
        "checkpoint",
        "embedding_model",
        "classifier_version",
        "source",
        "video",
        "started_at",
        "ended_at",
        "n_events",
        "n_confirmed",
        "n_persons",
        "session_dir",
    ]

    def __init__(
        self,
        checkpoint: Path,
        *,
        predictor_info: dict[str, Any],
        source: str = "live",
        video_name: str | None = None,
        session_id: str | None = None,
        model_tag: str | None = None,
    ) -> None:
        self.session_id = session_id or new_session_id()
        self.checkpoint = checkpoint.resolve()
        self.model_tag = model_tag or model_tag_from_checkpoint(self.checkpoint)
        self.source = source
        self.video_name = video_name
        self.started_at = _utc_now()
        self._event_seq = 0
        self._n_events = 0
        self._n_confirmed = 0
        self._track_ids: set[int] = set()

        meta = dict(predictor_info.get("meta") or {})
        sidecar = self.checkpoint.with_suffix(".json")
        if sidecar.is_file():
            try:
                sc = json.loads(sidecar.read_text(encoding="utf-8"))
                meta = {
                    **sc.get("meta", {}),
                    "classifier_version": sc.get("classifier_version") or meta.get("classifier_version"),
                    "embedding_version": sc.get("embedding_version") or meta.get("embedding_version"),
                    **meta,
                }
            except json.JSONDecodeError:
                pass

        self.manifest: dict[str, Any] = {
            "session_id": self.session_id,
            "run_id": self.session_id,
            "started_at": self.started_at,
            "source": source,
            "video": video_name,
            "checkpoint": str(self.checkpoint),
            "checkpoint_name": self.checkpoint.name,
            "model_tag": self.model_tag,
            "classifier_version": meta.get("classifier_version") or self.model_tag,
            "embedding_model": VJEPA_MODEL_ID,
            "embedding_version": meta.get("embedding_version") or self.model_tag,
            "emb_dim": predictor_info.get("emb_dim"),
            "class_names": predictor_info.get("class_names", []),
            "exclude_labels": predictor_info.get("exclude_labels", []),
            "training_meta": meta.get("meta") or meta,
        }

        run_folder = f"{self.session_id}_{self.model_tag}"
        self.session_dir = SESSIONS_DIR / _date_folder() / run_folder
        self.frames_dir = self.session_dir / "frames"
        self.embeddings_dir = self.session_dir / "embeddings"
        for d in (self.session_dir, self.frames_dir, self.embeddings_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._events_path = self.session_dir / "events.jsonl"
        self._manifest_path = self.session_dir / "manifest.json"
        self._manifest_path.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def events_path(self) -> Path:
        return self._events_path

    def _next_event_id(self) -> str:
        self._event_seq += 1
        return f"evt_{self._event_seq:05d}"

    @staticmethod
    def _annotate_frame(
        frame_bgr: np.ndarray,
        *,
        track_id: int,
        bbox: list[int],
        label: str,
        confidence: float,
        top_k: list[dict[str, Any]],
        model_tag: str,
        frame_idx: int,
    ) -> np.ndarray:
        disp = frame_bgr.copy()
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(disp, (x1, y1), (x2, y2), (76, 175, 80), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        header = f"#{track_id} {label} {confidence:.1%}"
        cv2.putText(disp, header, (x1, max(20, y1 - 10)), font, 0.55, (76, 175, 80), 2, cv2.LINE_AA)
        y = 30
        for rank, item in enumerate(top_k[:5], start=1):
            line = f"{rank}. {item['label'][:22]} {float(item['prob']):.1%}"
            cv2.putText(disp, line, (10, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
        footer = f"f{frame_idx} · {model_tag} · {VJEPA_MODEL_ID.split('/')[-1]}"
        cv2.putText(disp, footer, (10, disp.shape[0] - 12), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return disp

    def log_inference(
        self,
        *,
        track_id: int,
        frame_idx: int,
        frame_bgr: np.ndarray | None,
        bbox: list[int],
        prediction: dict[str, Any],
        label_changed: bool = False,
        infer_ms: float | None = None,
        save_for_review: bool = False,
    ) -> dict[str, Any]:
        """Append JSONL event; save frame + embedding on confirm or uncertain (HITL queue)."""
        self._track_ids.add(track_id)
        self._n_events += 1
        event_id = self._next_event_id()

        frame_rel: str | None = None
        emb_rel: str | None = None
        crop_rel: str | None = None

        persist_artifacts = label_changed or save_for_review
        if label_changed:
            self._n_confirmed += 1
        if persist_artifacts:
            label_slug = (prediction.get("label") or "unknown").replace(" ", "_").lower()[:32]
            frame_name = f"f{frame_idx:05d}_tid{track_id}_{label_slug}.jpg"
            emb_name = f"{event_id}_tid{track_id}.npy"
            crop_name = f"{event_id}_tid{track_id}_crop.jpg"

            if frame_bgr is not None:
                annotated = self._annotate_frame(
                    frame_bgr,
                    track_id=track_id,
                    bbox=bbox,
                    label=str(prediction.get("label") or ""),
                    confidence=float(prediction.get("confidence") or 0.0),
                    top_k=list(prediction.get("top_k") or []),
                    model_tag=self.model_tag,
                    frame_idx=frame_idx,
                )
                frame_path = self.frames_dir / frame_name
                cv2.imwrite(str(frame_path), annotated)
                frame_rel = str(frame_path.relative_to(SESSIONS_DIR))

            emb = prediction.get("embedding")
            if isinstance(emb, np.ndarray):
                emb_path = self.embeddings_dir / emb_name
                np.save(emb_path, emb.astype(np.float32))
                emb_rel = str(emb_path.relative_to(SESSIONS_DIR))

            crop = prediction.get("crop_bgr")
            if isinstance(crop, np.ndarray):
                crop_path = self.frames_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                crop_rel = str(crop_path.relative_to(SESSIONS_DIR))

        row: dict[str, Any] = {
            "event_id": event_id,
            "ts": time.time(),
            "ts_iso": _utc_now(),
            "session_id": self.session_id,
            "track_id": track_id,
            "person_id": f"track_{track_id}",
            "frame_idx": frame_idx,
            "bbox": bbox,
            "action_label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "class_index": prediction.get("class_index"),
            "top_k": prediction.get("top_k"),
            "all_probs": prediction.get("all_probs"),
            "label_changed": label_changed,
            "confirmed_detection": label_changed,
            "uncertain": bool(prediction.get("uncertain")),
            "save_for_review": save_for_review,
            "raw_label": prediction.get("raw_label"),
            "raw_confidence": prediction.get("raw_confidence"),
            "infer_ms": round(infer_ms, 2) if infer_ms is not None else None,
            "model_tag": self.model_tag,
            "classifier_version": self.manifest["classifier_version"],
            "embedding_version": self.manifest["embedding_version"],
            "embedding_model": VJEPA_MODEL_ID,
            "checkpoint_name": self.checkpoint.name,
            "frame_path": frame_rel,
            "crop_path": crop_rel,
            "embedding_path": emb_rel,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def finalize(self) -> dict[str, Any]:
        ended_at = _utc_now()
        events = self.load_events()
        by_label: dict[str, int] = {}
        by_track: dict[int, int] = {}
        for ev in events:
            lbl = ev.get("action_label")
            if lbl:
                by_label[lbl] = by_label.get(lbl, 0) + 1
            tid = ev.get("track_id")
            if tid is not None:
                by_track[int(tid)] = by_track.get(int(tid), 0) + 1

        summary = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "n_events": self._n_events,
            "n_confirmed": self._n_confirmed,
            "n_persons": len(self._track_ids),
            "events_by_label": by_label,
            "events_by_track": {str(k): v for k, v in sorted(by_track.items())},
            "session_dir": str(self.session_dir),
        }
        summary_path = self.session_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        self._append_index_row(
            ended_at=ended_at,
            n_events=self._n_events,
            n_confirmed=self._n_confirmed,
            n_persons=len(self._track_ids),
        )
        return summary

    def _append_index_row(
        self,
        *,
        ended_at: str,
        n_events: int,
        n_confirmed: int,
        n_persons: int,
    ) -> None:
        index_path = SESSIONS_DIR / "index.csv"
        write_header = not index_path.is_file()
        row = {
            "run_id": self.session_id,
            "date": _date_folder(),
            "model_tag": self.model_tag,
            "checkpoint": self.checkpoint.name,
            "embedding_model": VJEPA_MODEL_ID,
            "classifier_version": self.manifest["classifier_version"],
            "source": self.source,
            "video": self.video_name or "",
            "started_at": self.started_at,
            "ended_at": ended_at,
            "n_events": n_events,
            "n_confirmed": n_confirmed,
            "n_persons": n_persons,
            "session_dir": str(self.session_dir.relative_to(SESSIONS_DIR)),
        }
        with index_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.INDEX_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_events(self) -> list[dict[str, Any]]:
        if not self._events_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self._events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


def list_sessions(*, model_tag: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Read index.csv for comparing runs iteration-over-iteration."""
    index_path = SESSIONS_DIR / "index.csv"
    if not index_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if model_tag and row.get("model_tag") != model_tag:
                continue
            rows.append(row)
    return rows[-limit:]

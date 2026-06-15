"""Incremental HAR MLP-head retraining from HITL labels.

Flow:
  1. Collect (embedding, label) pairs from HarHumanLabel + HarTrackLabel
  2. Merge with base InHARD embeddings from har-research/outputs/
  3. Retrain the lightweight MLP head (backbone stays frozen)
  4. Save checkpoint to har-research/checkpoints/ (hot-swap) + outputs/retrain/<ts>/
  5. Reset HarModelRegistry so the next inference call picks up new weights
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import HarHumanLabel, HarPerson, HarSessionEvent, HarTrackLabel
from vision_ops_alerting.db.session import SessionLocal

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# retrain.py lives at: <repo>/vision-ops-backend/src/vision_ops_alerting/services/retrain.py
# parents[4] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]

_CHECKPOINT_DIR = _REPO_ROOT / "har-research" / "checkpoints"
_HAR_RESEARCH_DIR = _REPO_ROOT / "har-research"
_BASE_EMBEDDINGS_DIR = _REPO_ROOT / "har-research" / "outputs"
_OUTPUTS_DIR = _REPO_ROOT / "outputs" / "retrain"

# Checkpoint name → NPZ file that has the base InHARD embeddings for that backbone
_BACKBONE_META: dict[str, dict] = {
    "vjepa": {
        "ckpt_key": "har_vjepa_12c_topdown_allavail",
        "npz": "embeddings.npz",
    },
    "dinov2": {
        "ckpt_key": "har_dinov2_12c_topdown_allavail",
        "npz": "embeddings_dinov2.npz",
    },
}


# ── Job state ─────────────────────────────────────────────────────────────────

@dataclass
class _RetrainJob:
    status: str = "idle"          # idle | running | done | failed
    started_at: str | None = None
    finished_at: str | None = None
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_job = _RetrainJob()
_lock = threading.Lock()


def job_state() -> dict:
    with _lock:
        return {
            "status": _job.status,
            "started_at": _job.started_at,
            "finished_at": _job.finished_at,
            "logs": list(_job.logs[-200:]),
            "result": dict(_job.result),
            "error": _job.error,
        }


def _log(msg: str) -> None:
    logger.info("[retrain] %s", msg)
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _lock:
        _job.logs.append(f"[{ts}] {msg}")


# ── HAR-research import helpers ───────────────────────────────────────────────

def _ensure_har_research_path() -> None:
    root = str(_HAR_RESEARCH_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Embedding collection ──────────────────────────────────────────────────────

def _resolve_embedding(session_id: str, rel_path: str) -> np.ndarray | None:
    from vision_ops_alerting.services.har_session_store import resolve_artifact_path

    path = resolve_artifact_path(session_id, rel_path)
    if path is None or not path.exists():
        return None
    try:
        return np.load(str(path)).astype(np.float32)
    except Exception:
        return None


def _collect_human_labels(db: Session) -> tuple[list[np.ndarray], list[str]]:
    """Collect crop-level HITL labels from HarHumanLabel (usable_for_training or correct_label set)."""
    rows = (
        db.query(HarHumanLabel)
        .filter(
            HarHumanLabel.correct_label.isnot(None),
            HarHumanLabel.event_id.isnot(None),
        )
        .all()
    )

    X, y = [], []
    for row in rows:
        label = (row.correct_label or "").strip()
        if not label:
            continue
        ev = db.query(HarSessionEvent).filter(HarSessionEvent.id == row.event_id).first()
        if ev is None or not ev.embedding_path:
            continue
        emb = _resolve_embedding(ev.session_id, ev.embedding_path)
        if emb is not None and emb.ndim == 1:
            X.append(emb)
            y.append(label)
    return X, y


def _collect_track_labels(db: Session) -> tuple[list[np.ndarray], list[str]]:
    """Collect track-level dominant actions from HarTrackLabel."""
    rows = (
        db.query(HarTrackLabel)
        .filter(HarTrackLabel.dominant_action.isnot(None))
        .all()
    )

    X, y = [], []
    for row in rows:
        label = (row.dominant_action or "").strip()
        if not label:
            continue
        events = (
            db.query(HarSessionEvent)
            .filter(
                HarSessionEvent.session_id == row.session_id,
                HarSessionEvent.track_id == row.track_id,
                HarSessionEvent.embedding_path.isnot(None),
            )
            .all()
        )
        for ev in events:
            emb = _resolve_embedding(ev.session_id, ev.embedding_path)
            if emb is not None and emb.ndim == 1:
                X.append(emb)
                y.append(label)
    return X, y


def collect_hitl_embeddings(db: Session) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Return (X, y, class_names) from all available HITL-labeled data."""
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    hl_X, hl_y = _collect_human_labels(db)
    X_list.extend(hl_X)
    y_list.extend(hl_y)

    tl_X, tl_y = _collect_track_labels(db)
    X_list.extend(tl_X)
    y_list.extend(tl_y)

    if not X_list:
        return None

    # Check all embeddings have the same dimension
    dims = {e.shape[0] for e in X_list}
    if len(dims) > 1:
        # Keep only the most common dimension
        common_dim = max(dims, key=lambda d: sum(1 for e in X_list if e.shape[0] == d))
        X_list = [e for e in X_list if e.shape[0] == common_dim]
        y_list = [y_list[i] for i, e in enumerate(X_list) if e.shape[0] == common_dim]

    all_classes = sorted(set(y_list))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    X = np.stack(X_list).astype(np.float32)
    y = np.array([class_to_idx[lbl] for lbl in y_list], dtype=np.int64)
    return X, y, all_classes


# ── Base InHARD embeddings ────────────────────────────────────────────────────

def _load_base_npz(backbone: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    meta = _BACKBONE_META.get(backbone)
    if meta is None:
        return None
    path = _BASE_EMBEDDINGS_DIR / meta["npz"]
    if not path.is_file():
        return None
    try:
        data = np.load(str(path), allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        class_names = list(data["class_names"])
        return X, y, class_names
    except Exception as exc:
        logger.warning("Could not load base NPZ %s: %s", path.name, exc)
        return None


# ── Merge ─────────────────────────────────────────────────────────────────────

def _merge(
    base: tuple[np.ndarray, np.ndarray, list[str]] | None,
    hitl: tuple[np.ndarray, np.ndarray, list[str]] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if base is None and hitl is None:
        raise ValueError("No training data — label some tracks in HITL first.")

    if base is None:
        X_h, y_h, cls_h = hitl
        return X_h, y_h, cls_h

    if hitl is None:
        return base

    X_b, y_b, cls_b = base
    X_h, y_h_local, cls_h = hitl

    # Filter HITL to same embedding dim as base
    if X_h.shape[1] != X_b.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: base={X_b.shape[1]}, HITL={X_h.shape[1]}. "
            "Make sure HITL sessions used the same backbone."
        )

    # Unified class list: base first, then any new HITL classes
    all_classes = list(cls_b)
    for c in cls_h:
        if c not in all_classes:
            all_classes.append(c)

    c2i = {c: i for i, c in enumerate(all_classes)}
    y_b_remap = np.array([c2i[cls_b[i]] for i in y_b], dtype=np.int64)
    y_h_remap = np.array([c2i[cls_h[i]] for i in y_h_local], dtype=np.int64)

    return np.vstack([X_b, X_h]), np.concatenate([y_b_remap, y_h_remap]), all_classes


# ── Base class names from checkpoint sidecar ─────────────────────────────────

def _base_class_names(backbone: str) -> list[str]:
    meta = _BACKBONE_META.get(backbone, _BACKBONE_META["vjepa"])
    sidecar = _CHECKPOINT_DIR / f"{meta['ckpt_key']}.json"
    if sidecar.is_file():
        try:
            import json
            return json.loads(sidecar.read_text())["class_names"]
        except Exception:
            pass
    return []


# ── Dataset stats (for UI) ────────────────────────────────────────────────────

def dataset_stats(db: Session) -> dict:
    sessions = db.query(HarSessionEvent.session_id).distinct().count()

    # Registered identities with confirmed names
    persons_rows = db.query(HarPerson).all()
    identities = [
        {
            "global_person_id": p.id,
            "display_name": p.display_name or "Unknown",
            "n_appearances": p.n_appearances,
            "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None,
        }
        for p in persons_rows
    ]

    # HITL human labels with correct_label
    human_label_rows = (
        db.query(HarHumanLabel.correct_label)
        .filter(HarHumanLabel.correct_label.isnot(None))
        .all()
    )
    human_label_counts: dict[str, int] = {}
    for (lbl,) in human_label_rows:
        human_label_counts[lbl] = human_label_counts.get(lbl, 0) + 1

    # HITL track labels with dominant_action
    track_label_rows = (
        db.query(HarTrackLabel.dominant_action)
        .filter(HarTrackLabel.dominant_action.isnot(None))
        .all()
    )
    track_label_counts: dict[str, int] = {}
    for (lbl,) in track_label_rows:
        track_label_counts[lbl] = track_label_counts.get(lbl, 0) + 1

    # Merge all HITL action labels
    all_hitl_labels: dict[str, int] = {}
    for lbl, cnt in human_label_counts.items():
        all_hitl_labels[lbl] = all_hitl_labels.get(lbl, 0) + cnt
    for lbl, cnt in track_label_counts.items():
        all_hitl_labels[lbl] = all_hitl_labels.get(lbl, 0) + cnt

    # Split known vs new classes (vs current checkpoint)
    base_classes = set(_base_class_names("vjepa"))
    known_actions = {k: v for k, v in all_hitl_labels.items() if k in base_classes}
    new_actions = {k: v for k, v in all_hitl_labels.items() if k not in base_classes}

    base_vjepa = (_BASE_EMBEDDINGS_DIR / "embeddings.npz").is_file()
    base_dino = (_BASE_EMBEDDINGS_DIR / "embeddings_dinov2.npz").is_file()

    with _lock:
        last_retrain = _job.finished_at
        last_status = _job.status if _job.status != "idle" else None
        last_result = dict(_job.result)

    total_hitl = sum(all_hitl_labels.values())

    return {
        "sessions": sessions,
        "persons": len(identities),
        "identities": identities,
        "hitl_human_labels": sum(human_label_counts.values()),
        "hitl_track_labels": sum(track_label_counts.values()),
        "total_hitl_samples": total_hitl,
        "action_labels": {
            "known": [{"label": k, "samples": v} for k, v in sorted(known_actions.items())],
            "new": [{"label": k, "samples": v} for k, v in sorted(new_actions.items())],
        },
        "base_class_names": sorted(base_classes),
        "base_embeddings_available": {"vjepa": base_vjepa, "dinov2": base_dino},
        "last_retrain_at": last_retrain,
        "last_retrain_status": last_status,
        "last_retrain_result": last_result,
    }


# ── Retrain job ───────────────────────────────────────────────────────────────

def _run_retrain(backbone: str, epochs: int) -> None:
    try:
        _ensure_har_research_path()
        from lib.har_model import save_checkpoint
        from lib.har_train import train_har_head

        meta = _BACKBONE_META.get(backbone, _BACKBONE_META["vjepa"])
        ckpt_key = meta["ckpt_key"]

        _log(f"Retrain started | backbone={backbone}  epochs={epochs}  ckpt={ckpt_key}")

        with SessionLocal() as db:
            _log("Collecting HITL-labeled embeddings from DB...")
            hitl = collect_hitl_embeddings(db)

        if hitl is not None:
            _log(f"  HITL: {len(hitl[0])} samples, {len(hitl[2])} classes → {hitl[2]}")
        else:
            _log("  No confirmed HITL labels found — using base InHARD embeddings only")

        _log("Loading base InHARD embeddings...")
        base = _load_base_npz(backbone)
        if base is not None:
            _log(f"  Base: {len(base[0])} samples, {len(base[2])} classes")
        else:
            _log("  Base NPZ not found (run notebook 02 first if needed)")

        X, y, class_names = _merge(base, hitl)
        emb_dim = int(X.shape[1])
        _log(f"Merged: {len(X)} samples  {len(class_names)} classes  emb_dim={emb_dim}")

        _log(f"Training MLP head ({epochs} epochs)...")
        model, stats = train_har_head(
            X, y, class_names,
            exclude_labels=[],
            epochs=epochs,
            head_arch="mlp",
            use_focal_loss=True,
            use_weighted_sampler=True,
            use_mixup=len(X) > 50,
        )
        val_acc: float = stats.get("val_report", {}).get("accuracy", 0.0)
        _log(f"Training done — val_acc={val_acc:.1%}")

        # Save to live checkpoint dir (hot-swap)
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = _CHECKPOINT_DIR / f"{ckpt_key}.pt"
        retrain_meta = {
            "epochs": epochs,
            "n_samples": len(X),
            "n_classes": len(class_names),
            "n_hitl_samples": len(hitl[0]) if hitl else 0,
            "backbone": backbone,
            "retrained_at": datetime.now(timezone.utc).isoformat(),
            "val_accuracy": round(val_acc, 4),
        }
        save_checkpoint(
            ckpt_path,
            model=model,
            class_names=class_names,
            exclude_labels=[],
            emb_dim=emb_dim,
            head_arch="mlp",
            meta=retrain_meta,
        )
        _log(f"Checkpoint saved → {ckpt_path}")

        # Archive to outputs/retrain/<timestamp>/
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_dir = _OUTPUTS_DIR / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out_dir / f"{ckpt_key}.pt",
            model=model,
            class_names=class_names,
            exclude_labels=[],
            emb_dim=emb_dim,
            head_arch="mlp",
            meta=retrain_meta,
        )
        np.savez_compressed(
            str(out_dir / "training_data.npz"),
            X=X,
            y=y,
            class_names=np.array(class_names),
        )
        _log(f"Archive saved → outputs/retrain/{ts}/")

        # Hot-reload the live inference registry
        _log("Hot-reloading model registry...")
        try:
            from vision_ops_backend.vision.har import checkpoints as ckpt_mod
            ckpt_mod._registry = None
            ckpt_mod.get_registry().ensure_loaded()
            _log("Model registry reloaded — new weights are live")
        except Exception as exc:
            _log(f"Registry reload warning: {exc} (restart backend to apply)")

        with _lock:
            _job.status = "done"
            _job.finished_at = datetime.now(timezone.utc).isoformat()
            _job.result = {
                "val_accuracy": round(val_acc, 4),
                "n_samples": len(X),
                "n_classes": len(class_names),
                "class_names": class_names,
                "n_hitl_samples": len(hitl[0]) if hitl else 0,
                "output_dir": str(out_dir),
                "backbone": backbone,
            }

    except Exception as exc:
        logger.exception("Retrain failed: %s", exc)
        _log(f"ERROR: {exc}")
        with _lock:
            _job.status = "failed"
            _job.finished_at = datetime.now(timezone.utc).isoformat()
            _job.error = str(exc)


def start_retrain(backbone: str = "vjepa", epochs: int = 30) -> bool:
    """Start retrain in background thread. Returns False if already running."""
    with _lock:
        if _job.status == "running":
            return False
        _job.status = "running"
        _job.started_at = datetime.now(timezone.utc).isoformat()
        _job.finished_at = None
        _job.logs = []
        _job.result = {}
        _job.error = None

    thread = threading.Thread(
        target=_run_retrain,
        args=(backbone, epochs),
        daemon=True,
        name="har-retrain",
    )
    thread.start()
    return True

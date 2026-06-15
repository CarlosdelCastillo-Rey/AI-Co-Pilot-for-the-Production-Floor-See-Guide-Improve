"""HAR action labels — model classes plus supervisor-expanded custom catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sqlalchemy import distinct
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHECKPOINTS_DIR = _REPO_ROOT / "har-research" / "checkpoints"
_CUSTOM_LABELS_PATH = _REPO_ROOT / "vision-ops-backend" / "data" / "har_custom_action_labels.json"

_DEFAULT_LABELS: tuple[str, ...] = (
    "Assemble system",
    "Consult sheets",
    "No action",
    "Picking in front",
    "Picking left",
    "Put down component",
    "Put down measuring rod",
    "Put down screwdriver",
    "Put down subsystem",
    "Take component",
    "Take measuring rod",
    "Take screwdriver",
    "Take subsystem",
    "Turn sheets",
)


@lru_cache(maxsize=1)
def registered_har_action_labels() -> tuple[str, ...]:
    """Action classes from all registered model checkpoint JSONs (merged, sorted)."""
    seen: dict[str, str] = {}
    if _CHECKPOINTS_DIR.is_dir():
        for ckpt_file in sorted(_CHECKPOINTS_DIR.glob("*.json")):
            try:
                data = json.loads(ckpt_file.read_text(encoding="utf-8"))
                names = data.get("class_names")
                if isinstance(names, list):
                    for n in names:
                        key = str(n).strip()
                        if key and key.casefold() not in seen:
                            seen[key.casefold()] = key
            except (json.JSONDecodeError, OSError):
                pass
    if seen:
        return tuple(sorted(seen.values(), key=str.lower))
    return _DEFAULT_LABELS


def _normalize_label(label: str) -> str:
    return " ".join(str(label or "").strip().split())


def _load_custom_labels_file() -> list[str]:
    if not _CUSTOM_LABELS_PATH.is_file():
        return []
    try:
        data = json.loads(_CUSTOM_LABELS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [_normalize_label(x) for x in data if _normalize_label(x)]
    if isinstance(data, dict):
        labels = data.get("labels")
        if isinstance(labels, list):
            return [_normalize_label(x) for x in labels if _normalize_label(x)]
    return []


def _save_custom_labels_file(labels: list[str]) -> None:
    _CUSTOM_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({_normalize_label(x) for x in labels if _normalize_label(x)}, key=str.lower)
    _CUSTOM_LABELS_PATH.write_text(
        json.dumps({"labels": ordered}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _custom_labels_from_db(db: Session | None) -> list[str]:
    if db is None:
        return []
    from vision_ops_alerting.db.models import HarHumanLabel

    rows = (
        db.query(distinct(HarHumanLabel.correct_label))
        .filter(HarHumanLabel.correct_label.isnot(None))
        .all()
    )
    out: list[str] = []
    for (raw,) in rows:
        label = _normalize_label(raw or "")
        if label:
            out.append(label)
    return out


def custom_har_action_labels(db: Session | None = None) -> tuple[str, ...]:
    """Supervisor-added labels (persisted file + prior HITL corrections)."""
    seen: dict[str, str] = {}
    for label in _load_custom_labels_file() + _custom_labels_from_db(db):
        key = label.casefold()
        if key not in seen:
            seen[key] = label
    model_keys = {n.casefold() for n in registered_har_action_labels()}
    return tuple(sorted((v for k, v in seen.items() if k not in model_keys), key=str.lower))


def expandable_har_action_labels(db: Session | None = None) -> tuple[str, ...]:
    """Model classes plus custom supervisor labels."""
    merged: dict[str, str] = {}
    for label in (*registered_har_action_labels(), *custom_har_action_labels(db)):
        key = label.casefold()
        if key not in merged:
            merged[key] = label
    return tuple(sorted(merged.values(), key=str.lower))


def register_custom_har_action_label(label: str, db: Session | None = None) -> str | None:
    """Persist a new action label outside the current model class list."""
    name = _normalize_label(label)
    if not name:
        return None
    if name.casefold() in {n.casefold() for n in registered_har_action_labels()}:
        return name
    existing = {_normalize_label(x).casefold(): _normalize_label(x) for x in _load_custom_labels_file()}
    key = name.casefold()
    if key not in existing:
        existing[key] = name
        _save_custom_labels_file(list(existing.values()))
    return existing[key]


def list_action_labels(db: Session) -> dict[str, list[str]]:
    model = list(registered_har_action_labels())
    model_keys = {n.casefold() for n in model}
    custom = [label for label in custom_har_action_labels(db) if label.casefold() not in model_keys]
    return {
        "model_labels": model,
        "custom_labels": custom,
        "all_labels": list(expandable_har_action_labels(db)),
    }

"""InHARD inventory — trainable clips with subject metadata for subject splits."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lib.constants import (
    ALL_META_ACTIONS,
    BLOCKED_ACTIONS,
    INHARD_DOWNLOAD_URL,
    TRAINABLE_ACTIONS,
)
from lib.paths import OUTPUTS_DIR, find_inhard_root, inhard_segmented_dir


@dataclass(frozen=True)
class ClipRecord:
    path: Path
    label: str
    session: str = ""
    subject: str = ""
    duration_sec: float = 0.0


@dataclass
class DataReport:
    ok: bool
    clips: list[ClipRecord] = field(default_factory=list)
    n_classes: int = 0
    trainable_on_disk: dict[str, int] = field(default_factory=dict)
    blocked_on_disk: dict[str, int] = field(default_factory=dict)
    missing_trainable: list[str] = field(default_factory=list)
    error: str | None = None


def load_inhard_csv(root: Path | None = None) -> pd.DataFrame:
    root = root or find_inhard_root()
    if root is None:
        return pd.DataFrame()
    csv_path = root / "Segmented" / "InHARD.csv"
    if not csv_path.is_file():
        csv_path = root / "Online" / "InHARD.csv"
    if not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_clip_metadata_map(root: Path | None = None) -> dict[str, dict[str, str]]:
    df = load_inhard_csv(root)
    if df.empty:
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        fname = str(row.get("File", row.get("file", ""))).strip()
        if not fname:
            continue
        stem = Path(fname).stem
        out[fname] = {
            "subject": str(row.get("Subject", "unknown")),
            "session": str(row.get("File", fname)).split("_")[0] if "File" in df.columns else "",
        }
        out[stem] = out[fname]
    return out


def _count_mp4_in_folder(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.mp4")))


def inventory_on_disk(seg_dir: Path) -> tuple[dict[str, int], dict[str, int], list[str]]:
    trainable: dict[str, int] = {}
    blocked: dict[str, int] = {}
    missing: list[str] = []
    blocked_set = {b.casefold() for b in BLOCKED_ACTIONS}

    for label in TRAINABLE_ACTIONS:
        n = _count_mp4_in_folder(seg_dir / label)
        if n:
            trainable[label] = n
        else:
            missing.append(label)

    for label in BLOCKED_ACTIONS:
        n = _count_mp4_in_folder(seg_dir / label)
        if n:
            blocked[label] = n

    return trainable, blocked, missing


def _labels_for_training(exclude_labels: tuple[str, ...]) -> list[str]:
    exclude = {x.casefold() for x in exclude_labels}
    return [label for label in ALL_META_ACTIONS if label.casefold() not in exclude]


def _enrich_clip(path: Path, label: str, meta_map: dict[str, dict[str, str]]) -> ClipRecord:
    info = meta_map.get(path.name) or meta_map.get(path.stem) or {}
    return ClipRecord(
        path=path,
        label=label,
        session=info.get("session", ""),
        subject=info.get("subject", "unknown"),
    )


def stratified_sample_clips(
    root: Path | None = None,
    *,
    clips_per_class: int,
    exclude_labels: tuple[str, ...] = BLOCKED_ACTIONS,
    seed: int = 42,
) -> list[ClipRecord]:
    seg = inhard_segmented_dir(root)
    if seg is None:
        return []
    rng = random.Random(seed)
    meta_map = load_clip_metadata_map(root)
    clips: list[ClipRecord] = []
    for label in _labels_for_training(exclude_labels):
        folder = seg / label
        if not folder.is_dir():
            continue
        mp4s = sorted(folder.glob("*.mp4"))
        if not mp4s:
            continue
        k = min(clips_per_class, len(mp4s))
        picked = rng.sample(mp4s, k) if len(mp4s) > k else mp4s
        for path in picked:
            clips.append(_enrich_clip(path, label, meta_map))
    return clips


def collect_trainable_clips(
    root: Path | None = None,
    *,
    exclude_labels: tuple[str, ...] = BLOCKED_ACTIONS,
    max_clips: int | None = None,
) -> list[ClipRecord]:
    seg = inhard_segmented_dir(root)
    if seg is None:
        return []
    exclude = {x.casefold() for x in exclude_labels}
    meta_map = load_clip_metadata_map(root)
    clips: list[ClipRecord] = []
    for label in TRAINABLE_ACTIONS:
        if label.casefold() in exclude:
            continue
        folder = seg / label
        if not folder.is_dir():
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            clips.append(_enrich_clip(mp4, label, meta_map))
            if max_clips and len(clips) >= max_clips:
                return clips
    return clips


def resolve_training_clips(
    root: Path | None = None,
    *,
    exclude_labels: tuple[str, ...] = BLOCKED_ACTIONS,
    clips_per_class: int | None = None,
    max_clips: int | None = None,
    seed: int = 42,
) -> list[ClipRecord]:
    if clips_per_class is not None and clips_per_class > 0:
        return stratified_sample_clips(
            root,
            clips_per_class=clips_per_class,
            exclude_labels=exclude_labels,
            seed=seed,
        )
    return collect_trainable_clips(root, exclude_labels=exclude_labels, max_clips=max_clips)


def analyze_training_clips(
    root: Path | None = None,
    *,
    exclude_labels: tuple[str, ...] = BLOCKED_ACTIONS,
    min_classes: int = 1,
    max_clips: int | None = None,
    clips_per_class: int | None = None,
    seed: int = 42,
) -> DataReport:
    root = root or find_inhard_root()
    seg = inhard_segmented_dir(root)
    if root is None or seg is None:
        return DataReport(
            ok=False,
            error="InHARD not found — extract to data_sample/InHARD-master/01-InHARD/",
        )

    trainable_on_disk, blocked_on_disk, missing = inventory_on_disk(seg)
    clips = resolve_training_clips(
        root,
        exclude_labels=exclude_labels,
        clips_per_class=clips_per_class,
        max_clips=max_clips,
        seed=seed,
    )
    labels = {c.label for c in clips}
    n_classes = len(labels)

    if not clips:
        err = (
            f"No trainable action clips found (blocked: {list(exclude_labels)}).\n"
            f"  Blocked on disk (ignored): {blocked_on_disk or '(none)'}\n"
            f"  Trainable on disk: {trainable_on_disk or '(none)'}\n"
            f"  Missing folders for {len(missing)}/{len(TRAINABLE_ACTIONS)} actions"
        )
        if missing:
            err += f", e.g. {missing[:5]}…"
        err += f"\n  → Download full InHARD: {INHARD_DOWNLOAD_URL}"
        return DataReport(
            ok=False,
            clips=[],
            trainable_on_disk=trainable_on_disk,
            blocked_on_disk=blocked_on_disk,
            missing_trainable=missing,
            error=err,
        )

    if n_classes < min_classes:
        return DataReport(
            ok=False,
            clips=clips,
            n_classes=n_classes,
            trainable_on_disk=trainable_on_disk,
            blocked_on_disk=blocked_on_disk,
            missing_trainable=missing,
            error=f"Need at least {min_classes} action classes, found {n_classes}",
        )

    return DataReport(
        ok=True,
        clips=clips,
        n_classes=n_classes,
        trainable_on_disk=trainable_on_disk,
        blocked_on_disk=blocked_on_disk,
        missing_trainable=missing,
    )


def label_counts(clips: list[ClipRecord]) -> pd.Series:
    if not clips:
        return pd.Series(dtype=int)
    return pd.Series([c.label for c in clips]).value_counts().sort_index()


def subject_counts(clips: list[ClipRecord]) -> pd.Series:
    if not clips:
        return pd.Series(dtype=int)
    return pd.Series([c.subject or "unknown" for c in clips]).value_counts()


def save_step01_summary(report: DataReport, root: Path | None = None) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": report.ok,
        "n_clips": len(report.clips),
        "n_classes": report.n_classes,
        "n_subjects": len({c.subject for c in report.clips}),
        "trainable_on_disk": report.trainable_on_disk,
        "blocked_on_disk": report.blocked_on_disk,
        "missing_trainable": report.missing_trainable,
        "blocked_actions": list(BLOCKED_ACTIONS),
        "trainable_actions": list(TRAINABLE_ACTIONS),
        "inhard_root": str(root or find_inhard_root() or ""),
        "error": report.error,
        "pipeline_version": "v2",
    }
    out = OUTPUTS_DIR / "pipeline_step01_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out

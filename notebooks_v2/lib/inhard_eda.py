"""InHARD exploratory analysis helpers (segmented CSV + disk inventory)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.constants import BLOCKED_ACTIONS, TRAINABLE_ACTIONS
from lib.inhard import inventory_on_disk, load_inhard_csv
from lib.paths import OUTPUTS_DIR, find_inhard_root, inhard_segmented_dir

INHARD_EDA_DIR = OUTPUTS_DIR / "inhard_eda"


def load_segmented_eda(root: Path | None = None) -> pd.DataFrame:
    """Segmented InHARD.csv enriched with trainable/blocked flags."""
    root = root or find_inhard_root()
    df = load_inhard_csv(root)
    if df.empty:
        return df
    out = df.copy()
    out["meta_action"] = out["Meta_action_label"]
    out["is_blocked"] = out["meta_action"].isin(BLOCKED_ACTIONS)
    out["is_trainable"] = out["meta_action"].isin(TRAINABLE_ACTIONS)
    out["session"] = out["File"].astype(str) if "File" in out.columns else ""
    if "Duration_sec" not in out.columns and {"Action_start_rgb_sec", "Action_end_rgb_sec"} <= set(out.columns):
        out["Duration_sec"] = out["Action_end_rgb_sec"] - out["Action_start_rgb_sec"]
    return out


def disk_vs_csv_summary(root: Path | None = None) -> dict:
    root = root or find_inhard_root()
    seg = inhard_segmented_dir(root)
    trainable, blocked, missing = inventory_on_disk(seg) if seg else ({}, {}, [])
    df = load_segmented_eda(root)
    csv_counts = df["meta_action"].value_counts().to_dict() if len(df) else {}
    all_disk = {**blocked, **trainable}
    return {
        "inhard_root": str(root or ""),
        "csv_rows": len(df),
        "disk_mp4_total": sum(all_disk.values()),
        "trainable_clips_disk": sum(trainable.values()),
        "blocked_clips_disk": sum(blocked.values()),
        "trainable_classes": len(trainable),
        "missing_trainable_folders": missing,
        "per_class_csv": csv_counts,
        "per_class_disk": all_disk,
        "csv_matches_disk": sum(csv_counts.values()) == sum(all_disk.values()),
    }


def save_eda_summary(df: pd.DataFrame, root: Path | None = None) -> Path:
    root = root or find_inhard_root()
    summary = disk_vs_csv_summary(root)
    if len(df):
        train = df[df["is_trainable"]]
        summary["duration_sec"] = {
            "all_mean": round(float(df["Duration_sec"].mean()), 3),
            "all_median": round(float(df["Duration_sec"].median()), 3),
            "trainable_mean": round(float(train["Duration_sec"].mean()), 3),
            "trainable_median": round(float(train["Duration_sec"].median()), 3),
        }
        summary["n_sessions"] = int(df["session"].nunique())
        summary["n_subjects"] = int(df["Subject"].nunique()) if "Subject" in df.columns else None
        summary["n_operations"] = int(df["Operation"].nunique()) if "Operation" in df.columns else None
    out = INHARD_EDA_DIR / "inhard_eda_summary.json"
    INHARD_EDA_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Legacy path for pipeline notebooks
    legacy = OUTPUTS_DIR / "inhard_eda_summary.json"
    legacy.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out

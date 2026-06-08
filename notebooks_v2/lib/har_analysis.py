"""Training + live-session analysis for HAR model comprehension."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from lib.constants import VJEPA_MODEL_ID
from lib.har_model import load_checkpoint
from lib.paths import CHECKPOINTS_DIR, OUTPUTS_DIR, SESSIONS_DIR
from lib.session_log import list_sessions, model_tag_from_checkpoint

ANALYSIS_DIR = OUTPUTS_DIR / "har_analysis"

REPORT_METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings_npz(npz_path: Path | None = None) -> dict[str, Any]:
    npz_path = npz_path or (OUTPUTS_DIR / "embeddings.npz")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing embeddings: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    meta_csv = OUTPUTS_DIR / "embedding_meta.csv"
    meta_df = pd.read_csv(meta_csv) if meta_csv.is_file() else pd.DataFrame()
    return {
        "path": npz_path,
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "class_names": list(data["class_names"]),
        "meta_df": meta_df,
    }


def eval_checkpoint_on_embeddings(
    checkpoint: Path,
    *,
    npz_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    split: str = "subject",
) -> dict[str, Any]:
    """Precision / recall / F1 with subject-held-out or random split."""
    from lib.har_train import _split_indices

    bundle = load_embeddings_npz(npz_path)
    X, y = bundle["X"], bundle["y"]
    class_names = bundle["class_names"]
    subjects = None
    data = np.load(bundle["path"], allow_pickle=True)
    if "subjects" in data:
        subjects = data["subjects"].astype(str)
    model, info = load_checkpoint(checkpoint)

    if split == "full":
        train_idx = np.arange(len(y))
        test_idx = np.arange(len(y))
    else:
        train_idx, test_idx = _split_indices(
            y,
            subjects,
            test_size=test_size,
            seed=seed,
            split_mode=split if split in ("subject", "random") else "random",
        )

    X_test = X[test_idx]
    y_test = y[test_idx]

    device = info["device"]
    with torch.inference_mode():
        logits = model(torch.from_numpy(X_test).float().to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    y_pred = probs.argmax(axis=1)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    return {
        "checkpoint": str(checkpoint),
        "model_tag": model_tag_from_checkpoint(checkpoint),
        "class_names": class_names,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split": split,
        "y_test": y_test,
        "y_pred": y_pred,
        "probs": probs,
        "report": report,
        "confusion_matrix": cm,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "info": info,
        "embedding_meta": bundle["meta_df"],
    }


def load_sessions_index() -> pd.DataFrame:
    index_path = SESSIONS_DIR / "index.csv"
    if not index_path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(index_path)
    for col in ("n_events", "n_confirmed", "n_persons"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def load_session_events(session_dir: Path | str) -> pd.DataFrame:
    session_dir = Path(session_dir)
    if not session_dir.is_absolute():
        session_dir = SESSIONS_DIR / session_dir
    events_path = session_dir / "events.jsonl"
    if not events_path.is_file():
        return pd.DataFrame()
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_all_events(*, model_tag: str | None = None, source: str | None = None) -> pd.DataFrame:
    index = load_sessions_index()
    if index.empty:
        return pd.DataFrame()
    if model_tag:
        index = index[index["model_tag"] == model_tag]
    if source:
        index = index[index["source"] == source]
    frames: list[pd.DataFrame] = []
    for _, row in index.iterrows():
        sd = SESSIONS_DIR / str(row["session_dir"])
        ev = load_session_events(sd)
        if ev.empty:
            continue
        ev["session_dir"] = str(row["session_dir"])
        ev["index_date"] = row.get("date")
        ev["index_source"] = row.get("source")
        frames.append(ev)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["infer_ms"] = pd.to_numeric(out.get("infer_ms"), errors="coerce")
    return out


def _save_fig(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_class_balance(meta_df: pd.DataFrame, class_names: list[str], out_dir: Path) -> str:
    if meta_df.empty or "label" not in meta_df.columns:
        counts = pd.Series({c: 0 for c in class_names})
    else:
        counts = meta_df["label"].value_counts().reindex(class_names, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#4CAF50")
    ax.set_title("Training sample — clips per class")
    ax.set_xlabel("clips")
    return _save_fig(out_dir / "01_train_class_balance.png")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path, *, normalize: bool = True) -> str:
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    short = [c[:18] + "…" if len(c) > 19 else c for c in class_names]
    sns.heatmap(mat, annot=False, cmap="Blues", xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (holdout)")
    ax.set_title("Confusion matrix (row-normalized)" if normalize else "Confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return _save_fig(out_dir / "02_confusion_matrix.png")


def plot_per_class_metrics(report: dict, class_names: list[str], out_dir: Path) -> str:
    rows = []
    for name in class_names:
        if name in report:
            rows.append({"class": name, **report[name]})
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["precision"], width=w, label="Precision", color="#2196F3")
    ax.bar(x, df["recall"], width=w, label="Recall", color="#FF9800")
    ax.bar(x + w, df["f1-score"], width=w, label="F1", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:16] for c in df["class"]], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class metrics (holdout set)")
    return _save_fig(out_dir / "03_per_class_prf1.png")


def plot_training_history(pipeline_summary_path: Path | None, out_dir: Path) -> str | None:
    path = pipeline_summary_path or (OUTPUTS_DIR / "pipeline_run_summary.json")
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("steps", {}).get("03_train", {}).get("history")
    if not history:
        return None
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", marker="o", ms=3)
    ax.plot(df["epoch"], df["val_loss"], label="Val loss", marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    ax.set_title("Training curve (last pipeline run)")
    return _save_fig(out_dir / "04_training_loss.png")


def plot_session_confidence(events: pd.DataFrame, out_dir: Path) -> str | None:
    if events.empty or "confidence" not in events.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    events["confidence"].hist(bins=20, ax=axes[0], color="#673AB7", edgecolor="white")
    axes[0].set_title("Confidence distribution (all inferences)")
    axes[0].set_xlabel("confidence")
    if "confirmed_detection" in events.columns:
        confirmed = events[events["confirmed_detection"] == True]  # noqa: E712
        if not confirmed.empty:
            confirmed["confidence"].hist(bins=15, ax=axes[1], color="#009688", edgecolor="white")
    axes[1].set_title("Confidence — confirmed detections only")
    axes[1].set_xlabel("confidence")
    return _save_fig(out_dir / "05_session_confidence.png")


def plot_session_label_distribution(events: pd.DataFrame, out_dir: Path) -> str | None:
    if events.empty or "action_label" not in events.columns:
        return None
    counts = events["action_label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#3F51B5")
    ax.set_title("Live/eval — predicted action frequency")
    ax.set_xlabel("inference count")
    return _save_fig(out_dir / "06_session_label_distribution.png")


def plot_model_comparison(index_df: pd.DataFrame, out_dir: Path) -> str | None:
    if index_df.empty or index_df["model_tag"].nunique() < 1:
        return None
    agg = index_df.groupby("model_tag", as_index=False).agg(
        sessions=("run_id", "count"),
        total_events=("n_events", "sum"),
        total_confirmed=("n_confirmed", "sum"),
        avg_persons=("n_persons", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    agg.plot(x="model_tag", y="total_events", kind="bar", ax=axes[0], legend=False, color="#607D8B")
    axes[0].set_title("Total inferences by model")
    agg.plot(x="model_tag", y="total_confirmed", kind="bar", ax=axes[1], legend=False, color="#795548")
    axes[1].set_title("Confirmed detections by model")
    if len(agg) > 1:
        agg.plot(x="model_tag", y="sessions", kind="bar", ax=axes[2], legend=False, color="#E91E63")
        axes[2].set_title("Sessions logged")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    return _save_fig(out_dir / "07_model_comparison.png")


def plot_inference_latency(events: pd.DataFrame, out_dir: Path) -> str | None:
    if events.empty or events["infer_ms"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [events["infer_ms"].dropna()]
    if "index_source" in events.columns and events["index_source"].nunique() > 1:
        data = [g["infer_ms"].dropna() for _, g in events.groupby("index_source")]
        ax.boxplot(data, labels=sorted(events["index_source"].unique()))
    else:
        ax.boxplot(data, labels=["all"])
    ax.set_ylabel("ms")
    ax.set_title("V-JEPA2 + MLP inference latency")
    return _save_fig(out_dir / "08_inference_latency.png")


def metrics_table(report: dict, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in class_names:
        if name in report:
            rows.append(
                {
                    "class": name,
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1": report[name]["f1-score"],
                    "support": report[name]["support"],
                }
            )
    return pd.DataFrame(rows)


def build_report_markdown(
    *,
    eval_result: dict[str, Any],
    index_df: pd.DataFrame,
    events: pd.DataFrame,
    chart_paths: dict[str, str | None],
    out_dir: Path,
) -> str:
    tag = eval_result["model_tag"]
    lines = [
        f"# HAR Model Analysis — `{tag}`",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Model identity",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Checkpoint | `{Path(eval_result['checkpoint']).name}` |",
        f"| Embedding | `{VJEPA_MODEL_ID}` |",
        f"| Classes | {len(eval_result['class_names'])} |",
        f"| Train clips | {eval_result['n_train']} |",
        f"| Holdout clips | {eval_result['n_test']} |",
        "",
        "## Holdout metrics (training set)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | **{eval_result['accuracy']:.1%}** |",
        f"| Macro F1 | **{eval_result['macro_f1']:.3f}** |",
        f"| Weighted F1 | **{eval_result['weighted_f1']:.3f}** |",
        f"| Macro precision | {eval_result['macro_precision']:.3f} |",
        f"| Macro recall | {eval_result['macro_recall']:.3f} |",
        "",
        "## Live / eval sessions",
        "",
    ]
    if not index_df.empty:
        sub = index_df[index_df["model_tag"] == tag] if tag else index_df
        lines.append(f"- Sessions logged: **{len(sub)}**")
        lines.append(f"- Total inferences: **{int(sub['n_events'].sum())}**")
        lines.append(f"- Confirmed detections: **{int(sub['n_confirmed'].sum())}**")
    if not events.empty:
        lines.append(f"- Mean confidence: **{events['confidence'].mean():.1%}**")
        if not events["infer_ms"].isna().all():
            lines.append(f"- Mean infer latency: **{events['infer_ms'].mean():.0f} ms**")
    lines.extend(["", "## Charts", ""])
    for name, path in chart_paths.items():
        if path:
            rel = Path(path).relative_to(out_dir)
            lines.append(f"- `{name}`: {rel}")
    lines.append("")
    return "\n".join(lines)


def run_full_analysis(
    checkpoint: Path | None = None,
    *,
    out_dir: Path | None = None,
    model_tag: str | None = None,
    split: str = "subject",
) -> dict[str, Any]:
    """Run training + session analysis; save charts and REPORT.md."""
    if checkpoint is None:
        candidates = sorted(CHECKPOINTS_DIR.glob("har_vjepa_*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint in checkpoints/")
        checkpoint = candidates[-1]
    checkpoint = Path(checkpoint)
    tag = model_tag or model_tag_from_checkpoint(checkpoint)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = _ensure_dir(out_dir or (ANALYSIS_DIR / f"{stamp}_{tag}"))

    eval_result = eval_checkpoint_on_embeddings(checkpoint, split=split)
    index_df = load_sessions_index()
    events = load_all_events(model_tag=tag)
    meta_df = eval_result.get("embedding_meta", pd.DataFrame())

    charts: dict[str, str | None] = {}
    charts["class_balance"] = plot_class_balance(meta_df, eval_result["class_names"], out_dir)
    charts["confusion_matrix"] = plot_confusion_matrix(
        eval_result["confusion_matrix"], eval_result["class_names"], out_dir
    )
    charts["per_class_metrics"] = plot_per_class_metrics(eval_result["report"], eval_result["class_names"], out_dir)
    charts["training_loss"] = plot_training_history(None, out_dir)
    charts["session_confidence"] = plot_session_confidence(events, out_dir)
    charts["session_labels"] = plot_session_label_distribution(events, out_dir)
    charts["model_comparison"] = plot_model_comparison(index_df, out_dir)
    charts["inference_latency"] = plot_inference_latency(events, out_dir)

    metrics_df = metrics_table(eval_result["report"], eval_result["class_names"])
    metrics_df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    summary = {
        "model_tag": tag,
        "checkpoint": str(checkpoint),
        "embedding_model": VJEPA_MODEL_ID,
        "n_classes": len(eval_result["class_names"]),
        "n_train": eval_result["n_train"],
        "n_test": eval_result["n_test"],
        **{k: eval_result[k] for k in REPORT_METRICS},
        "n_sessions": int(len(index_df[index_df["model_tag"] == tag])) if not index_df.empty else 0,
        "n_live_events": int(len(events)),
        "charts": charts,
        "out_dir": str(out_dir),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = build_report_markdown(
        eval_result=eval_result,
        index_df=index_df,
        events=events,
        chart_paths=charts,
        out_dir=out_dir,
    )
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    return {
        "eval": eval_result,
        "metrics_df": metrics_df,
        "index_df": index_df,
        "events": events,
        "charts": charts,
        "summary": summary,
        "out_dir": out_dir,
        "report_path": out_dir / "REPORT.md",
    }

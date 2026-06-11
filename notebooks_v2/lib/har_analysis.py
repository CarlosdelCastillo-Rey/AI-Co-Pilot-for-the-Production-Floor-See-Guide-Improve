"""v2 HAR analysis — UMAP/t-SNE clusters, frame strips, YOLO detection viz, per-person heatmaps."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.gridspec as gridspec
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
)
from sklearn.model_selection import train_test_split

from lib.constants import VJEPA_MODEL_ID
from lib.har_model import load_checkpoint
from lib.paths import ARCHIVE_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, SESSIONS_DIR
from lib.session_log import list_sessions, model_tag_from_checkpoint

ANALYSIS_DIR = OUTPUTS_DIR / "har_analysis_v2"
REPORT_METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")

TAB20 = plt.cm.get_cmap("tab20")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(path: Path, dpi: int = 150) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return str(path)


# ── Embedding loaders ─────────────────────────────────────────────────────────

def load_embeddings_npz(npz_path: Path | None = None) -> dict[str, Any]:
    npz_path = npz_path or (OUTPUTS_DIR / "embeddings.npz")
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    meta_csv = npz_path.parent / (
        "embedding_meta_dinov2.csv" if "dinov2" in npz_path.name else "embedding_meta.csv"
    )
    meta_df = pd.read_csv(meta_csv) if meta_csv.is_file() else pd.DataFrame()
    return {
        "path":        npz_path,
        "X":           data["X"].astype(np.float32),
        "y":           data["y"].astype(np.int64),
        "class_names": list(data["class_names"]),
        "subjects":    data["subjects"].astype(str) if "subjects" in data else None,
        "meta_df":     meta_df,
    }


def eval_checkpoint(
    checkpoint: Path,
    *,
    npz_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    split: str = "subject",
) -> dict[str, Any]:
    from lib.har_train import _split_indices

    bundle = load_embeddings_npz(npz_path)
    X, y   = bundle["X"], bundle["y"]
    class_names = bundle["class_names"]
    subjects    = bundle["subjects"]
    model, info = load_checkpoint(checkpoint)

    if split == "full":
        train_idx = test_idx = np.arange(len(y))
    else:
        train_idx, test_idx = _split_indices(
            y, subjects,
            test_size=test_size, seed=seed,
            split_mode=split if split in ("subject", "random") else "random",
        )

    X_test, y_test = X[test_idx], y[test_idx]
    device = info["device"]
    with torch.inference_mode():
        logits = model(torch.from_numpy(X_test).float().to(device))
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
    y_pred = probs.argmax(axis=1)

    report = classification_report(
        y_test, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    return {
        "checkpoint":       str(checkpoint),
        "model_tag":        model_tag_from_checkpoint(checkpoint),
        "class_names":      class_names,
        "n_train":          int(len(train_idx)),
        "n_test":           int(len(test_idx)),
        "split":            split,
        "train_idx":        train_idx,
        "test_idx":         test_idx,
        "y_test":           y_test,
        "y_pred":           y_pred,
        "probs":            probs,
        "report":           report,
        "confusion_matrix": cm,
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "macro_f1":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1":      float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_precision":  float(report["macro avg"]["precision"]),
        "macro_recall":     float(report["macro avg"]["recall"]),
        "info":             info,
        "embedding_meta":   bundle["meta_df"],
        "npz_path":         str(bundle["path"]),
        "X_all":            X,
        "y_all":            y,
        "subjects":         subjects,
    }


# ── 1. Class balance ──────────────────────────────────────────────────────────

def plot_class_balance(meta_df: pd.DataFrame, class_names: list[str], out_dir: Path) -> str:
    if meta_df.empty or "label" not in meta_df.columns:
        counts = pd.Series({c: 0 for c in class_names})
    else:
        counts = meta_df["label"].value_counts().reindex(class_names, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = counts.sort_values().plot(kind="barh", ax=ax, color=[TAB20(i / max(len(class_names)-1, 1)) for i in range(len(class_names))])
    ax.axvline(counts.mean(), color="red", lw=1.5, ls="--", label=f"mean={counts.mean():.0f}")
    ax.set_title("Training clips per class (full dataset usage)", fontsize=13)
    ax.set_xlabel("clips")
    ax.legend()
    for bar, val in zip(ax.patches, counts.sort_values().values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(int(val)), va="center", fontsize=8)
    return _save_fig(out_dir / "01_class_balance.png")


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path, *, normalize: bool = True) -> str:
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    short = [c[:18] + "…" if len(c) > 19 else c for c in class_names]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(mat, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=short, yticklabels=short, ax=ax,
                linewidths=0.3, linecolor="lightgray", cbar_kws={"label": "recall" if normalize else "count"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True (holdout)", fontsize=12)
    ax.set_title("Confusion matrix — row-normalized" if normalize else "Confusion matrix", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    return _save_fig(out_dir / "02_confusion_matrix.png")


# ── 3. Per-class F1 / P / R bars ─────────────────────────────────────────────

def plot_per_class_metrics(report: dict, class_names: list[str], out_dir: Path) -> str:
    rows = [{"class": n, **report[n]} for n in class_names if n in report]
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w,   df["precision"], w, label="Precision", color="#2196F3")
    ax.bar(x,       df["recall"],    w, label="Recall",    color="#FF9800")
    ax.bar(x + w,   df["f1-score"],  w, label="F1",        color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:16] for c in df["class"]], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, color="green", ls="--", lw=1, label="target 90%")
    ax.legend()
    ax.set_title("Per-class Precision / Recall / F1 (holdout)", fontsize=13)
    for bar in ax.patches:
        if bar.get_height() > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=6, rotation=90)
    return _save_fig(out_dir / "03_per_class_prf1.png")


# ── 4. Training history ───────────────────────────────────────────────────────

def plot_training_history(history: list[dict], out_dir: Path, *, tag: str = "") -> str:
    if not history:
        return ""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", marker="o", ms=2)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   marker="o", ms=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title(f"Training curve {tag}")
    if "lr" in df.columns:
        axes[1].plot(df["epoch"], df["lr"], color="purple")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("CosineAnnealingLR schedule")
    else:
        axes[1].axis("off")
    return _save_fig(out_dir / "04_training_history.png")


# ── 5. UMAP embedding clusters ────────────────────────────────────────────────

def plot_umap_clusters(eval_result: dict, out_dir: Path, *, seed: int = 42, n_neighbors: int = 15) -> str:
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors)
        method  = "UMAP"
    except ImportError:
        warnings.warn("umap-learn not installed; falling back to t-SNE")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
        method  = "t-SNE"

    X   = eval_result["X_all"]
    y   = eval_result["y_all"]
    y_p = np.full(len(y), -1, dtype=int)
    y_p[eval_result["test_idx"]] = eval_result["y_pred"]
    class_names = eval_result["class_names"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X2 = reducer.fit_transform(X)

    n_cls = len(class_names)
    colors = [TAB20(i / max(n_cls - 1, 1)) for i in range(n_cls)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: true labels
    for i, name in enumerate(class_names):
        mask = y == i
        axes[0].scatter(X2[mask, 0], X2[mask, 1], c=[colors[i]], label=name[:16],
                        s=15, alpha=0.55, edgecolors="none")
    axes[0].set_title(f"{method} — True labels (all clips)", fontsize=12)
    axes[0].legend(loc="best", fontsize=7, markerscale=1.5, ncol=2)
    axes[0].set_xlabel(f"{method}1"); axes[0].set_ylabel(f"{method}2")

    # Right: correct vs wrong on holdout set
    test_mask  = np.zeros(len(y), dtype=bool)
    test_mask[eval_result["test_idx"]] = True
    correct    = y_p == y
    wrong_mask = test_mask & ~correct

    axes[1].scatter(X2[~test_mask, 0], X2[~test_mask, 1], c="lightgray", s=10, alpha=0.3, label="train")
    for i, name in enumerate(class_names):
        m = test_mask & (y == i) & correct
        axes[1].scatter(X2[m, 0], X2[m, 1], c=[colors[i]], s=20, alpha=0.7, edgecolors="none")
    axes[1].scatter(X2[wrong_mask, 0], X2[wrong_mask, 1],
                    facecolors="none", edgecolors="red", s=55, lw=1.2, label="misclassified")
    axes[1].set_title(f"{method} — Holdout: correct (color) vs misclassified (red ring)", fontsize=12)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xlabel(f"{method}1"); axes[1].set_ylabel(f"{method}2")

    fig.suptitle(f"Embedding Cluster Analysis — {eval_result['model_tag']}", fontsize=14, y=1.01)
    return _save_fig(out_dir / "05_umap_clusters.png", dpi=130)


# ── 6. Frame strip comparison ─────────────────────────────────────────────────

def plot_frame_strip_comparison(
    clip_paths_per_class: dict[str, list[Path]],
    out_dir: Path,
    *,
    n_clips: int = 3,
    n_frames_per_clip: int = 5,
    view: str = "topdown",
) -> str:
    """Visual grid: each row = one action class, columns = sampled frames."""
    from lib.crop_extract import extract_inhard_view
    from lib.embeddings import extract_clip_from_file

    class_names = list(clip_paths_per_class.keys())
    n_rows = len(class_names)
    n_cols = n_clips * n_frames_per_clip
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, cls in enumerate(class_names):
        paths = clip_paths_per_class[cls][:n_clips]
        col = 0
        for clip_path in paths:
            frames = extract_clip_from_file(clip_path, max_frames=32)
            if not frames:
                col += n_frames_per_clip
                continue
            indices = np.linspace(0, len(frames) - 1, n_frames_per_clip, dtype=int)
            for idx in indices:
                frame = extract_inhard_view(frames[idx], view=view)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (112, 112))
                if col < n_cols:
                    axes[r, col].imshow(frame_rgb)
                    axes[r, col].axis("off")
                col += 1
        # fill remaining cols
        while col < n_cols:
            axes[r, col].axis("off")
            col += 1
        axes[r, 0].set_ylabel(cls[:18], fontsize=8, rotation=0, ha="right", va="center")

    fig.suptitle(f"Frame strip comparison — view: {view}", fontsize=12)
    plt.subplots_adjust(wspace=0.02, hspace=0.1, left=0.15)
    return _save_fig(out_dir / "06_frame_strip_comparison.png", dpi=100)


# ── 7. YOLO person detection visualization ───────────────────────────────────

def plot_yolo_detection_grid(
    clip_paths: list[Path],
    labels: list[str],
    out_dir: Path,
    *,
    n_clips: int = 12,
    view: str = "topdown",
) -> str:
    """Show per-clip person detection bounding boxes over extracted sub-view."""
    from lib.crop_extract import extract_inhard_view, frame_to_person_crop
    from lib.embeddings import extract_clip_from_file

    clips = list(zip(clip_paths, labels))[:n_clips]
    n_cols = 4
    n_rows = int(np.ceil(len(clips) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (clip_path, label) in enumerate(clips):
        frames = extract_clip_from_file(clip_path, max_frames=32)
        if not frames:
            continue
        mid   = frames[len(frames) // 2]
        sub   = extract_inhard_view(mid, view=view)
        crop, bbox = frame_to_person_crop(mid, view=view)
        display = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB).copy()
        if bbox is not None:
            x1, y1, x2, y2, conf = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(display, f"{conf:.2f}", (x1, max(y1-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        else:
            cv2.putText(display, "no det", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
        axes[i].imshow(display)
        axes[i].set_title(f"{label[:20]}\n{clip_path.name[:22]}", fontsize=7)
        axes[i].axis("off")

    fig.suptitle(f"YOLO person detection — view: {view}", fontsize=12)
    return _save_fig(out_dir / "07_yolo_detection_grid.png", dpi=120)


# ── 8. Per-person accuracy heatmap ───────────────────────────────────────────

def plot_per_person_heatmap(eval_result: dict, out_dir: Path) -> str:
    subjects = eval_result.get("subjects")
    if subjects is None:
        return ""
    test_subj = subjects[eval_result["test_idx"]]
    y_test    = eval_result["y_test"]
    y_pred    = eval_result["y_pred"]
    class_names = eval_result["class_names"]

    uniq_subj = sorted(set(test_subj))
    rows = []
    for s in uniq_subj:
        mask = test_subj == s
        acc_per_class = []
        for c in range(len(class_names)):
            cls_mask = mask & (y_test == c)
            if cls_mask.sum() == 0:
                acc_per_class.append(np.nan)
            else:
                acc_per_class.append((y_pred[cls_mask] == y_test[cls_mask]).mean())
        rows.append(acc_per_class)

    df = pd.DataFrame(rows, index=uniq_subj, columns=[c[:14] for c in class_names])
    fig, ax = plt.subplots(figsize=(14, max(4, 0.5 * len(uniq_subj))))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax, linewidths=0.3, cbar_kws={"label": "accuracy"}, mask=df.isna())
    ax.set_title("Per-person × per-class accuracy (holdout subjects)", fontsize=13)
    ax.set_xlabel("Action class"); ax.set_ylabel("Subject")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    return _save_fig(out_dir / "08_per_person_heatmap.png")


# ── 9. Confidence calibration ─────────────────────────────────────────────────

def plot_confidence_calibration(eval_result: dict, out_dir: Path, *, n_bins: int = 10) -> str:
    probs   = eval_result["probs"]
    y_test  = eval_result["y_test"]
    y_pred  = eval_result["y_pred"]
    conf    = probs.max(axis=1)
    correct = (y_pred == y_test).astype(float)

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_mid = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            bin_acc.append(correct[mask].mean())
            bin_mid.append((lo + hi) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Calibration curve
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    axes[0].plot(bin_mid, bin_acc, "o-", color="#E91E63", label="model")
    axes[0].fill_between(bin_mid, bin_acc, bin_mid, alpha=0.15, color="red", label="gap")
    axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Confidence calibration curve")
    axes[0].legend(); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    # Confidence distribution: correct vs wrong
    axes[1].hist(conf[correct == 1], bins=20, alpha=0.6, color="#4CAF50", label=f"correct  (n={int(correct.sum())})")
    axes[1].hist(conf[correct == 0], bins=20, alpha=0.7, color="#F44336", label=f"wrong    (n={int((1-correct).sum())})")
    axes[1].axvline(0.25, color="orange", ls="--", lw=1.5, label="min_conf threshold")
    axes[1].set_xlabel("Max softmax confidence"); axes[1].set_ylabel("count")
    axes[1].set_title("Confidence distribution — correct vs wrong")
    axes[1].legend()

    return _save_fig(out_dir / "09_confidence_calibration.png")


# ── 10. Confusion pair deep-dive ──────────────────────────────────────────────

def plot_top_confusions(eval_result: dict, out_dir: Path, *, top_k: int = 6) -> str:
    cm          = eval_result["confusion_matrix"].copy()
    class_names = eval_result["class_names"]
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    top = pairs[:top_k]

    if not top:
        return ""

    labels  = [f"{t}→{p}" for _, t, p in top]
    counts  = [c for c, _, _ in top]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(labels[::-1], counts[::-1], color="#FF5722")
    ax.set_xlabel("# misclassifications")
    ax.set_title(f"Top-{top_k} confusion pairs (true → predicted)", fontsize=12)
    for bar, val in zip(ax.patches, counts[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(int(val)), va="center")
    return _save_fig(out_dir / "10_top_confusions.png")


# ── 11. View comparison strip ─────────────────────────────────────────────────

def plot_3view_comparison(clip_path: Path, out_dir: Path, *, label: str = "") -> str:
    """Show all 3 InHARD views side-by-side for a single clip."""
    from lib.crop_extract import extract_all_views, frame_to_person_crop
    from lib.embeddings import extract_clip_from_file
    from lib.constants import INHARD_VIEW_TOPDOWN, INHARD_VIEW_SIDE, INHARD_VIEW_FRONT

    frames = extract_clip_from_file(clip_path, max_frames=32)
    if not frames:
        return ""
    views_order = [INHARD_VIEW_TOPDOWN, INHARD_VIEW_SIDE, INHARD_VIEW_FRONT]
    view_labels = ["Overhead (top-down)", "Side (eye-level)", "Front (low-angle)"]
    indices = np.linspace(0, len(frames)-1, 5, dtype=int)

    fig, axes = plt.subplots(3, 5, figsize=(14, 8))
    for r, (view, vlabel) in enumerate(zip(views_order, view_labels)):
        for c, idx in enumerate(indices):
            sub = extract_all_views(frames[idx])[view]
            crop, bbox = frame_to_person_crop(frames[idx], view=view)
            disp = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB).copy()
            if bbox is not None:
                x1, y1, x2, y2, conf = bbox
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 2)
            axes[r, c].imshow(disp)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(vlabel, fontsize=9, rotation=0, ha="right", va="center")
            if r == 0:
                axes[r, c].set_title(f"t={idx}", fontsize=8)
    fig.suptitle(f"3-view comparison — {label or clip_path.name}", fontsize=12)
    plt.subplots_adjust(wspace=0.02, hspace=0.05, left=0.2)
    fname = "11_3view_comparison.png"
    return _save_fig(out_dir / fname)


# ── 12. Training loss comparison backbone ─────────────────────────────────────

def plot_backbone_comparison(bench: pd.DataFrame, out_dir: Path) -> str:
    ok = bench.dropna(subset=["macro_f1"])
    if ok.empty:
        return ""
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.4*len(ok))))
    metrics = [("macro_f1", "Macro F1"), ("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1")]
    for ax, (col, title) in zip(axes, metrics):
        colors = ["#E91E63" if "vjepa" in str(r.get("checkpoint","")) else "#2196F3"
                  for _, r in ok.iterrows()]
        ok.plot(x="model_tag", y=col, kind="barh", ax=ax, legend=False, color=colors)
        ax.set_xlim(0, 1)
        ax.axvline(0.9, color="green", ls="--", lw=1)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Checkpoint leaderboard (red=VJEPA, blue=DINOv2)", fontsize=12)
    return _save_fig(out_dir / "12_backbone_leaderboard.png")


# ── 13. Embedding PCA outliers ───────────────────────────────────────────────

def plot_pca_outliers(eval_result: dict, out_dir: Path, *, seed: int = 42) -> str:
    from sklearn.decomposition import PCA
    X       = eval_result["X_all"][eval_result["test_idx"]]
    y       = eval_result["y_test"]
    y_pred  = eval_result["y_pred"]
    class_names = eval_result["class_names"]
    if len(X) < 10:
        return ""
    X2    = PCA(n_components=2, random_state=seed).fit_transform(X)
    wrong = y != y_pred
    n_cls = len(class_names)
    colors = [TAB20(i / max(n_cls-1, 1)) for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        m = (y == i) & ~wrong
        ax.scatter(X2[m, 0], X2[m, 1], c=[colors[i]], s=20, alpha=0.5, label=name[:14])
    ax.scatter(X2[wrong, 0], X2[wrong, 1], facecolors="none",
               edgecolors="red", s=70, lw=1.5, label="misclassified", zorder=5)
    ax.set_title("PCA of holdout embeddings — correct (color) vs misclassified (red ring)", fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    return _save_fig(out_dir / "13_pca_outliers.png")


# ── Full analysis orchestrator ────────────────────────────────────────────────

def run_v2_analysis(
    checkpoint: Path | None = None,
    *,
    npz_path: Path | None = None,
    out_dir: Path | None = None,
    split: str = "subject",
    seed: int = 42,
    sample_clips_for_strip: dict[str, list[Path]] | None = None,
    yolo_sample_clips: list[tuple[Path, str]] | None = None,
    training_history: list[dict] | None = None,
    example_clip_for_3view: Path | None = None,
) -> dict[str, Any]:
    """Full v2 analysis run — all charts, UMAP, frame strips, YOLO grid."""
    if checkpoint is None:
        candidates = sorted(CHECKPOINTS_DIR.glob("har_*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint in checkpoints/")
        checkpoint = candidates[-1]
    checkpoint = Path(checkpoint)
    tag   = model_tag_from_checkpoint(checkpoint)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = _ensure_dir(out_dir or (ANALYSIS_DIR / f"{stamp}_{tag}"))

    # Eval
    result = eval_checkpoint(checkpoint, npz_path=npz_path, split=split, seed=seed)
    meta_df = result["embedding_meta"]

    charts: dict[str, str] = {}
    charts["class_balance"]     = plot_class_balance(meta_df, result["class_names"], out_dir)
    charts["confusion_matrix"]  = plot_confusion_matrix(result["confusion_matrix"], result["class_names"], out_dir)
    charts["per_class_metrics"] = plot_per_class_metrics(result["report"], result["class_names"], out_dir)
    if training_history:
        charts["training_history"] = plot_training_history(training_history, out_dir, tag=tag)
    charts["umap_clusters"]        = plot_umap_clusters(result, out_dir, seed=seed)
    charts["pca_outliers"]         = plot_pca_outliers(result, out_dir, seed=seed)
    charts["confidence_calib"]     = plot_confidence_calibration(result, out_dir)
    charts["top_confusions"]       = plot_top_confusions(result, out_dir)
    charts["per_person_heatmap"]   = plot_per_person_heatmap(result, out_dir)

    if sample_clips_for_strip:
        charts["frame_strip"] = plot_frame_strip_comparison(sample_clips_for_strip, out_dir)
    if yolo_sample_clips:
        paths  = [p for p, _ in yolo_sample_clips]
        labels = [l for _, l in yolo_sample_clips]
        charts["yolo_detection"] = plot_yolo_detection_grid(paths, labels, out_dir)
    if example_clip_for_3view:
        charts["3view_comparison"] = plot_3view_comparison(
            example_clip_for_3view, out_dir,
            label=example_clip_for_3view.parent.name,
        )

    # Metrics table
    rows = [{"class": n, **result["report"][n]}
            for n in result["class_names"] if n in result["report"]]
    pd.DataFrame(rows).to_csv(out_dir / "per_class_metrics.csv", index=False)

    summary = {
        "model_tag":    tag,
        "checkpoint":   str(checkpoint),
        "split":        split,
        "npz_path":     result["npz_path"],
        "n_classes":    len(result["class_names"]),
        "n_train":      result["n_train"],
        "n_test":       result["n_test"],
        "accuracy":     result["accuracy"],
        "macro_f1":     result["macro_f1"],
        "weighted_f1":  result["weighted_f1"],
        "macro_precision": result["macro_precision"],
        "macro_recall":    result["macro_recall"],
        "charts":       {k: str(v) for k, v in charts.items()},
        "out_dir":      str(out_dir),
        "generated_at": stamp,
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown report
    lines = [
        f"# v2 HAR Analysis — `{tag}`",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Accuracy** | **{result['accuracy']:.1%}** |",
        f"| **Macro F1** | **{result['macro_f1']:.3f}** |",
        f"| Weighted F1 | {result['weighted_f1']:.3f} |",
        f"| Macro Precision | {result['macro_precision']:.3f} |",
        f"| Macro Recall | {result['macro_recall']:.3f} |",
        f"| Train clips | {result['n_train']} |",
        f"| Val clips   | {result['n_test']} |",
        "",
        "## Charts",
        "",
    ]
    for name, path in charts.items():
        if path:
            rel = Path(path).name
            lines.append(f"- **{name}**: `{rel}`")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== Analysis complete ===")
    print(f"  Accuracy  : {result['accuracy']:.1%}")
    print(f"  Macro F1  : {result['macro_f1']:.3f}")
    print(f"  Charts    : {len(charts)} saved → {out_dir}")

    return {
        "eval":       result,
        "summary":    summary,
        "charts":     charts,
        "out_dir":    out_dir,
        "report_path": out_dir / "REPORT.md",
    }

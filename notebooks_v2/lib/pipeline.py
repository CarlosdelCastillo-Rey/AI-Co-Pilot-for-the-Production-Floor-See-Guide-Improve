"""v2 HAR pipeline orchestrator — all improvements wired end-to-end."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lib.constants import (
    BACKBONE_DINOV2,
    BACKBONE_VJEPA,
    BLOCKED_ACTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_FRAMES,
    DEFAULT_CLIPS_PER_CLASS,
    DEFAULT_DWELL_WINDOWS,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_HEAD_ARCH,
    DEFAULT_INFER_EVERY,
    DEFAULT_INHARD_VIEW,
    DEFAULT_LR,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIXUP_ALPHA,
    DEFAULT_MIXUP_N_AUG,
    DEFAULT_SEED,
    DEFAULT_SPLIT_MODE,
    DEFAULT_STALE_SEC,
    DEFAULT_SUPCON_EPOCHS,
    DEFAULT_TEMPORAL_AGG,
    DEFAULT_TEST_SIZE,
    DEFAULT_USE_FOCAL_LOSS,
    DEFAULT_USE_MIXUP,
    DEFAULT_USE_WEIGHTED_SAMPLER,
    TRAINABLE_ACTIONS,
)
from lib.crop_extract import crops_for_embedding, crops_for_embedding_with_meta
from lib.dino_embeddings import extract_dinov2_embedding
from lib.embeddings import extract_vjepa_embedding
from lib.har_train import train_from_npz
from lib.human_labels import trainable_human_rows
from lib.inhard import analyze_training_clips, label_counts, save_step01_summary
from lib.paths import CHECKPOINTS_DIR, OUTPUTS_DIR, default_mock_video, find_inhard_root, list_mock_videos

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    exclude_train: tuple[str, ...] = BLOCKED_ACTIONS
    exclude_infer: tuple[str, ...] = BLOCKED_ACTIONS
    clips_per_class: int | None    = DEFAULT_CLIPS_PER_CLASS   # None → ALL clips
    max_clips: int | None          = None
    sample_seed: int               = DEFAULT_SEED
    inhard_view: str               = DEFAULT_INHARD_VIEW        # "topdown"|"side"|"front"|"full"

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone: str             = BACKBONE_VJEPA
    backbones: tuple[str,...] = (BACKBONE_VJEPA, BACKBONE_DINOV2)
    embedding_mode: str       = "yolo_crop"                     # "yolo_crop"|"full_clip"
    temporal_agg: str         = DEFAULT_TEMPORAL_AGG            # "attention"|"mean"

    # ── Training ─────────────────────────────────────────────────────────────
    head_arch: str              = DEFAULT_HEAD_ARCH             # "mlp"|"gru"
    train_epochs: int           = DEFAULT_EPOCHS
    batch_size: int             = DEFAULT_BATCH_SIZE
    lr: float                   = DEFAULT_LR
    split_mode: str             = DEFAULT_SPLIT_MODE
    test_size: float            = DEFAULT_TEST_SIZE
    use_focal_loss: bool        = DEFAULT_USE_FOCAL_LOSS
    focal_gamma: float          = DEFAULT_FOCAL_GAMMA
    use_weighted_sampler: bool  = DEFAULT_USE_WEIGHTED_SAMPLER
    use_mixup: bool             = DEFAULT_USE_MIXUP
    mixup_alpha: float          = DEFAULT_MIXUP_ALPHA
    mixup_n_aug: int            = DEFAULT_MIXUP_N_AUG
    use_supcon: bool            = False
    supcon_epochs: int          = DEFAULT_SUPCON_EPOCHS

    # ── Cache / skip ──────────────────────────────────────────────────────────
    skip_embeddings_if_exists: bool = False
    skip_train_if_exists: bool      = False

    # ── Inference / eval ──────────────────────────────────────────────────────
    min_confidence: float  = DEFAULT_MIN_CONFIDENCE
    infer_every: int       = DEFAULT_INFER_EVERY
    buffer_frames: int     = DEFAULT_BUFFER_FRAMES
    dwell_windows: int     = DEFAULT_DWELL_WINDOWS
    eval_max_frames: int   = 600

    # ── Pipeline gates ────────────────────────────────────────────────────────
    run_data_check:   bool = True
    run_embeddings:   bool = True
    run_train:        bool = True
    run_analysis:     bool = True
    run_compare:      bool = True
    run_eval_video:   bool = False
    run_live_app:     bool = False
    analysis_split:   str  = "subject"
    include_human_labels: bool = True
    live_webcam: int | None    = None
    live_max_seconds: float | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_tag(cfg: PipelineConfig) -> str:
    n = str(cfg.clips_per_class) + "each" if cfg.clips_per_class else "allavail"
    return f"12c_{cfg.inhard_view}_{n}"


def ckpt_prefix(backbone: str) -> str:
    return "har_vjepa" if backbone == BACKBONE_VJEPA else "har_dinov2"


def ckpt_name(cfg: PipelineConfig, backbone: str) -> str:
    return f"{ckpt_prefix(backbone)}_{run_tag(cfg)}.pt"


def npz_name(backbone: str) -> str:
    return "embeddings.npz" if backbone == BACKBONE_VJEPA else "embeddings_dinov2.npz"


def _log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def _extract_embedding(frames_bgr: list, *, backbone: str, temporal_agg: str = "attention"):
    if backbone == BACKBONE_DINOV2:
        return extract_dinov2_embedding(frames_bgr, temporal_agg=temporal_agg)
    return extract_vjepa_embedding(frames_bgr)


# ── Step 01: data check ───────────────────────────────────────────────────────

def step_data_check(cfg: PipelineConfig) -> dict:
    root   = find_inhard_root()
    report = analyze_training_clips(
        root,
        exclude_labels=cfg.exclude_train,
        min_classes=len(TRAINABLE_ACTIONS),
        clips_per_class=cfg.clips_per_class,
        max_clips=cfg.max_clips,
        seed=cfg.sample_seed,
    )
    if report.error:
        for line in report.error.split("\n"):
            _log("01", line)
    else:
        by_label = label_counts(report.clips)
        _log("01", f"Clips: {len(report.clips)} across {report.n_classes} classes")
        _log("01", f"  view={cfg.inhard_view}  clips_per_class={cfg.clips_per_class or 'ALL'}")
        _log("01", f"  imbalance ratio={by_label.max()/max(by_label.min(),1):.1f}x  "
                   f"(max={by_label.max()} min={by_label.min()})")
    out   = save_step01_summary(report, root)
    mocks = [p.name for p in list_mock_videos()]
    _log("01", f"Mock videos: {mocks or '(none)'}")
    return {"summary": str(out), "ok": report.ok, "n_clips": len(report.clips)}


# ── Step 02: embeddings ───────────────────────────────────────────────────────

def step_embeddings(cfg: PipelineConfig) -> dict:
    root   = find_inhard_root()
    report = analyze_training_clips(
        root,
        exclude_labels=cfg.exclude_train,
        clips_per_class=cfg.clips_per_class,
        max_clips=cfg.max_clips,
        seed=cfg.sample_seed,
    )
    if not report.ok:
        raise FileNotFoundError(report.error)

    classes      = sorted({c.label for c in report.clips})
    label_to_idx = {c: i for i, c in enumerate(classes)}
    clips        = report.clips
    _log("02", f"Extracting embeddings for {len(clips)} clips · view={cfg.inhard_view}")

    # YOLO crops once (shared across backbones)
    _log("02", "Pass 1: YOLO crops …")
    crop_cache: list[tuple] = []
    for rec in tqdm(clips, desc="YOLO crops"):
        crop_frames = crops_for_embedding(
            rec.path,
            mode=cfg.embedding_mode,
            view=cfg.inhard_view,
            buffer_frames=cfg.buffer_frames,
        )
        if crop_frames:
            crop_cache.append((rec, crop_frames))
    _log("02", f"  {len(crop_cache)}/{len(clips)} clips had valid crops")

    out: dict[str, dict] = {}
    for backbone in cfg.backbones:
        npz_path = OUTPUTS_DIR / npz_name(backbone)
        meta_csv = OUTPUTS_DIR / ("embedding_meta_dinov2.csv" if backbone == BACKBONE_DINOV2 else "embedding_meta.csv")

        if cfg.skip_embeddings_if_exists and npz_path.is_file():
            _log("02", f"Skip {backbone} — {npz_path.name} exists")
            out[backbone] = {"path": str(npz_path), "skipped": True}
            continue

        _log("02", f"Pass 2: {backbone} encoding {len(crop_cache)} clips …")
        min_frames = 1 if backbone == BACKBONE_DINOV2 else 4
        X_list, y_list, subjects, meta_rows = [], [], [], []

        for rec, crop_frames in tqdm(crop_cache, desc=f"{backbone} encode"):
            if len(crop_frames) < min_frames:
                continue
            try:
                emb = _extract_embedding(crop_frames, backbone=backbone, temporal_agg=cfg.temporal_agg)
            except Exception as exc:
                _log("02", f"  skip {rec.path.name}: {exc}")
                continue
            X_list.append(emb)
            y_list.append(label_to_idx[rec.label])
            subjects.append(rec.subject or "unknown")
            meta_rows.append({
                "path": str(rec.path), "label": rec.label,
                "subject": rec.subject, "session": rec.session,
                "source": "inhard", "backbone": backbone,
                "view": cfg.inhard_view,
            })

        # Merge human-review labels
        if cfg.include_human_labels:
            n_human = _merge_human_samples(
                classes, label_to_idx, X_list, y_list, meta_rows, subjects,
                backbone=backbone, temporal_agg=cfg.temporal_agg,
            )
            if n_human:
                _log("02", f"  +{n_human} human-verified samples")

        if not X_list:
            raise RuntimeError(f"No embeddings extracted for backbone={backbone}")

        X    = np.stack(X_list).astype(np.float32)
        y    = np.array(y_list, dtype=np.int64)
        subj = np.array(subjects, dtype=object)
        np.savez(npz_path, X=X, y=y,
                 class_names=np.array(classes, dtype=object),
                 subjects=subj, backbone=np.array(backbone))
        pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
        _log("02", f"  Saved {X.shape} → {npz_path.name}")

        # Class distribution report
        from collections import Counter
        counts = Counter(rec.label for rec, _ in crop_cache)
        _log("02", f"  Class distribution: {dict(sorted(counts.items()))}")

        out[backbone] = {
            "path": str(npz_path), "n_samples": len(X),
            "n_classes": len(classes), "backbone": backbone,
            "view": cfg.inhard_view,
        }
    return out


def _merge_human_samples(
    classes, label_to_idx, X_list, y_list, meta_rows, subjects,
    *, backbone, temporal_agg="attention",
) -> int:
    from lib.crop_extract import crops_from_jpeg
    n = 0
    for row in trainable_human_rows():
        label = str(row.get("correct_label") or row.get("predicted_label") or "").strip()
        if label not in label_to_idx:
            continue
        emb_path  = str(row.get("embedding_path") or "").strip()
        crop_path = str(row.get("crop_path") or row.get("image_path") or "").strip()
        emb = None
        if backbone == BACKBONE_VJEPA and emb_path and Path(emb_path).is_file():
            emb = np.load(emb_path).astype(np.float32)
        elif crop_path and Path(crop_path).is_file():
            crops = crops_from_jpeg(crop_path)
            if len(crops) < (1 if backbone == BACKBONE_DINOV2 else 4):
                continue
            try:
                emb = _extract_embedding(crops, backbone=backbone, temporal_agg=temporal_agg)
            except Exception:
                continue
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(label_to_idx[label])
        subjects.append("human_review")
        meta_rows.append({"path": crop_path or emb_path, "label": label, "source": "human", "backbone": backbone})
        n += 1
    return n


# ── Step 03: train ────────────────────────────────────────────────────────────

def step_train(cfg: PipelineConfig) -> dict:
    out: dict[str, dict] = {}
    for backbone in cfg.backbones:
        npz_path = OUTPUTS_DIR / npz_name(backbone)
        if not npz_path.is_file():
            _log("03", f"Skip {backbone} — npz missing, run embeddings first")
            out[backbone] = {"skipped": True, "reason": "no_npz"}
            continue

        ckpt = CHECKPOINTS_DIR / ckpt_name(cfg, backbone)
        if cfg.skip_train_if_exists and ckpt.is_file():
            _log("03", f"Skip {backbone} — checkpoint exists")
            out[backbone] = {"skipped": True, "reason": "exists", "checkpoint": str(ckpt)}
            continue

        _log("03", f"Training {backbone} head …")
        stats = train_from_npz(
            npz_path, ckpt,
            exclude_labels=list(cfg.exclude_infer),
            epochs=cfg.train_epochs,
            split_mode=cfg.split_mode,
            head_arch=cfg.head_arch,
            use_focal_loss=cfg.use_focal_loss,
            use_weighted_sampler=cfg.use_weighted_sampler,
            use_mixup=cfg.use_mixup,
            use_supcon=cfg.use_supcon,
            backbone=backbone,
        )
        val_report = stats.get("val_report", {})
        _log("03", f"  {backbone} → acc={val_report.get('accuracy', '?'):.3f}  "
                   f"macro_f1={val_report.get('macro avg', {}).get('f1-score', '?'):.3f}  "
                   f"ckpt={ckpt.name}")
        out[backbone] = stats
    return out


# ── Step 04: analysis ─────────────────────────────────────────────────────────

def step_analysis(cfg: PipelineConfig, training_results: dict | None = None) -> dict:
    from lib.har_analysis import run_v2_analysis
    from lib.inhard import collect_trainable_clips, resolve_training_clips

    out: dict[str, dict] = {}
    for backbone in cfg.backbones:
        ckpt     = CHECKPOINTS_DIR / ckpt_name(cfg, backbone)
        npz_path = OUTPUTS_DIR / npz_name(backbone)
        if not ckpt.is_file():
            out[backbone] = {"skipped": True, "reason": "no_checkpoint"}
            continue

        _log("06", f"Analysis {backbone} …")
        # Build sample clips for strip and YOLO grid
        clips = resolve_training_clips(
            find_inhard_root(),
            exclude_labels=cfg.exclude_train,
            clips_per_class=5,
            seed=cfg.sample_seed,
        )
        by_class: dict[str, list[Path]] = {}
        for rec in clips:
            by_class.setdefault(rec.label, []).append(rec.path)
        yolo_sample = [(rec.path, rec.label) for rec in clips[:16]]

        # Pick an example clip for 3-view comparison
        example = clips[0].path if clips else None

        # Training history from results
        hist = None
        if training_results and backbone in training_results:
            hist = training_results[backbone].get("history")

        try:
            r = run_v2_analysis(
                ckpt,
                npz_path=npz_path,
                split=cfg.analysis_split,
                sample_clips_for_strip=by_class,
                yolo_sample_clips=yolo_sample,
                training_history=hist,
                example_clip_for_3view=example,
            )
            out[backbone] = {
                "out_dir":  str(r["out_dir"]),
                "accuracy": r["summary"]["accuracy"],
                "macro_f1": r["summary"]["macro_f1"],
                "n_test":   r["summary"]["n_test"],
                "report_path": str(r["report_path"]),
                "charts":   r["charts"],
            }
            _log("06", f"  {backbone}: acc={r['summary']['accuracy']:.1%}  "
                       f"macro_f1={r['summary']['macro_f1']:.3f}  → {r['out_dir']}")
        except Exception as exc:
            _log("06", f"  Analysis failed ({backbone}): {exc}")
            out[backbone] = {"error": str(exc)}
    return out


# ── Step 05: backbone comparison ─────────────────────────────────────────────

def step_compare(cfg: PipelineConfig) -> dict:
    from lib.har_analysis import eval_checkpoint, load_embeddings_npz

    tag  = run_tag(cfg)
    rows = []
    for backbone in cfg.backbones:
        ckpt     = CHECKPOINTS_DIR / ckpt_name(cfg, backbone)
        npz_path = OUTPUTS_DIR / npz_name(backbone)
        if not ckpt.is_file() or not npz_path.is_file():
            rows.append({"backbone": backbone, "status": "missing"})
            continue
        r = eval_checkpoint(ckpt, npz_path=npz_path, split=cfg.analysis_split)
        rows.append({
            "backbone":    backbone,
            "checkpoint":  ckpt.name,
            "status":      "ok",
            "accuracy":    r["accuracy"],
            "macro_f1":    r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "n_test":      r["n_test"],
        })

    df  = pd.DataFrame(rows)
    out = OUTPUTS_DIR / f"backbone_comparison_{tag}.json"
    payload = {"tag": tag, "results": df.to_dict(orient="records")}
    ok = df[df.get("status", "") == "ok"] if "status" in df.columns else df
    if len(ok) >= 1 and "macro_f1" in ok.columns:
        best = ok.sort_values("macro_f1", ascending=False).iloc[0]
        payload["winner"] = {
            "backbone": str(best["backbone"]),
            "macro_f1": float(best["macro_f1"]),
            "accuracy": float(best["accuracy"]),
        }
        _log("compare", f"Winner: {payload['winner']['backbone']}  "
                        f"F1={payload['winner']['macro_f1']:.4f}  "
                        f"acc={payload['winner']['accuracy']:.4f}")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ── Step 06: eval video ───────────────────────────────────────────────────────

def step_eval_video(cfg: PipelineConfig) -> dict:
    from lib.eval_video import render_eval_video
    out: dict[str, dict] = {}
    video = default_mock_video()
    if video is None:
        _log("04", "No mock video found — skipping eval render")
        return {"skipped": True}
    for backbone in cfg.backbones:
        ckpt = CHECKPOINTS_DIR / ckpt_name(cfg, backbone)
        if not ckpt.is_file():
            continue
        out_path = OUTPUTS_DIR / f"perperson_eval_{ckpt_name(cfg, backbone).replace('.pt','.mp4')}"
        _log("04", f"Rendering {video.name} → {out_path.name}")
        out[backbone] = render_eval_video(
            video_path=video, checkpoint=ckpt, output_path=out_path,
            max_frames=cfg.eval_max_frames, infer_every=cfg.infer_every,
            buffer_frames=cfg.buffer_frames, dwell_windows=cfg.dwell_windows,
            min_confidence=cfg.min_confidence,
        )
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "config":           asdict(cfg),
        "steps":            {},
        "status":           "ok",
        "pipeline_version": "v2",
        "backbones":        list(cfg.backbones),
        "improvements": [
            "all_clips_no_cap",
            f"view={cfg.inhard_view}",
            f"temporal_agg={cfg.temporal_agg}",
            f"head={cfg.head_arch}",
            f"focal_loss={cfg.use_focal_loss}",
            f"weighted_sampler={cfg.use_weighted_sampler}",
            f"mixup={cfg.use_mixup}",
            f"supcon={cfg.use_supcon}",
            f"epochs={cfg.train_epochs}",
        ],
    }

    try:
        if cfg.run_data_check:
            results["steps"]["01_data"] = step_data_check(cfg)
            if not results["steps"]["01_data"].get("ok"):
                results["status"] = "blocked_no_train_data"
                cfg = replace(cfg, run_embeddings=False, run_train=False, run_analysis=False)

        if cfg.run_embeddings:
            results["steps"]["02_embeddings"] = step_embeddings(cfg)

        train_results = None
        if cfg.run_train:
            train_results = step_train(cfg)
            results["steps"]["03_train"] = train_results

        if cfg.run_analysis:
            results["steps"]["06_analysis"] = step_analysis(cfg, train_results)

        if cfg.run_compare and len(cfg.backbones) >= 2:
            results["steps"]["07_compare"] = step_compare(cfg)

        if cfg.run_eval_video:
            results["steps"]["04_eval"] = step_eval_video(cfg)

    except Exception as exc:
        results["status"] = "error"
        results["error"]  = str(exc)
        raise
    finally:
        summary_path = OUTPUTS_DIR / "pipeline_v2_run_summary.json"
        summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        _log("pipeline", f"Summary → {summary_path}")

    return results

"""Orchestrate unified HAR pipeline (01→07): crop-aligned embeddings + HITL."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lib.constants import (
    BLOCKED_ACTIONS,
    DEFAULT_EMBEDDING_MODE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SPLIT_MODE,
    DEFAULT_USE_CLASS_WEIGHTS,
    TRAINABLE_ACTIONS,
)
from lib.crop_extract import crops_for_embedding, crops_from_jpeg
from lib.embeddings import extract_vjepa_embedding
from lib.eval_video import render_eval_video
from lib.har_train import train_from_npz
from lib.human_labels import trainable_human_rows
from lib.inhard import analyze_training_clips, label_counts, save_step01_summary
from lib.live_app import run_live
from lib.pipeline_cache import (
    EMBEDDINGS_NPZ,
    apply_cache_skips,
    embeddings_cache_valid,
    save_embeddings_manifest,
    save_train_manifest,
    train_cache_valid,
)
from lib.paths import CHECKPOINTS_DIR, OUTPUTS_DIR, default_mock_video, find_inhard_root, list_mock_videos

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # v2 defaults: 12 trainable classes, crop-aligned, subject split
    exclude_train: tuple[str, ...] = BLOCKED_ACTIONS
    exclude_infer: tuple[str, ...] = BLOCKED_ACTIONS
    min_train_classes: int = len(TRAINABLE_ACTIONS)
    max_clips: int | None = None
    clips_per_class: int | None = 100
    sample_seed: int = 42
    train_epochs: int = 25

    embedding_mode: str = DEFAULT_EMBEDDING_MODE
    split_mode: str = DEFAULT_SPLIT_MODE
    use_class_weights: bool = DEFAULT_USE_CLASS_WEIGHTS
    include_human_labels: bool = True
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    skip_embeddings_if_exists: bool = False
    skip_train_if_checkpoint_exists: bool = False
    skip_eval_if_exists: bool = False
    skip_if_cached: bool = False

    eval_max_frames: int = 600
    infer_every: int = 16
    buffer_frames: int = 32
    dwell_windows: int = 2

    run_data_check: bool = True
    run_embeddings: bool = True
    run_train: bool = True
    run_eval_video: bool = True
    run_live_app: bool = False
    live_webcam: int | None = None
    live_max_seconds: float | None = None

    checkpoint_name: str = "har_vjepa_12c_crop_100each.pt"
    eval_video_name: str = "perperson_eval_12c_crop.mp4"


def _log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def step_data_check(cfg: PipelineConfig) -> dict:
    root = find_inhard_root()
    report = analyze_training_clips(
        root,
        exclude_labels=cfg.exclude_train,
        min_classes=cfg.min_train_classes,
        max_clips=cfg.max_clips,
        clips_per_class=cfg.clips_per_class,
        seed=cfg.sample_seed,
    )
    if report.error:
        for line in report.error.split("\n"):
            _log("01", line)
    else:
        by_label = label_counts(report.clips)
        _log("01", f"Sample: {len(report.clips)} clips, {report.n_classes} classes")
        if cfg.clips_per_class:
            _log("01", f"  {cfg.clips_per_class} clips/class · mode={cfg.embedding_mode}")
        _log("01", f"  Labels: {', '.join(by_label.index[:6])}{'…' if len(by_label) > 6 else ''}")
    out = save_step01_summary(report, root)
    mocks = [p.name for p in list_mock_videos()]
    _log("01", f"Mock videos: {mocks or '(none)'}")
    return {"summary": str(out), "ok": report.ok, "n_clips": len(report.clips)}


def _merge_human_samples(
    classes: list[str],
    label_to_idx: dict[str, int],
    X_list: list[np.ndarray],
    y_list: list[int],
    meta_rows: list[dict],
    subjects: list[str],
) -> int:
    from lib.embeddings import extract_vjepa_embedding

    n_added = 0
    for row in trainable_human_rows():
        label = str(row.get("correct_label") or row.get("predicted_label") or "").strip()
        if label not in label_to_idx:
            continue

        emb_path = str(row.get("embedding_path") or "").strip()
        crop_path = str(row.get("crop_path") or row.get("image_path") or "").strip()

        if emb_path and Path(emb_path).is_file():
            emb = np.load(emb_path).astype(np.float32)
        elif crop_path and Path(crop_path).is_file():
            crops = crops_from_jpeg(crop_path)
            if len(crops) < 4:
                continue
            try:
                emb = extract_vjepa_embedding(crops)
            except Exception:
                continue
        else:
            continue

        X_list.append(emb)
        y_list.append(label_to_idx[label])
        subjects.append("human_review")
        meta_rows.append(
            {
                "path": crop_path or emb_path,
                "label": label,
                "source": "human",
                "label_id": row.get("label_id"),
            }
        )
        n_added += 1
    return n_added


def step_embeddings(cfg: PipelineConfig) -> dict:
    npz_path = EMBEDDINGS_NPZ
    meta_path = OUTPUTS_DIR / "embedding_meta.csv"

    if cfg.skip_if_cached and embeddings_cache_valid(cfg):
        manifest = json.loads((OUTPUTS_DIR / "embeddings.manifest.json").read_text(encoding="utf-8"))
        _log("02", f"Skip — cached embeddings ({manifest.get('n_samples')} samples)")
        return {"path": str(npz_path), "skipped": True, "reason": "cache_hit", **manifest}
    if cfg.skip_embeddings_if_exists and npz_path.is_file():
        _log("02", f"Skip — embeddings exist ({npz_path.name})")
        return {"path": str(npz_path), "skipped": True, "reason": "exists"}

    root = find_inhard_root()
    report = analyze_training_clips(
        root,
        exclude_labels=cfg.exclude_train,
        min_classes=cfg.min_train_classes,
        max_clips=cfg.max_clips,
        clips_per_class=cfg.clips_per_class,
        seed=cfg.sample_seed,
    )
    if not report.ok:
        raise FileNotFoundError(report.error or "No trainable clips")

    clips = report.clips
    classes = sorted({c.label for c in clips})
    label_to_idx = {c: i for i, c in enumerate(classes)}
    _log("02", f"Extracting {len(clips)} InHARD clips · mode={cfg.embedding_mode}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []
    subjects: list[str] = []

    for rec in tqdm(clips, desc="V-JEPA2 crop"):
        crop_frames = crops_for_embedding(
            rec.path,
            mode=cfg.embedding_mode,
            buffer_frames=cfg.buffer_frames,
        )
        if len(crop_frames) < 4:
            continue
        try:
            emb = extract_vjepa_embedding(crop_frames)
        except Exception as exc:
            _log("02", f"skip {rec.path.name}: {exc}")
            continue
        X_list.append(emb)
        y_list.append(label_to_idx[rec.label])
        subjects.append(rec.subject or "unknown")
        meta_rows.append(
            {
                "path": str(rec.path),
                "label": rec.label,
                "subject": rec.subject,
                "session": rec.session,
                "source": "inhard",
                "embedding_mode": cfg.embedding_mode,
            }
        )

    n_human = 0
    if cfg.include_human_labels:
        n_human = _merge_human_samples(classes, label_to_idx, X_list, y_list, meta_rows, subjects)
        if n_human:
            _log("02", f"Merged {n_human} human-verified samples")

    if not X_list:
        raise RuntimeError("No embeddings extracted")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subj = np.array(subjects, dtype=object)
    np.savez(npz_path, X=X, y=y, class_names=np.array(classes, dtype=object), subjects=subj)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    save_embeddings_manifest(cfg, n_samples=len(X), n_classes=len(classes), n_human=n_human)
    _log("02", f"Saved {X.shape} → {npz_path} (+{n_human} human)")
    return {
        "path": str(npz_path),
        "n_samples": len(X),
        "n_classes": len(classes),
        "n_human": n_human,
        "embedding_mode": cfg.embedding_mode,
    }


def step_train(cfg: PipelineConfig) -> dict:
    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    if cfg.skip_if_cached and train_cache_valid(cfg):
        _log("03", f"Skip — checkpoint matches config → {ckpt.name}")
        return {"checkpoint": str(ckpt), "skipped": True, "reason": "cache_hit"}
    if cfg.skip_train_if_checkpoint_exists and ckpt.is_file() and not cfg.skip_if_cached:
        _log("03", f"Skip — exists {ckpt}")
        return {"checkpoint": str(ckpt), "skipped": True}

    npz_path = EMBEDDINGS_NPZ
    if not npz_path.is_file():
        raise FileNotFoundError(f"Run embeddings first: {npz_path}")

    stats = train_from_npz(
        npz_path,
        ckpt,
        exclude_labels=list(cfg.exclude_infer),
        epochs=cfg.train_epochs,
        split_mode=cfg.split_mode,
        use_class_weights=cfg.use_class_weights,
    )
    save_train_manifest(cfg)
    _log("03", f"Checkpoint → {ckpt} (split={cfg.split_mode}, weights={cfg.use_class_weights})")
    return stats


def step_eval_video(cfg: PipelineConfig) -> dict:
    out = OUTPUTS_DIR / cfg.eval_video_name
    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    if not ckpt.is_file():
        raise FileNotFoundError(f"Train first: {ckpt}")
    if cfg.skip_eval_if_exists and out.is_file():
        _log("04", f"Skip — exists {out}")
        return {"output": str(out), "skipped": True}

    video = default_mock_video()
    if video is None:
        raise FileNotFoundError("Add a mock .mp4 to data_sample/mock-videos/")

    _log("04", f"Rendering {video.name} → {out.name}")
    return render_eval_video(
        video_path=video,
        checkpoint=ckpt,
        output_path=out,
        max_frames=cfg.eval_max_frames,
        infer_every=cfg.infer_every,
        buffer_frames=cfg.buffer_frames,
        dwell_windows=cfg.dwell_windows,
        min_confidence=cfg.min_confidence,
    )


def step_live_app(cfg: PipelineConfig) -> dict:
    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    if not ckpt.is_file():
        raise FileNotFoundError(f"Train first: {ckpt}")
    video = None if cfg.live_webcam is not None else default_mock_video()
    _log("05", "Opening live window (q to quit)")
    code = run_live(
        checkpoint=ckpt,
        video=video,
        webcam=cfg.live_webcam,
        infer_every=cfg.infer_every,
        buffer_frames=cfg.buffer_frames,
        dwell_windows=cfg.dwell_windows,
        max_seconds=cfg.live_max_seconds,
        min_confidence=cfg.min_confidence,
    )
    return {"exit_code": code}


def run_pipeline(cfg: PipelineConfig) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg, cache_info = apply_cache_skips(cfg)
    results: dict = {"config": asdict(cfg), "cache": cache_info, "steps": {}, "status": "ok", "pipeline_version": "unified"}

    try:
        if cfg.run_data_check:
            results["steps"]["01_data"] = step_data_check(cfg)
            if not results["steps"]["01_data"].get("ok"):
                results["status"] = "blocked_no_train_data"
                if cfg.run_embeddings or cfg.run_train:
                    _log("pipeline", "Steps 02–03 skipped — no InHARD clips")
                    cfg.run_embeddings = False
                    cfg.run_train = False

        if cfg.run_embeddings:
            results["steps"]["02_embeddings"] = step_embeddings(cfg)
        if cfg.run_train:
            results["steps"]["03_train"] = step_train(cfg)
        if cfg.run_eval_video:
            if (CHECKPOINTS_DIR / cfg.checkpoint_name).is_file():
                results["steps"]["04_eval"] = step_eval_video(cfg)
            else:
                _log("04", "Skipped — no checkpoint")
                results["steps"]["04_eval"] = {"skipped": True, "reason": "no_checkpoint"}
        if cfg.run_live_app:
            results["steps"]["05_live"] = step_live_app(cfg)
    except Exception as exc:
        results["status"] = "error"
        results["error"] = str(exc)
        raise
    finally:
        summary_path = OUTPUTS_DIR / "pipeline_run_summary.json"
        summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        _log("pipeline", f"Summary → {summary_path}")

    return results

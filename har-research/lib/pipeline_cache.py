"""Config-aware cache for pipeline v2 (crop mode, split, human labels)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.constants import BACKBONE_DINOV2, BACKBONE_VJEPA, DINOV2_HUB_MODEL, VJEPA_MODEL_ID
from lib.paths import CHECKPOINTS_DIR, HUMAN_LABELS_DIR, OUTPUTS_DIR, find_inhard_root

EMBEDDINGS_MANIFEST = OUTPUTS_DIR / "embeddings.manifest.json"
EMBEDDINGS_NPZ = OUTPUTS_DIR / "embeddings.npz"


def backbone_name(cfg: Any) -> str:
    b = getattr(cfg, "backbone", BACKBONE_VJEPA)
    if b in ("dinov3", BACKBONE_DINOV2):
        return BACKBONE_DINOV2
    return BACKBONE_VJEPA


def embeddings_npz_for(cfg: Any) -> Path:
    if backbone_name(cfg) == BACKBONE_DINOV2:
        return OUTPUTS_DIR / "embeddings_dinov2.npz"
    return EMBEDDINGS_NPZ


def embeddings_manifest_for(cfg: Any) -> Path:
    if backbone_name(cfg) == BACKBONE_DINOV2:
        return OUTPUTS_DIR / "embeddings_dinov2.manifest.json"
    return EMBEDDINGS_MANIFEST


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _human_labels_fingerprint() -> str:
    csv_path = HUMAN_LABELS_DIR / "labels.csv"
    if not csv_path.is_file():
        return "none"
    stat = csv_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def embedding_config_dict(cfg: Any) -> dict[str, Any]:
    bb = backbone_name(cfg)
    payload = {
        "step": "embeddings",
        "pipeline_version": "v2",
        "backbone": bb,
        "clips_per_class": cfg.clips_per_class,
        "max_clips": cfg.max_clips,
        "sample_seed": cfg.sample_seed,
        "exclude_train": sorted(cfg.exclude_train),
        "min_train_classes": cfg.min_train_classes,
        "embedding_mode": getattr(cfg, "embedding_mode", "yolo_crop"),
        "include_human_labels": getattr(cfg, "include_human_labels", True),
        "human_labels_fp": _human_labels_fingerprint(),
    }
    if bb == BACKBONE_DINOV2:
        payload["dino_model"] = DINOV2_HUB_MODEL
    else:
        payload["vjepa_model"] = VJEPA_MODEL_ID
    return payload


def train_config_dict(cfg: Any, *, embeddings_fingerprint: str) -> dict[str, Any]:
    return {
        "step": "train",
        "pipeline_version": "v2",
        "checkpoint_name": cfg.checkpoint_name,
        "train_epochs": cfg.train_epochs,
        "exclude_infer": sorted(cfg.exclude_infer),
        "split_mode": getattr(cfg, "split_mode", "subject"),
        "use_class_weights": getattr(cfg, "use_class_weights", True),
        "embeddings_fingerprint": embeddings_fingerprint,
    }


def checkpoint_manifest_path(cfg: Any) -> Path:
    return CHECKPOINTS_DIR / f"{Path(cfg.checkpoint_name).stem}.pipeline.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def embeddings_cache_valid(cfg: Any) -> bool:
    npz_path = embeddings_npz_for(cfg)
    manifest_path = embeddings_manifest_for(cfg)
    if not npz_path.is_file():
        return False
    manifest = load_json(manifest_path)
    if not manifest:
        return False
    expected = _fingerprint(embedding_config_dict(cfg))
    return manifest.get("fingerprint") == expected


def train_cache_valid(cfg: Any) -> bool:
    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    if not ckpt.is_file():
        return False
    manifest = load_json(checkpoint_manifest_path(cfg))
    if not manifest:
        return False
    emb_fp = _current_embeddings_fingerprint(cfg)
    expected = _fingerprint(train_config_dict(cfg, embeddings_fingerprint=emb_fp))
    return manifest.get("fingerprint") == expected and manifest.get("embeddings_fingerprint") == emb_fp


def _current_embeddings_fingerprint(cfg: Any) -> str:
    manifest = load_json(embeddings_manifest_for(cfg))
    if manifest and manifest.get("fingerprint"):
        return str(manifest["fingerprint"])
    return _fingerprint(embedding_config_dict(cfg))


def save_embeddings_manifest(cfg: Any, *, n_samples: int, n_classes: int, n_human: int = 0) -> Path:
    config = embedding_config_dict(cfg)
    npz_path = embeddings_npz_for(cfg)
    manifest_path = embeddings_manifest_for(cfg)
    payload = {
        "fingerprint": _fingerprint(config),
        "config": config,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "n_human_samples": n_human,
        "npz_path": npz_path.name,
        "inhard_root": str(find_inhard_root() or ""),
        "created_at": _utc_now(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def save_train_manifest(cfg: Any) -> Path:
    emb_fp = _current_embeddings_fingerprint(cfg)
    config = train_config_dict(cfg, embeddings_fingerprint=emb_fp)
    path = checkpoint_manifest_path(cfg)
    payload = {
        "fingerprint": _fingerprint(config),
        "config": config,
        "embeddings_fingerprint": emb_fp,
        "checkpoint": cfg.checkpoint_name,
        "created_at": _utc_now(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def cache_status(cfg: Any) -> dict[str, Any]:
    npz_path = embeddings_npz_for(cfg)
    manifest_path = embeddings_manifest_for(cfg)
    emb_manifest = load_json(manifest_path)
    train_manifest = load_json(checkpoint_manifest_path(cfg))
    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    return {
        "backbone": backbone_name(cfg),
        "embeddings_npz": npz_path.is_file(),
        "embeddings_manifest": manifest_path.is_file(),
        "embeddings_valid": embeddings_cache_valid(cfg),
        "embeddings_fingerprint_expected": _fingerprint(embedding_config_dict(cfg)),
        "embeddings_fingerprint_saved": (emb_manifest or {}).get("fingerprint"),
        "embeddings_n_samples": (emb_manifest or {}).get("n_samples"),
        "embeddings_n_human": (emb_manifest or {}).get("n_human_samples"),
        "checkpoint_exists": ckpt.is_file(),
        "train_manifest": checkpoint_manifest_path(cfg).is_file(),
        "train_valid": train_cache_valid(cfg),
        "train_fingerprint_expected": _fingerprint(
            train_config_dict(cfg, embeddings_fingerprint=_current_embeddings_fingerprint(cfg))
        ),
        "train_fingerprint_saved": (train_manifest or {}).get("fingerprint"),
        "checkpoint_name": cfg.checkpoint_name,
    }


def _expected_sample_count(cfg: Any) -> int | None:
    try:
        from lib.inhard import analyze_training_clips

        report = analyze_training_clips(
            exclude_labels=tuple(cfg.exclude_train),
            min_classes=int(cfg.min_train_classes),
            max_clips=cfg.max_clips,
            clips_per_class=cfg.clips_per_class,
            seed=int(cfg.sample_seed),
        )
        return len(report.clips) if report.ok else None
    except Exception:
        return None


def backfill_cache_manifests(cfg: Any, *, force: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"embeddings_backfilled": False, "train_backfilled": False}
    npz_path = embeddings_npz_for(cfg)
    manifest_path = embeddings_manifest_for(cfg)
    if not npz_path.is_file():
        return result

    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    n_samples = int(len(data["X"]))
    n_classes = int(len(data["class_names"]))
    expected = _expected_sample_count(cfg)
    sample_ok = expected is None or abs(n_samples - expected) <= 2

    if (force or not manifest_path.is_file()) and sample_ok:
        save_embeddings_manifest(cfg, n_samples=n_samples, n_classes=n_classes)
        result["embeddings_backfilled"] = True
        result["n_samples"] = n_samples

    ckpt = CHECKPOINTS_DIR / cfg.checkpoint_name
    if ckpt.is_file() and (force or not checkpoint_manifest_path(cfg).is_file()) and embeddings_cache_valid(cfg):
        save_train_manifest(cfg)
        result["train_backfilled"] = True

    return result


def apply_cache_skips(cfg: Any) -> tuple[Any, dict[str, Any]]:
    if cfg.skip_if_cached:
        backfill_cache_manifests(cfg)

    status = cache_status(cfg)
    if not cfg.skip_if_cached:
        return cfg, status

    if status["embeddings_valid"]:
        cfg.skip_embeddings_if_exists = True
    else:
        cfg.skip_embeddings_if_exists = False

    if status["train_valid"]:
        cfg.skip_train_if_checkpoint_exists = True
    else:
        cfg.skip_train_if_checkpoint_exists = False

    return cfg, status


def list_checkpoint_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CHECKPOINTS_DIR.glob("*.pipeline.json")):
        data = load_json(path) or {}
        rows.append(
            {
                "checkpoint": data.get("checkpoint", path.stem.replace(".pipeline", "") + ".pt"),
                "fingerprint": data.get("fingerprint"),
                "epochs": (data.get("config") or {}).get("train_epochs"),
                "embeddings_fingerprint": data.get("embeddings_fingerprint"),
                "created_at": data.get("created_at"),
                "manifest": path.name,
            }
        )
    return rows

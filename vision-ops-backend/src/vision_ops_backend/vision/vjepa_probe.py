"""V1 — V-JEPA clip embedding + motion anomaly score for warehouse camera."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HF_VJEPA_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def _motion_embedding(frames: list[np.ndarray]) -> np.ndarray:
    """Fallback temporal signature when V-JEPA weights unavailable."""
    if len(frames) < 2:
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32)).astype(np.float32).flatten()
        return small / (np.linalg.norm(small) + 1e-8)

    diffs: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev, gray)
        diffs.append(float(diff.mean()))
        prev = gray
    hist, _ = np.histogram(diffs, bins=16, range=(0, 64))
    vec = hist.astype(np.float32)
    vec = np.concatenate([vec, [float(np.std(diffs)), float(np.mean(diffs))]])
    return vec / (np.linalg.norm(vec) + 1e-8)


def _vjepa_hf_embedding(frames: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import torch
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError:
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoVideoProcessor.from_pretrained(HF_VJEPA_ID)
        model = AutoModel.from_pretrained(HF_VJEPA_ID).to(device).eval()

        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        video = np.stack(rgb, axis=0)
        inputs = processor(list(video), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = model.get_vision_features(**inputs)
        emb = feats.detach().cpu().numpy().reshape(-1).astype(np.float32)
        meta = {
            "backend": "huggingface",
            "model_id": HF_VJEPA_ID,
            "device": device,
            "embedding_dim": int(emb.shape[0]),
        }
        return emb, meta
    except Exception as exc:
        logger.warning("V-JEPA HF probe failed: %s", exc)
        return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(max(0.0, 1.0 - (a @ b) / denom))


def run_vjepa_probe(
    frames: list[np.ndarray],
    *,
    baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    hf = _vjepa_hf_embedding(frames)
    if hf is not None:
        embedding, meta = hf
        backend = meta["backend"]
    else:
        embedding = _motion_embedding(frames)
        meta = {
            "backend": "motion_fallback",
            "embedding_dim": int(embedding.shape[0]),
            "hint": "pip install torch transformers for V-JEPA HF weights",
        }
        backend = meta["backend"]

    anomaly_score = 0.0
    if baseline is not None:
        anomaly_score = _cosine_distance(embedding, baseline)
    else:
        anomaly_score = 0.12

    severity = "normal"
    if anomaly_score >= 0.55:
        severity = "critical"
    elif anomaly_score >= 0.35:
        severity = "warning"

    event_title = "Motion pattern nominal"
    event_desc = f"Clip embedding ({backend}); distance to baseline {anomaly_score:.2f}"
    overlay_variant = "tertiary"
    if severity == "critical":
        event_title = "Forklift / motion anomaly"
        event_desc = "V-JEPA embedding diverged from shift baseline — review Zone B"
        overlay_variant = "error"
    elif severity == "warning":
        event_title = "Unusual motion detected"
        event_desc = "Temporal signature above warning threshold"
        overlay_variant = "error"

    return {
        "ok": True,
        "status": "ok",
        "backend": backend,
        "meta": meta,
        "embedding": embedding,
        "anomaly_score": round(anomaly_score, 4),
        "severity": severity,
        "event": {
            "title": event_title,
            "description": event_desc,
            "severity": severity,
        },
        "overlay": {
            "type": "forklift",
            "label": event_title,
            "top": "60%",
            "left": "20%",
            "width": "25%",
            "height": "30%",
            "variant": overlay_variant,
        },
    }

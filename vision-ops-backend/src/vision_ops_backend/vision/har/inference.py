"""Run HAR classification for a single model (har-research)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vision_ops_backend.vision.har.checkpoints import get_registry
from vision_ops_backend.vision.har.constants import spec_for_model
from vision_ops_backend.vision.har.device import har_device, torch_available

logger = logging.getLogger(__name__)


def _class_name(class_names: list[str], idx: int) -> str:
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"


def apply_class_exclusions(probs, class_names: list[str], exclude_labels: list[str] | None):
    """Zero excluded classes and re-normalize softmax (no retraining required)."""
    import torch

    if not exclude_labels:
        return probs
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs, dtype=torch.float32)
    exclude = {label.strip().casefold() for label in exclude_labels if label.strip()}
    if not exclude:
        return probs
    masked = probs.clone()
    for idx, name in enumerate(class_names):
        if name.strip().casefold() in exclude:
            masked[idx] = 0.0
    total = float(masked.sum().item())
    if total <= 0:
        return probs
    return masked / total


def _format_prediction(probs, class_names: list[str], top_k: int = 3) -> dict[str, Any]:
    import torch

    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs)
    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k)
    top = [
        {"label": _class_name(class_names, int(i)), "prob": round(float(v), 4)}
        for v, i in zip(values.tolist(), indices.tolist())
    ]
    pred_idx = int(probs.argmax().item())
    conf = float(probs.max().item())
    return {
        "label": _class_name(class_names, pred_idx),
        "class_index": pred_idx,
        "confidence": round(conf, 4),
        "top_k": top,
    }


def run_har_inference(
    model_id: str,
    frames_bgr: list[np.ndarray],
    *,
    top_k: int = 3,
    exclude_labels: list[str] | None = None,
    return_embedding: bool = False,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Classify activity for one v2 model on a frame list."""
    if not torch_available():
        raise RuntimeError("torch not installed — run: uv sync --extra har")

    spec = spec_for_model(model_id)
    if spec is None:
        raise ValueError(f"unknown model_id: {model_id}")

    registry = get_registry()
    registry.ensure_loaded()
    if not registry._ready.get(spec.ckpt_key):
        raise RuntimeError(registry._reasons.get(spec.ckpt_key, "model not ready"))

    predictor = registry.predictors.get(spec.ckpt_key)
    if predictor is None:
        raise RuntimeError(registry._reasons.get(spec.ckpt_key, "model not loaded"))

    raw = predictor.predict_from_crops(frames_bgr, top_k=max(1, min(10, top_k)))
    class_names = list(predictor.class_names)

    import torch

    probs = torch.tensor([raw["all_probs"].get(name, 0.0) for name in class_names], dtype=torch.float32)
    if exclude_labels:
        probs = apply_class_exclusions(probs, class_names, exclude_labels)

    prediction = _format_prediction(probs, class_names, top_k=max(1, min(10, top_k)))
    conf = float(prediction["confidence"])
    gate = min_confidence if min_confidence is not None else predictor.min_confidence
    if gate is not None and conf < gate:
        prediction["raw_label"] = prediction["label"]
        prediction["raw_confidence"] = prediction["confidence"]
        prediction["label"] = None
        prediction["uncertain"] = True
    else:
        prediction["uncertain"] = False
        prediction["raw_label"] = prediction["label"]
        prediction["raw_confidence"] = prediction["confidence"]

    all_probs = {
        class_names[i]: round(float(probs[i].item()), 4) for i in range(len(class_names))
    }
    prediction["all_probs"] = all_probs

    backbone = str(getattr(predictor, "backbone", "vjepa"))
    result: dict[str, Any] = {
        "model_id": model_id,
        "backend": f"har-research:{backbone}",
        "device": har_device(),
        "prediction": prediction,
    }
    if return_embedding:
        emb = raw.get("embedding")
        if emb is not None:
            if hasattr(emb, "tolist"):
                result["embedding"] = emb.tolist()
            else:
                result["embedding"] = list(emb)
    if exclude_labels:
        result["excluded_labels"] = list(exclude_labels)
    return result

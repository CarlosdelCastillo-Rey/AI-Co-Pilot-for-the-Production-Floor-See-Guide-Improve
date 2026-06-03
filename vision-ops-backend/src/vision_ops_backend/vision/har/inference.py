"""Run HAR classification for a single model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vision_ops_backend.vision.har.checkpoints import (
    CKPT_KEY_DINOV2_MCJEPA,
    CKPT_KEY_DINOV2_PURO,
    CKPT_KEY_VJEPA2_MCJEPA_FROZEN,
    CKPT_KEY_VJEPA2_MCJEPA_PARTIAL,
    CKPT_KEY_VJEPA2_PURO,
    get_registry,
)
from vision_ops_backend.vision.har.constants import spec_for_model
from vision_ops_backend.vision.har.device import har_device, torch_available
from vision_ops_backend.vision.har.extractors import get_dino_extractor, get_vjepa_extractor
from vision_ops_backend.vision.har.frames import frames_to_rgb_batch

logger = logging.getLogger(__name__)


def _class_name(class_names: list[str], idx: int) -> str:
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"


def _format_prediction(probs, class_names: list[str], top_k: int = 3) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

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


def run_har_inference(model_id: str, frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    """Classify activity for one model on a frame list."""
    if not torch_available():
        raise RuntimeError("torch not installed — run: uv sync --extra har")

    spec = spec_for_model(model_id)
    if spec is None:
        raise ValueError(f"unknown model_id: {model_id}")

    registry = get_registry()
    registry.ensure_loaded()
    if not registry._ready.get(spec.ckpt_key):
        raise RuntimeError(registry._reasons.get(spec.ckpt_key, "model not ready"))

    import torch
    import torch.nn.functional as F

    rgb_frames = frames_to_rgb_batch(frames_bgr)
    batch = [rgb_frames]
    ckpt_key = spec.ckpt_key
    backend = "har_checkpoint"

    with torch.inference_mode():
        if ckpt_key == CKPT_KEY_DINOV2_PURO:
            dino = get_dino_extractor()
            seq = dino(batch)
            emb = seq.mean(dim=1)
            logits = registry.classifiers[ckpt_key](emb)
            backend = "dinov2_hf:facebook/dinov2-small"
        elif ckpt_key == CKPT_KEY_DINOV2_MCJEPA:
            dino = get_dino_extractor()
            seq = dino(batch)
            emb = registry.dmc_enc(seq)
            logits = registry.classifiers[ckpt_key](emb)
            backend = "dinov2_hf+mcjepa_encoder"
        elif ckpt_key == CKPT_KEY_VJEPA2_PURO:
            vj = get_vjepa_extractor()
            tokens = vj(batch)
            emb = tokens.mean(dim=1)
            logits = registry.classifiers[ckpt_key](emb)
            backend = "vjepa2_hf:facebook/vjepa2-vitl-fpc64-256"
        elif ckpt_key in (CKPT_KEY_VJEPA2_MCJEPA_FROZEN, CKPT_KEY_VJEPA2_MCJEPA_PARTIAL):
            vj = get_vjepa_extractor()
            tokens = vj(batch)
            out = registry.classifiers[ckpt_key](tokens)
            logits = out["logits"]
            backend = f"vjepa2_hf+mcjepa_head:{ckpt_key}"
        else:
            raise ValueError(f"unsupported ckpt_key: {ckpt_key}")

        probs = F.softmax(logits, dim=-1)[0].cpu()

    class_names = registry.class_names or [f"class_{i}" for i in range(probs.numel())]
    prediction = _format_prediction(probs, class_names)
    return {
        "model_id": model_id,
        "backend": backend,
        "device": har_device(),
        "prediction": prediction,
    }

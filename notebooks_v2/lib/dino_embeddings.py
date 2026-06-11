"""v2 DINOv2 embeddings — temporal attention pooling replacing mean-pool."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lib.constants import (
    DEFAULT_TEMPORAL_AGG,
    DINOV2_HF_MODEL_ID,
    DINOV2_HUB_MODEL,
    IMAGE_SIZE,
    NUM_FRAMES,
)
from lib.embeddings import get_device, sample_frames
from lib.paths import MODELS_DIR

logger = logging.getLogger(__name__)

_model: Any      = None
_processor: Any  = None
_backend: str    = ""
_attn_pool: Any  = None   # TemporalAttentionPool singleton (lazy init)


def _ensure_attn_pool(dim: int, device: str) -> Any:
    global _attn_pool
    if _attn_pool is None or getattr(_attn_pool, "_dim", None) != dim:
        from lib.har_model import TemporalAttentionPool
        _attn_pool = TemporalAttentionPool(dim).to(device)
        _attn_pool._dim = dim
        logger.info("TemporalAttentionPool initialized (dim=%d)", dim)
    return _attn_pool


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_dinov2_hub(*, source: str, repo: str):
    device = get_device()
    weights = os.environ.get("DINOV2_WEIGHTS_PATH", "").strip() or True
    kwargs: dict[str, Any] = {"pretrained": weights}
    if isinstance(weights, str) and weights:
        kwargs["pretrained"] = Path(weights).expanduser()
    model = torch.hub.load(repo, DINOV2_HUB_MODEL, source=source, **kwargs)
    return model.to(device).eval()


def _load_dinov2_hf():
    global _model, _processor, _backend
    from transformers import AutoImageProcessor, AutoModel
    device = get_device()
    _processor = AutoImageProcessor.from_pretrained(DINOV2_HF_MODEL_ID)
    _model     = AutoModel.from_pretrained(DINOV2_HF_MODEL_ID).to(device).eval()
    for p in _model.parameters():
        p.requires_grad = False
    _backend = "hf"
    return _model, _processor


def _load_dinov2():
    global _model, _processor, _backend
    if _model is not None:
        return _model, _processor
    local = MODELS_DIR / "dinov2-main"
    try:
        if local.is_dir():
            _model = _load_dinov2_hub(source="local",  repo=str(local))
        else:
            _model = _load_dinov2_hub(source="github", repo="facebookresearch/dinov2")
        for p in _model.parameters():
            p.requires_grad = False
        _processor = None
        _backend   = "hub"
    except Exception as exc:
        logger.warning("DINOv2 hub failed (%s); trying HuggingFace …", exc)
        return _load_dinov2_hf()
    return _model, _processor


# ── Frame preprocessing ───────────────────────────────────────────────────────

def _preprocess_hub(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    arr  = np.stack(frames_rgb).astype(np.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2)
    device = get_device()
    return (batch.to(device) - mean.to(device)) / std.to(device)


def _cls_from_hub_output(feats):
    if isinstance(feats, dict):  return feats["x_norm_clstoken"]
    if isinstance(feats, (tuple, list)): return feats[0]
    return feats


# ── Main extraction function ──────────────────────────────────────────────────

def extract_dinov2_embedding(
    frames_bgr: list[np.ndarray],
    *,
    temporal_agg: str = DEFAULT_TEMPORAL_AGG,
) -> np.ndarray:
    """Per-frame CLS tokens → temporal aggregation → 1024-dim embedding.

    temporal_agg:
        "mean"      — simple mean-pool (legacy, loses sequence order)
        "attention" — learnable temporal attention (recommended)
    """
    if len(frames_bgr) < 1:
        raise ValueError("need at least 1 frame")
    model, processor = _load_dinov2()
    frames_rgb = sample_frames(frames_bgr, n=NUM_FRAMES, size=IMAGE_SIZE)
    device     = get_device()

    with torch.inference_mode():
        if _backend == "hf":
            from PIL import Image
            pil = [Image.fromarray(f) for f in frames_rgb]
            inputs = processor(images=pil, return_tensors="pt").to(device)
            out    = model(**inputs)
            cls    = out.last_hidden_state[:, 0]     # (T, D)
        else:
            batch = _preprocess_hub(frames_rgb)
            feats = model.forward_features(batch) if hasattr(model, "forward_features") else model(batch)
            cls   = _cls_from_hub_output(feats)      # (T, D)
            if cls.ndim == 1:
                cls = cls.unsqueeze(0)

        if temporal_agg == "attention":
            pool  = _ensure_attn_pool(cls.shape[-1], device)
            pool.eval()
            emb   = pool(cls)                        # (D,)
        else:
            emb = cls.mean(dim=0)                    # (D,) — legacy

        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)


def extract_dinov2_per_frame(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Return per-frame CLS tokens (T, 1024) for GRU-based training."""
    if len(frames_bgr) < 1:
        raise ValueError("need at least 1 frame")
    model, processor = _load_dinov2()
    frames_rgb = sample_frames(frames_bgr, n=NUM_FRAMES, size=IMAGE_SIZE)
    device     = get_device()
    with torch.inference_mode():
        if _backend == "hf":
            from PIL import Image
            pil  = [Image.fromarray(f) for f in frames_rgb]
            inp  = processor(images=pil, return_tensors="pt").to(device)
            out  = model(**inp)
            cls  = out.last_hidden_state[:, 0]
        else:
            batch = _preprocess_hub(frames_rgb)
            feats = model.forward_features(batch) if hasattr(model, "forward_features") else model(batch)
            cls   = _cls_from_hub_output(feats)
            if cls.ndim == 1:
                cls = cls.unsqueeze(0)
    return cls.cpu().numpy().astype(np.float32)   # (T, D)


def embedding_dim() -> int:
    model, _ = _load_dinov2()
    return int(getattr(model, "embed_dim", getattr(getattr(model, "config", None), "hidden_size", 1024)))


# backward compat alias
extract_dinov3_embedding = extract_dinov2_embedding

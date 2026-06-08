"""Frozen DINOv2 frame embeddings (mean-pooled over clip) for HAR comparison."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lib.constants import DINOV2_HUB_MODEL, DINOV2_HF_MODEL_ID, IMAGE_SIZE, NUM_FRAMES
from lib.embeddings import get_device, sample_frames
from lib.paths import MODELS_DIR

logger = logging.getLogger(__name__)

_model: Any = None
_processor: Any = None
_backend: str = ""


def _ensure_dinov2_on_path() -> Path:
    root = MODELS_DIR / "dinov2-main"
    if not root.is_dir():
        raise FileNotFoundError(
            f"DINOv2 source not found at {root}. "
            "Use torch.hub (default) or clone https://github.com/facebookresearch/dinov2 into models/dinov2-main."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _load_dinov2_hub(*, source: str, repo: str):
    import torch

    device = get_device()
    logger.info("Loading DINOv2 (%s) %s on %s …", source, DINOV2_HUB_MODEL, device)
    weights = os.environ.get("DINOV2_WEIGHTS_PATH", "").strip() or True
    kwargs: dict[str, Any] = {"pretrained": weights}
    if isinstance(weights, str) and weights:
        kwargs["pretrained"] = Path(weights).expanduser()
    model = torch.hub.load(repo, DINOV2_HUB_MODEL, source=source, **kwargs)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_dinov2_hf():
    global _model, _processor, _backend
    from transformers import AutoImageProcessor, AutoModel

    device = get_device()
    logger.info("Loading DINOv2 (HF) %s on %s …", DINOV2_HF_MODEL_ID, device)
    _processor = AutoImageProcessor.from_pretrained(DINOV2_HF_MODEL_ID)
    _model = AutoModel.from_pretrained(DINOV2_HF_MODEL_ID).to(device).eval()
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
            _model = _load_dinov2_hub(source="local", repo=str(local))
        else:
            _model = _load_dinov2_hub(source="github", repo="facebookresearch/dinov2")
        _processor = None
        _backend = "hub"
        return _model, _processor
    except Exception as exc:
        logger.warning("DINOv2 hub load failed (%s); trying HuggingFace …", exc)
        return _load_dinov2_hf()


def _frames_to_pil(frames_rgb: list[np.ndarray]):
    from PIL import Image

    return [Image.fromarray(f) for f in frames_rgb]


def _preprocess_frames_hub(frames_rgb: list[np.ndarray]):
    import torch

    arr = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2)
    device = get_device()
    return ((batch.to(device) - mean.to(device)) / std.to(device))


def _cls_from_hub_output(feats: Any):
    if isinstance(feats, dict):
        return feats["x_norm_clstoken"]
    if isinstance(feats, (tuple, list)):
        return feats[0]
    return feats


def extract_dinov2_embedding(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Temporal mean of per-frame CLS tokens (DINOv2 ViT-L/14 on sampled crops)."""
    import torch
    import torch.nn.functional as F

    if len(frames_bgr) < 1:
        raise ValueError("need at least 1 frame")
    model, processor = _load_dinov2()
    frames_rgb = sample_frames(frames_bgr, n=NUM_FRAMES, size=IMAGE_SIZE)
    device = get_device()

    with torch.inference_mode():
        if _backend == "hf":
            inputs = processor(images=_frames_to_pil(frames_rgb), return_tensors="pt").to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0]
        else:
            batch = _preprocess_frames_hub(frames_rgb)
            if hasattr(model, "forward_features"):
                feats = model.forward_features(batch)
            else:
                feats = model(batch)
            cls = _cls_from_hub_output(feats)
            if cls.ndim == 1:
                cls = cls.unsqueeze(0)
        emb = cls.mean(dim=0)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)


# Backward-compatible alias (old notebooks referenced dinov3)
extract_dinov3_embedding = extract_dinov2_embedding


def embedding_dim() -> int:
    model, _ = _load_dinov2()
    return int(getattr(model, "embed_dim", getattr(getattr(model, "config", None), "hidden_size", 1024)))

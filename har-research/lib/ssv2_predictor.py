"""SSv2HarPredictor — drop-in for HarPredictor using facebook/vjepa2-vitl-fpc16-256-ssv2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lib.constants import DEFAULT_MIN_CONFIDENCE

logger = logging.getLogger(__name__)


class _SSv2MLP(torch.nn.Module):
    """LayerNorm MLP matching the architecture saved by the SSv2 training script."""

    def __init__(self, din: int, nc: int, hidden: int = 512, dropout: float = 0.30):
        super().__init__()
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, nc),
        )

    def forward(self, x):
        return self.net(x)


VJEPA_SSV2_MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
SSV2_NUM_FRAMES     = 16
SSV2_IMAGE_SIZE     = 256

_ssv2_model: Any = None


def _load_ssv2_backbone():
    global _ssv2_model
    if _ssv2_model is not None:
        return _ssv2_model
    from transformers import AutoModel

    device = _get_device()
    logger.info("Loading V-JEPA2 SSv2 %s on %s …", VJEPA_SSV2_MODEL_ID, device)
    model = AutoModel.from_pretrained(VJEPA_SSV2_MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _ssv2_model = model
    return _ssv2_model


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sample_frames(frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
    import cv2

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    n = SSV2_NUM_FRAMES
    sz = SSV2_IMAGE_SIZE
    if len(rgb) >= n:
        idx = np.linspace(0, len(rgb) - 1, n).astype(int)
        selected = [rgb[i] for i in idx]
    else:
        selected = list(rgb) + [rgb[-1]] * (n - len(rgb))
    import cv2 as _cv2
    return [_cv2.resize(f, (sz, sz), interpolation=_cv2.INTER_LINEAR) for f in selected]


def extract_ssv2_embedding(frames_bgr: list[np.ndarray]) -> np.ndarray:
    """Extract fused-mean embedding using facebook/vjepa2-vitl-fpc16-256-ssv2."""
    device = _get_device()
    model = _load_ssv2_backbone()
    frames_rgb = _sample_frames(frames_bgr)

    arr = np.stack(frames_rgb).astype(np.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    video = (video - mean.to(device)) / std.to(device)

    with torch.inference_mode():
        try:
            out = model(video, skip_predictor=True)
        except TypeError:
            out = model(video)
        if hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif isinstance(out, dict):
            tokens = out.get("last_hidden_state") or list(out.values())[0]
        else:
            tokens = out[0] if isinstance(out, (tuple, list)) else out
        if tokens.ndim == 4:
            b, t, n, d = tokens.shape
            tokens = tokens.reshape(b, t * n, d)
        emb = tokens.mean(dim=1).squeeze(0)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)


class SSv2HarPredictor:
    """Predictor for the V-JEPA2 SSv2 fpc16-256 checkpoint.

    Drop-in replacement for HarPredictor — same .predict_from_crops() interface.
    Uses facebook/vjepa2-vitl-fpc16-256-ssv2 at inference time instead of fpc64-256.
    """

    backbone = "vjepa-ssv2"

    def __init__(self, checkpoint: Path, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(checkpoint, map_location=device_str, weights_only=False)
        self.class_names: list[str] = list(payload["class_names"])
        self.exclude_labels: list[str] = list(payload.get("exclude_labels", []))
        self.device = device_str
        self.min_confidence = min_confidence

        emb_dim = int(payload["emb_dim"])
        num_classes = len(self.class_names)
        self.model = _SSv2MLP(emb_dim, num_classes).to(device_str)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict_from_crops(self, crops_bgr: list[np.ndarray], *, top_k: int = 5) -> dict[str, Any]:
        if len(crops_bgr) < 4:
            return {"label": None, "confidence": 0.0, "top_k": [], "all_probs": {}, "uncertain": True}

        emb = extract_ssv2_embedding(crops_bgr)
        x = torch.from_numpy(emb).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)[0].cpu()

        if self.exclude_labels:
            exclude = {lb.strip().casefold() for lb in self.exclude_labels}
            masked = probs.clone()
            for i, name in enumerate(self.class_names):
                if name.strip().casefold() in exclude:
                    masked[i] = 0.0
            total = float(masked.sum().item())
            if total > 0:
                probs = masked / total

        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        top = [
            {"label": self.class_names[int(i)], "prob": round(float(v), 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
        all_probs = {
            self.class_names[i]: round(float(probs[i].item()), 4)
            for i in range(len(self.class_names))
        }
        pred_idx = int(probs.argmax().item())
        conf = float(probs[pred_idx].item())
        uncertain = conf < self.min_confidence
        label = None if uncertain else self.class_names[pred_idx]

        return {
            "label": label,
            "confidence": round(conf, 4),
            "class_index": pred_idx if not uncertain else None,
            "top_k": top,
            "all_probs": all_probs,
            "embedding": emb,
            "crop_bgr": crops_bgr[-1] if crops_bgr else None,
            "uncertain": uncertain,
            "raw_label": self.class_names[pred_idx],
            "raw_confidence": round(conf, 4),
        }

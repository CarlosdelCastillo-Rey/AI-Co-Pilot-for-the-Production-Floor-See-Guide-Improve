"""Run HAR on person crop buffers with blocked-label masking + confidence gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lib.constants import DEFAULT_MIN_CONFIDENCE
from lib.embeddings import extract_vjepa_embedding, sample_frames
from lib.har_model import load_checkpoint


def apply_class_exclusions(probs: torch.Tensor, class_names: list[str], exclude_labels: list[str]) -> torch.Tensor:
    if not exclude_labels:
        return probs
    exclude = {l.strip().casefold() for l in exclude_labels}
    masked = probs.clone()
    for i, name in enumerate(class_names):
        if name.strip().casefold() in exclude:
            masked[i] = 0.0
    total = float(masked.sum().item())
    if total <= 0:
        return probs
    return masked / total


class HarPredictor:
    def __init__(self, checkpoint: Path, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.model, self.info = load_checkpoint(checkpoint)
        self.class_names: list[str] = self.info["class_names"]
        self.exclude_labels: list[str] = list(self.info.get("exclude_labels", []))
        self.device = self.info["device"]
        self.min_confidence = min_confidence

    def predict_from_crops(self, crops_bgr: list[np.ndarray], *, top_k: int = 5) -> dict[str, Any]:
        if len(crops_bgr) < 4:
            return {"label": None, "confidence": 0.0, "top_k": [], "all_probs": {}, "uncertain": True}

        emb = extract_vjepa_embedding(crops_bgr)
        x = torch.from_numpy(emb).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)[0].cpu()
        probs = apply_class_exclusions(probs, self.class_names, self.exclude_labels)
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        top = [
            {"label": self.class_names[int(i)], "prob": round(float(v), 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
        all_probs = {
            self.class_names[i]: round(float(probs[i].item()), 4) for i in range(len(self.class_names))
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

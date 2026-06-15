"""V-JEPA2 PowerMean-7 ensemble predictor.

Architecture: 7 MLP classifiers trained on V-JEPA2 T=8 fused-mean embeddings,
combined with Power Mean q=0.5 for final class probabilities.

Best result on InHARD 12-class:
  Accuracy        = 0.878141
  Balanced Acc    = 0.848224
  F1 Macro        = 0.860986
  F1 Weighted     = 0.874309
  Mean AUC        = 0.9573
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.constants import (
    DEFAULT_MIN_CONFIDENCE,
    POWERMEAN7_CLASSES,
    POWERMEAN7_MODEL_CONFIGS,
    POWERMEAN7_NUM_FRAMES,
    POWERMEAN7_Q,
    POWERMEAN7_SUBDIR,
)
from lib.embeddings import extract_vjepa_embedding


class SimpleMLPClassifier(nn.Module):
    """Single MLP head used in the PowerMean-7 ensemble."""

    def __init__(self, encoder_dim: int = 1024, num_classes: int = 12,
                 hidden_dim: int = 512, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PowerMeanEnsemblePredictor:
    """Drop-in replacement for HarPredictor using the 7-MLP Power Mean ensemble.

    Accepts the same interface as HarPredictor.predict_from_crops():
        crops_bgr: list[np.ndarray]  — BGR frames (YOLO person crops or raw clips)

    Internally extracts a V-JEPA2 embedding with T=8 frames and runs all 7 MLPs,
    then combines probabilities with Power Mean q=0.5.
    """

    backbone = "vjepa-powermean7"

    def __init__(self, checkpoint_dir: Path, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.min_confidence = min_confidence
        self.class_names: list[str] = list(POWERMEAN7_CLASSES)
        self.exclude_labels: list[str] = []

        device_str = self._get_device()
        self.device = torch.device(device_str)
        self._models = self._load_models()

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_models(self) -> list[nn.Module]:
        models = []
        num_classes = len(self.class_names)
        for cfg in POWERMEAN7_MODEL_CONFIGS:
            pt_path = self.checkpoint_dir / cfg["filename"]
            if not pt_path.is_file():
                raise FileNotFoundError(f"PowerMean-7 checkpoint not found: {pt_path}")
            ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
            model = SimpleMLPClassifier(
                encoder_dim=1024,
                num_classes=num_classes,
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
        return models

    @torch.no_grad()
    def _power_mean(self, probs_stack: torch.Tensor) -> torch.Tensor:
        """Power Mean aggregation: normalize(mean(p^q)^(1/q)), q=0.5.

        Input:  [N, C]  — N ensemble members, C classes
        Output: [C]
        """
        q = POWERMEAN7_Q
        probs_power = torch.mean(probs_stack ** q, dim=0) ** (1.0 / q)  # [C]
        return probs_power / probs_power.sum().clamp(min=1e-8)

    # ── public interface (matches HarPredictor) ───────────────────────────────

    def predict_from_crops(self, crops_bgr: list[np.ndarray], *, top_k: int = 5) -> dict[str, Any]:
        if len(crops_bgr) < 1:
            return {
                "label": None, "confidence": 0.0, "top_k": [],
                "all_probs": {}, "uncertain": True,
                "raw_label": None, "raw_confidence": 0.0,
            }

        emb = extract_vjepa_embedding(crops_bgr, num_frames=POWERMEAN7_NUM_FRAMES)
        z = torch.from_numpy(emb).float()
        z = F.normalize(z, p=2, dim=0).unsqueeze(0).to(self.device)  # [1, 1024]

        probs_list: list[torch.Tensor] = []
        for model in self._models:
            logits = model(z)
            probs_list.append(F.softmax(logits, dim=1))

        probs_stack = torch.stack(probs_list, dim=0)   # [7, 1, C]
        probs_final = self._power_mean(probs_stack.squeeze(1))     # [12]

        probs_cpu = probs_final.cpu()
        k = min(top_k, probs_cpu.numel())
        values, indices = torch.topk(probs_cpu, k)
        top = [
            {"label": self.class_names[int(i)], "prob": round(float(v), 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
        all_probs = {
            self.class_names[i]: round(float(probs_cpu[i].item()), 4)
            for i in range(len(self.class_names))
        }
        pred_idx = int(probs_cpu.argmax().item())
        conf = float(probs_cpu[pred_idx].item())
        uncertain = conf < self.min_confidence
        raw_label = self.class_names[pred_idx]

        return {
            "label": None if uncertain else raw_label,
            "confidence": round(conf, 4),
            "class_index": pred_idx if not uncertain else None,
            "top_k": top,
            "all_probs": all_probs,
            "embedding": emb,
            "crop_bgr": crops_bgr[-1] if crops_bgr else None,
            "uncertain": uncertain,
            "raw_label": raw_label,
            "raw_confidence": round(conf, 4),
        }

    @classmethod
    def from_checkpoint_dir(cls, base_checkpoints_dir: Path, **kwargs) -> "PowerMeanEnsemblePredictor":
        """Construct from the parent checkpoints/ directory (adds powermean7/ subdir)."""
        return cls(base_checkpoints_dir / POWERMEAN7_SUBDIR, **kwargs)

    def is_ready(self) -> bool:
        return len(self._models) == len(POWERMEAN7_MODEL_CONFIGS)

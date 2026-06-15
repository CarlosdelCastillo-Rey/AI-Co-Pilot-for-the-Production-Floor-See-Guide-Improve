"""v2 HAR model heads: MLP, Bi-GRU, SupCon projector, temporal attention pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal attention pooling ────────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """Learnable per-frame weighting — replaces mean-pool.

    The MLP learns which frames in the clip are most discriminative
    (e.g. the grasp moment vs. idle frames).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D) or (B, T, D)
        w = torch.softmax(self.score(x), dim=-2)  # softmax over time
        return (w * x).sum(dim=-2)                # weighted sum → (D,) or (B, D)


# ── MLP head (v2: BatchNorm + residual-style) ─────────────────────────────────

class HarMLP(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Bi-GRU sequential head ────────────────────────────────────────────────────

class HarGRU(nn.Module):
    """Processes per-frame embedding sequences → classification.

    Input:  (B, T, emb_dim)  — T per-frame embeddings
    Output: (B, num_classes)

    Bidirectional GRU captures both "take" (rising) and "put down" (falling)
    action trajectories, which are mirror-symmetric under mean-pooling.
    """
    def __init__(self, emb_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)  # reduce dim before RNN
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = F.relu(self.proj(x))          # (B, T, hidden)
        _, h = self.gru(x)                 # h: (2*layers, B, hidden)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # last fwd + bwd hidden
        return self.head(h)


# ── SupCon projector ──────────────────────────────────────────────────────────

class SupConProjector(nn.Module):
    """2-layer MLP projector for Supervised Contrastive Learning.

    Applied on top of frozen backbone embeddings before the SupCon loss.
    Discarded after pre-training; only the backbone repr is kept.
    """
    def __init__(self, emb_dim: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # L2-normalize for cosine similarity


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    class_names: list[str],
    exclude_labels: list[str],
    emb_dim: int,
    head_arch: str = "mlp",
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict":    model.state_dict(),
        "class_names":   class_names,
        "exclude_labels": exclude_labels,
        "emb_dim":       emb_dim,
        "num_classes":   len(class_names),
        "head_arch":     head_arch,
        "meta":          meta or {},
    }
    torch.save(payload, path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "class_names":    class_names,
                "exclude_labels": exclude_labels,
                "emb_dim":        emb_dim,
                "num_classes":    len(class_names),
                "head_arch":      head_arch,
                "classifier_version": (meta or {}).get("classifier_version"),
                "embedding_version":  (meta or {}).get("embedding_version"),
                "meta": meta or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path,
    device: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    emb_dim     = int(payload["emb_dim"])
    class_names = list(payload["class_names"])
    head_arch   = payload.get("head_arch", "mlp")
    n_classes   = len(class_names)

    if head_arch == "gru":
        model = HarGRU(emb_dim, n_classes)
    else:
        model = HarMLP(emb_dim, n_classes)

    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    info = {
        "class_names":    class_names,
        "exclude_labels": list(payload.get("exclude_labels", [])),
        "emb_dim":        emb_dim,
        "head_arch":      head_arch,
        "meta":           payload.get("meta", {}),
        "device":         device,
    }
    return model, info

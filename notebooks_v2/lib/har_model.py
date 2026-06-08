"""HAR MLP classifier head (trainable on frozen embeddings)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class HarMLP(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(
    path: Path,
    *,
    model: HarMLP,
    class_names: list[str],
    exclude_labels: list[str],
    emb_dim: int,
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "exclude_labels": exclude_labels,
        "emb_dim": emb_dim,
        "num_classes": len(class_names),
        "meta": meta or {},
    }
    torch.save(payload, path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "exclude_labels": exclude_labels,
                "emb_dim": emb_dim,
                "num_classes": len(class_names),
                "classifier_version": (meta or {}).get("classifier_version"),
                "embedding_version": (meta or {}).get("embedding_version"),
                "meta": meta or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(path: Path, device: str | None = None) -> tuple[HarMLP, dict[str, Any]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    emb_dim = int(payload["emb_dim"])
    class_names = list(payload["class_names"])
    model = HarMLP(emb_dim, len(class_names))
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    info = {
        "class_names": class_names,
        "exclude_labels": list(payload.get("exclude_labels", [])),
        "emb_dim": emb_dim,
        "meta": payload.get("meta", {}),
        "device": device,
    }
    return model, info

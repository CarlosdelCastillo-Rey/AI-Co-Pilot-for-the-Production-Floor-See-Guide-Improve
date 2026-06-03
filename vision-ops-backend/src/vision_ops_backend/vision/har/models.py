"""PyTorch architectures for Avance 4 HAR (aligned with interfaz_mcjepa_v2.py)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DinoToMCJEPAEncoder(nn.Module):
    """Multi-scale temporal encoder over DINOv2 frame embeddings."""

    def __init__(self, d_in: int, hidden: int = 256, nhead: int = 4, layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.short_encoder = self._mk(hidden, nhead, layers)
        self.mid_encoder = self._mk(hidden, nhead, layers)
        self.global_encoder = self._mk(hidden, nhead, layers)
        self.norm = nn.LayerNorm(hidden * 3)

    def _mk(self, h: int, n: int, l: int) -> nn.TransformerEncoder:
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h,
                nhead=n,
                dim_feedforward=h * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=l,
        )

    def forward(self, dino_seq: torch.Tensor) -> torch.Tensor:
        x = self.proj(dino_seq)
        t = x.shape[1]
        sl, ml = max(1, t // 4), max(1, t // 2)
        zs = self.short_encoder(x[:, -sl:]).mean(1)
        zm = self.mid_encoder(x[:, -ml:]).mean(1)
        zg = self.global_encoder(x).mean(1)
        return self.norm(torch.cat([zs, zm, zg], dim=-1))


class EmbeddingMLPClassifier(nn.Module):
    """Light MLP classifier on frozen embeddings."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MCJEPAHead(nn.Module):
    """MC-JEPA head on V-JEPA2 tokens."""

    def __init__(
        self,
        d_in: int,
        hidden: int = 256,
        nhead: int = 4,
        layers: int = 1,
        num_classes: int = 14,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.short_encoder = self._mk(hidden, nhead, layers)
        self.mid_encoder = self._mk(hidden, nhead, layers)
        self.global_encoder = self._mk(hidden, nhead, layers)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def _mk(self, h: int, n: int, l: int) -> nn.TransformerEncoder:
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h,
                nhead=n,
                dim_feedforward=h * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=l,
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.proj(tokens)
        n = x.shape[1]
        sl, ml = max(1, n // 4), max(1, n // 2)
        zs = self.short_encoder(x[:, -sl:]).mean(1)
        zm = self.mid_encoder(x[:, -ml:]).mean(1)
        zg = self.global_encoder(x).mean(1)
        z_ctx = self.fusion(torch.cat([zs, zm], dim=-1))
        return {
            "logits": self.classifier(z_ctx),
            "z_context": z_ctx,
            "z_pred_global": self.predictor(z_ctx),
            "z_global": zg,
        }

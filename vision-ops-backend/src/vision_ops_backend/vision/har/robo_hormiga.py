"""Robo Hormiga petty-theft detector — V-JEPA2 embeddings + 4 heuristic features + MLP-7."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Checkpoint relative to repo root (har-research/checkpoints/)
_DEFAULT_CKPT = (
    Path(__file__).resolve().parents[5] / "har-research" / "checkpoints" / "robo_hormiga_mlp7.pt"
)
INPUT_DIM = 1028  # 1024 (V-JEPA2 embedding) + 4 heuristic features


# ── Heuristic feature extractors (from notebooks/robo_hormiga.ipynb cell 11) ──

def compute_aspect_ratio(frames_bgr: list[np.ndarray]) -> float:
    """Fraction of horizontally-active columns — proxy for silhouette width."""
    widths: list[float] = []
    for f in frames_bgr[::4]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        col_var = gray.var(axis=0)
        active = int((col_var > col_var.mean()).sum())
        widths.append(active / max(1, f.shape[1]))
    return float(np.mean(widths)) if widths else 0.0


def compute_bg_diff(frames_bgr: list[np.ndarray]) -> float:
    """Mean absolute pixel change first↔last frame — object disappearance proxy."""
    if len(frames_bgr) < 2:
        return 0.0
    first = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    last = cv2.cvtColor(frames_bgr[-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(first - last).mean()) / 255.0


def compute_motion_asymmetry(frames_bgr: list[np.ndarray]) -> float:
    """Optical-flow asymmetry: extend-then-fold arm motion signals theft."""
    if len(frames_bgr) < 4:
        return 0.0
    flows_x: list[float] = []
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for f in frames_bgr[1:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev.astype(np.float32), curr.astype(np.float32),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        flows_x.append(float(flow[..., 0].mean()))
        prev = curr
    n = len(flows_x)
    h1 = float(np.mean(flows_x[: n // 2])) if n // 2 > 0 else 0.0
    h2 = float(np.mean(flows_x[n // 2 :])) if n - n // 2 > 0 else 0.0
    return abs(h1 - h2)


def compute_delta_emb(emb: np.ndarray, emb_prev: np.ndarray | None) -> float:
    """Cosine distance between current and previous clip embedding."""
    if emb_prev is None:
        return 0.0
    n1, n2 = np.linalg.norm(emb), np.linalg.norm(emb_prev)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(1.0 - np.dot(emb / n1, emb_prev / n2))


# ── MLP-7 architecture (matches notebooks/robo_hormiga.ipynb cell 14) ─────────

def _build_mlp7(input_dim: int = INPUT_DIM) -> Any:
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.30),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.20),
        nn.Linear(64, 32), nn.GELU(),
        nn.Linear(32, 16), nn.GELU(),
        nn.Linear(16, 2),
    )


# ── Detector ──────────────────────────────────────────────────────────────────

class RoboHormigaDetector:
    """
    Per-track petty-theft detector with majority-vote temporal smoothing.

    Modes:
    - mlp7      : V-JEPA2 embedding + 4 heuristic features → MLP-7 → P(theft)
    - heuristic : 4 features only with fixed weights (no checkpoint needed)
    """

    # Heuristic fallback weights (ar, bg_diff, motion_asym, delta_emb)
    _H_WEIGHTS = (0.25, 0.35, 0.25, 0.15)
    # Per-feature normalization denominators for heuristic mode
    _H_NORMS = (0.8, 0.15, 2.0, 0.5)

    def __init__(
        self,
        ckpt_path: Path | None = None,
        threshold: float = 0.45,
        k_vote: int = 3,
    ) -> None:
        self.threshold = threshold
        self.k_vote = k_vote
        self._model: Any = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self.mode = "heuristic"

        self._prob_history: dict[int, deque[float]] = {}
        self._prev_emb: dict[int, np.ndarray] = {}

        self._try_load(ckpt_path or _DEFAULT_CKPT)

    def _try_load(self, path: Path) -> None:
        if not path.is_file():
            logger.info("RoboHormiga: no checkpoint at %s — heuristic mode", path)
            return
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = _build_mlp7(int(ckpt.get("input_dim", INPUT_DIM)))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model = model
            if "scaler_mean" in ckpt:
                self._scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
                self._scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
            if "threshold" in ckpt:
                self.threshold = float(ckpt["threshold"])
            if "k_vote" in ckpt:
                self.k_vote = int(ckpt["k_vote"])
            self.mode = "mlp7"
            logger.info(
                "RoboHormiga: MLP-7 loaded from %s (thr=%.2f, k=%d)",
                path, self.threshold, self.k_vote,
            )
        except Exception as exc:
            logger.warning("RoboHormiga: checkpoint load failed (%s) — heuristic mode", exc)

    def reset(self, track_id: int) -> None:
        self._prob_history.pop(track_id, None)
        self._prev_emb.pop(track_id, None)

    def reset_all(self) -> None:
        self._prob_history.clear()
        self._prev_emb.clear()

    def update(
        self,
        track_id: int,
        crops_bgr: list[np.ndarray],
        embedding: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run theft detection for one track.
        Returns: {"theft_prob": float, "theft_alert": bool, "mode": str}
        """
        if not crops_bgr:
            return {"theft_prob": 0.0, "theft_alert": False, "mode": self.mode}

        ar = compute_aspect_ratio(crops_bgr)
        bg = compute_bg_diff(crops_bgr)
        ma = compute_motion_asymmetry(crops_bgr)
        de = compute_delta_emb(embedding, self._prev_emb.get(track_id)) if embedding is not None else 0.0

        if embedding is not None:
            self._prev_emb[track_id] = embedding.copy()

        if self._model is not None and embedding is not None:
            theft_prob = self._mlp7_prob(embedding, ar, bg, ma, de)
        else:
            theft_prob = self._heuristic_prob(ar, bg, ma, de)

        if track_id not in self._prob_history:
            self._prob_history[track_id] = deque(maxlen=self.k_vote)
        self._prob_history[track_id].append(theft_prob)
        window = list(self._prob_history[track_id])
        alert = (sum(1 for p in window if p >= self.threshold) / len(window)) >= 0.5

        return {"theft_prob": round(float(theft_prob), 4), "theft_alert": bool(alert), "mode": self.mode}

    def _mlp7_prob(self, emb: np.ndarray, ar: float, bg: float, ma: float, de: float) -> float:
        import torch
        import torch.nn.functional as F

        extra = np.array([ar, bg, ma, de], dtype=np.float32)
        if self._scaler_mean is not None and self._scaler_scale is not None:
            extra = (extra - self._scaler_mean) / np.maximum(self._scaler_scale, 1e-8)

        x = torch.from_numpy(np.concatenate([emb.astype(np.float32), extra])).unsqueeze(0)
        with torch.no_grad():
            return float(F.softmax(self._model(x), dim=1)[0, 1].item())

    def _heuristic_prob(self, ar: float, bg: float, ma: float, de: float) -> float:
        """Weighted linear combination of normalized heuristic features."""
        vals = (ar, bg, ma, de)
        scores = [min(1.0, v / max(n, 1e-8)) for v, n in zip(vals, self._H_NORMS)]
        return float(sum(w * s for w, s in zip(self._H_WEIGHTS, scores)))

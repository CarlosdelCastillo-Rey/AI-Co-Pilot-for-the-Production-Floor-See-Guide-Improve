"""Frozen V-JEPA2 embeddings via HuggingFace + OpenCV preprocessing."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from lib.constants import IMAGE_SIZE, NUM_FRAMES, VJEPA_MODEL_ID

logger = logging.getLogger(__name__)

_model: Any = None
_processor_failed = False


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Conv3D in V-JEPA often fails on MPS — prefer CPU for stability
        return "cpu"
    return "cpu"


def _load_vjepa():
    global _model, _processor_failed
    if _model is not None:
        return _model
    import torch
    from transformers import AutoModel

    device = get_device()
    logger.info("Loading V-JEPA2 %s on %s …", VJEPA_MODEL_ID, device)
    model = AutoModel.from_pretrained(VJEPA_MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    _model = model
    return _model


def sample_frames(frames_bgr: list[np.ndarray], n: int = NUM_FRAMES, size: int = IMAGE_SIZE) -> list[np.ndarray]:
    if not frames_bgr:
        return []
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    if len(rgb) >= n:
        idx = np.linspace(0, len(rgb) - 1, n).astype(int)
        selected = [rgb[i] for i in idx]
    else:
        selected = list(rgb) + [rgb[-1]] * (n - len(rgb))
    return [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in selected]


def extract_clip_from_file(path, max_frames: int = 64) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _preprocess_video_tensor(frames_rgb: list[np.ndarray]):
    import torch

    # (T, H, W, C) -> (1, T, C, H, W) — matches HF VJEPA2Model + vision-ops-backend
    arr = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
    device = get_device()
    return ((video.to(device) - mean.to(device)) / std.to(device))


def extract_vjepa_embedding(frames_bgr: list[np.ndarray], num_frames: int = NUM_FRAMES) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if len(frames_bgr) < 1:
        raise ValueError("need at least 1 frame")
    model = _load_vjepa()
    frames_rgb = sample_frames(frames_bgr, n=num_frames)
    batch = _preprocess_video_tensor(frames_rgb)
    with torch.inference_mode():
        try:
            out = model(batch, skip_predictor=True)
        except TypeError:
            out = model(batch)
        if hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif isinstance(out, dict) and "last_hidden_state" in out:
            tokens = out["last_hidden_state"]
        else:
            tokens = out[0] if isinstance(out, (tuple, list)) else out
        emb = tokens.mean(dim=1).squeeze(0)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)

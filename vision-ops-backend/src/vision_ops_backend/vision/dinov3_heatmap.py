"""D2 — spatial attention / patch-similarity heatmap for assembly camera."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HF_DINOV2_ID = "facebook/dinov2-base"


def _patch_cosine_heatmap(frame: np.ndarray, grid: int = 16) -> tuple[np.ndarray, np.ndarray, str]:
    """DINO-style demo: cosine similarity of patches to center anchor (no weights)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (grid * 8, grid * 8))
    sh, sw = small.shape[:2]
    cell_h, cell_w = sh // grid, sw // grid
    feats: list[np.ndarray] = []
    for gy in range(grid):
        for gx in range(grid):
            patch = small[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            feats.append(gray.flatten())
    feats_arr = np.stack(feats, axis=0)
    feats_arr -= feats_arr.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(feats_arr, axis=1, keepdims=True) + 1e-8
    feats_norm = feats_arr / norms
    cy, cx = grid // 2, grid // 2
    anchor_idx = cy * grid + cx
    anchor = feats_norm[anchor_idx]
    sims = feats_norm @ anchor
    sim_grid = sims.reshape(grid, grid)
    sim_up = cv2.resize(sim_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    sim_up = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
    heat_u8 = (sim_up * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame, 0.45, heat_color, 0.55, 0)
    return heat_color, overlay, "patch_similarity_demo"


def _dinov2_hf_heatmap(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(HF_DINOV2_ID)
        model = AutoModel.from_pretrained(HF_DINOV2_ID).to(device).eval()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)

        tokens = out.last_hidden_state[0, 1:, :].cpu().numpy()
        n = tokens.shape[0]
        side = int(np.sqrt(n))
        if side * side != n:
            return None
        tokens = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
        anchor = tokens[n // 2 + side // 2]
        sims = tokens @ anchor
        sim_grid = sims.reshape(side, side)
        h, w = frame.shape[:2]
        sim_up = cv2.resize(sim_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        sim_up = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
        heat_u8 = (sim_up * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(frame, 0.45, heat_color, 0.55, 0)
        return heat_color, overlay, f"dinov2_hf:{HF_DINOV2_ID}"
    except Exception as exc:
        logger.warning("DINOv2 HF heatmap failed: %s", exc)
        return None


def _hot_region_overlay(frame: np.ndarray, heat: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {
            "type": "machine",
            "label": "DINO heatmap — no hotspot",
            "top": "25%",
            "left": "25%",
            "width": "50%",
            "height": "50%",
            "variant": "tertiary",
        }
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    fh, fw = frame.shape[:2]
    return {
        "type": "machine",
        "label": "Assembly focus (DINO heatmap)",
        "top": f"{y / fh * 100:.1f}%",
        "left": f"{x / fw * 100:.1f}%",
        "width": f"{bw / fw * 100:.1f}%",
        "height": f"{bh / fh * 100:.1f}%",
        "variant": "primary",
    }


def generate_heatmap(frame: np.ndarray) -> dict[str, Any]:
    hf = _dinov2_hf_heatmap(frame)
    if hf is not None:
        heat, overlay, backend = hf
    else:
        heat, overlay, backend = _patch_cosine_heatmap(frame)

    return {
        "ok": True,
        "backend": backend,
        "heatmap_bgr": heat,
        "overlay_bgr": overlay,
        "overlay": _hot_region_overlay(frame, heat),
    }

"""Frame preprocessing for HAR models."""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 224
NUM_FRAMES = 16


def frames_to_rgb_batch(
    frames_bgr: list[np.ndarray],
    num_frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> list[np.ndarray]:
    """Sample frames to fixed count and resize to RGB uint8 arrays."""
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    if len(frames_rgb) >= num_frames:
        idx = np.linspace(0, len(frames_rgb) - 1, num_frames).astype(int)
        selected = [frames_rgb[i] for i in idx]
    else:
        selected = list(frames_rgb) + [frames_rgb[-1]] * (num_frames - len(frames_rgb))
    return [cv2.resize(f, (image_size, image_size)) for f in selected]

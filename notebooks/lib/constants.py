"""InHARD labels and pipeline defaults (notebook-only stack)."""

from __future__ import annotations

# Never train or predict these meta-actions
BLOCKED_ACTIONS: tuple[str, ...] = ("Assemble system", "No action")

# 12 industrial meta-actions used for training / live HAR
TRAINABLE_ACTIONS: tuple[str, ...] = (
    "Consult sheets",
    "Turn sheets",
    "Take screwdriver",
    "Put down screwdriver",
    "Picking in front",
    "Picking left",
    "Take component",
    "Put down component",
    "Take measuring rod",
    "Put down measuring rod",
    "Take subsystem",
    "Put down subsystem",
)

# Full InHARD label set (14 meta-actions)
ALL_META_ACTIONS: tuple[str, ...] = TRAINABLE_ACTIONS + BLOCKED_ACTIONS

INHARD_DOWNLOAD_URL = "https://zenodo.org/record/4003541"

VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 16
IMAGE_SIZE = 224
CROP_SIZE = 224
DEFAULT_BBOX_PADDING = 0.15

# Per-person live defaults
DEFAULT_INFER_EVERY = 16
DEFAULT_BUFFER_FRAMES = 32
DEFAULT_DWELL_WINDOWS = 2
DEFAULT_STALE_SEC = 2.0

"""InHARD labels and pipeline v2 defaults."""

from __future__ import annotations

BLOCKED_ACTIONS: tuple[str, ...] = ("Assemble system", "No action")

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

ALL_META_ACTIONS: tuple[str, ...] = TRAINABLE_ACTIONS + BLOCKED_ACTIONS

INHARD_DOWNLOAD_URL = "https://zenodo.org/record/4003541"

VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 16
IMAGE_SIZE = 224
CROP_SIZE = 224
DEFAULT_BBOX_PADDING = 0.15

DEFAULT_INFER_EVERY = 16
DEFAULT_BUFFER_FRAMES = 32
DEFAULT_DWELL_WINDOWS = 2
DEFAULT_STALE_SEC = 2.0

# v2 inference / training
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_EMBEDDING_MODE = "yolo_crop"  # "yolo_crop" | "full_clip"
DEFAULT_SPLIT_MODE = "subject"  # "subject" | "random"
DEFAULT_USE_CLASS_WEIGHTS = True
DEFAULT_INCLUDE_HUMAN_LABELS = True

HITL_ACTION_VERDICTS = ("yes", "no", "dont_know", "maybe")
HITL_PERSON_VERDICTS = ("yes", "no", "unknown")

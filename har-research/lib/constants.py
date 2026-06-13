"""v2 pipeline constants — all tuneable knobs in one place."""

from __future__ import annotations

# ── Dataset ──────────────────────────────────────────────────────────────────
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

# ── Backbones ─────────────────────────────────────────────────────────────────
VJEPA_MODEL_ID    = "facebook/vjepa2-vitl-fpc64-256"
DINOV2_HUB_MODEL  = "dinov2_vitl14"
DINOV2_HF_MODEL_ID = "facebook/dinov2-large"
BACKBONE_VJEPA  = "vjepa"
BACKBONE_DINOV2 = "dinov2"
SUPPORTED_BACKBONES: tuple[str, ...] = (BACKBONE_VJEPA, BACKBONE_DINOV2)

# ── Frame / clip ──────────────────────────────────────────────────────────────
NUM_FRAMES   = 16
IMAGE_SIZE   = 224
CROP_SIZE    = 224
DEFAULT_BBOX_PADDING = 0.15

# InHARD 3-view composite layout (1280×720)
#   top-left  (0:360, 0:640)   → overhead / top-down  ← most discriminative
#   top-right (0:360, 640:)    → side / eye-level
#   bot-right (360:, 640:)     → front / low-angle
INHARD_VIEW_TOPDOWN = "topdown"   # best for direction discrimination
INHARD_VIEW_SIDE    = "side"
INHARD_VIEW_FRONT   = "front"
INHARD_VIEW_FULL    = "full"      # original full mosaic (legacy)
DEFAULT_INHARD_VIEW = INHARD_VIEW_TOPDOWN

# ── Embedding ─────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODE = "yolo_crop"   # "yolo_crop" | "full_clip" | "view_crop"
DEFAULT_TEMPORAL_AGG   = "attention"   # "mean" | "attention"
TEMPORAL_PROJ_DIM      = 128           # SupCon projector output dim

# ── Data sampling ─────────────────────────────────────────────────────────────
DEFAULT_CLIPS_PER_CLASS = None          # None → use ALL available clips
DEFAULT_SPLIT_MODE      = "subject"
DEFAULT_TEST_SIZE       = 0.2
DEFAULT_SEED            = 42

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS         = 100
DEFAULT_BATCH_SIZE     = 64
DEFAULT_LR             = 1e-3
DEFAULT_WEIGHT_DECAY   = 1e-4
DEFAULT_DROPOUT        = 0.3
DEFAULT_FOCAL_GAMMA    = 2.0
DEFAULT_MIXUP_ALPHA    = 0.2
DEFAULT_MIXUP_N_AUG    = 2
DEFAULT_USE_CLASS_WEIGHTS  = True
DEFAULT_USE_WEIGHTED_SAMPLER = True
DEFAULT_USE_FOCAL_LOSS     = True
DEFAULT_USE_MIXUP          = True

# SupCon stage
DEFAULT_SUPCON_EPOCHS      = 50
DEFAULT_SUPCON_LR          = 5e-4
DEFAULT_SUPCON_TEMPERATURE = 0.07
DEFAULT_SUPCON_PROJ_DIM    = 128

# Head architecture: "mlp" | "gru"
DEFAULT_HEAD_ARCH = "mlp"

# ── Inference ─────────────────────────────────────────────────────────────────
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_INFER_EVERY    = 16
DEFAULT_BUFFER_FRAMES  = 32
DEFAULT_DWELL_WINDOWS  = 2
DEFAULT_STALE_SEC      = 2.0
DEFAULT_INCLUDE_HUMAN_LABELS = True

# ── HITL (Human-In-The-Loop) ──────────────────────────────────────────────────
HITL_ACTION_VERDICTS = ("yes", "no", "dont_know", "maybe")
HITL_PERSON_VERDICTS = ("yes", "no", "unknown")

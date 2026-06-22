"""HAR model and mock camera identifiers (har-research only)."""

from __future__ import annotations

from dataclasses import dataclass

HAR_MODEL_V2_VJEPA        = "v2-vjepa"
HAR_MODEL_V2_DINOV2       = "v2-dinov2"
HAR_MODEL_V2_POWERMEAN7   = "v2-vjepa-powermean7"
HAR_MODEL_V2_SSV2         = "v2-vjepa2-ssv2"

HAR_MODEL_IDS: tuple[str, ...] = (
    HAR_MODEL_V2_POWERMEAN7,  # best model — listed first
    HAR_MODEL_V2_SSV2,
    HAR_MODEL_V2_VJEPA,
    HAR_MODEL_V2_DINOV2,
)

HAR_BENCH_CAMERA_ID = "cam-har-bench"

MOCK_WALL_SLOT_COUNT = 4


def mock_slot_camera_id(slot: int) -> str:
    return f"cam-har-mock-{slot}"


def mock_slot_index(camera_id: str) -> int | None:
    if not camera_id.startswith("cam-har-mock-"):
        return None
    try:
        return int(camera_id.rsplit("-", 1)[-1])
    except ValueError:
        return None


def is_har_mock_slot(camera_id: str) -> bool:
    return mock_slot_index(camera_id) is not None


HAR_CAMERA_IDS: tuple[str, ...] = (
    "cam-har-01",
    "cam-har-02",
    "cam-har-03",
    "cam-har-04",
)

# Checkpoint file stems under har-research/checkpoints (without .pt)
CKPT_KEY_V2_VJEPA      = "har_vjepa_12c_topdown_allavail"
CKPT_KEY_V2_DINOV2     = "har_dinov2_12c_topdown_allavail"
CKPT_KEY_POWERMEAN7    = "powermean7"  # directory, not a single .pt
CKPT_KEY_V2_SSV2       = "har_vjepa2_ssv2_12c_fused_mean"


@dataclass(frozen=True)
class HarModelSpec:
    model_id: str
    camera_id: str
    ckpt_key: str
    label: str
    display_name: str


HAR_MODELS: tuple[HarModelSpec, ...] = (
    HarModelSpec(
        HAR_MODEL_V2_POWERMEAN7,
        "cam-har-03",
        CKPT_KEY_POWERMEAN7,
        "V-JEPA2 PowerMean-7 ★",
        "HAR — V-JEPA2 PowerMean-7 (Best 87.8%)",
    ),
    HarModelSpec(
        HAR_MODEL_V2_SSV2,
        "cam-har-04",
        CKPT_KEY_V2_SSV2,
        "V-JEPA2 SSv2 (82.7%)",
        "HAR — V-JEPA2 SSv2 fpc16-256",
    ),
    HarModelSpec(
        HAR_MODEL_V2_VJEPA,
        "cam-har-01",
        CKPT_KEY_V2_VJEPA,
        "V-JEPA v2",
        "HAR — V-JEPA v2",
    ),
    HarModelSpec(
        HAR_MODEL_V2_DINOV2,
        "cam-har-02",
        CKPT_KEY_V2_DINOV2,
        "DINOv2 v2",
        "HAR — DINOv2 v2",
    ),
)

_MODEL_BY_ID = {m.model_id: m for m in HAR_MODELS}
_CAMERA_BY_ID = {m.camera_id: m for m in HAR_MODELS}


def spec_for_model(model_id: str) -> HarModelSpec | None:
    return _MODEL_BY_ID.get(model_id)


def spec_for_camera(camera_id: str) -> HarModelSpec | None:
    return _CAMERA_BY_ID.get(camera_id)


def camera_id_for_model(model_id: str) -> str | None:
    spec = spec_for_model(model_id)
    return spec.camera_id if spec else None


def model_id_for_camera(camera_id: str) -> str | None:
    spec = spec_for_camera(camera_id)
    return spec.model_id if spec else None


def is_har_camera(camera_id: str) -> bool:
    return camera_id in _CAMERA_BY_ID or camera_id == HAR_BENCH_CAMERA_ID or is_har_mock_slot(camera_id)


def is_har_bench_camera(camera_id: str) -> bool:
    return camera_id == HAR_BENCH_CAMERA_ID


HAR_MODEL_COLORS: dict[str, str] = {
    HAR_MODEL_V2_POWERMEAN7: "#A5D6A7",  # green — best model
    HAR_MODEL_V2_SSV2:       "#CE93D8",  # purple — SSv2 backbone
    HAR_MODEL_V2_VJEPA:      "#FFB74D",
    HAR_MODEL_V2_DINOV2:     "#4FC3F7",
}

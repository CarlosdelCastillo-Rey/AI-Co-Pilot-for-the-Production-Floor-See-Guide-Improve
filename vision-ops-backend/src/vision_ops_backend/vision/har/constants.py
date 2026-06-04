"""HAR model and mock camera identifiers."""

from __future__ import annotations

from dataclasses import dataclass

HAR_MODEL_DINOV2_PURO = "dinov2-puro"
HAR_MODEL_DINOV2_MCJEPA = "dinov2-mcjepa"
HAR_MODEL_VJEPA2_PURO = "vjepa2-puro"
HAR_MODEL_VJEPA2_MCJEPA_FROZEN = "vjepa2-mcjepa-frozen"
HAR_MODEL_VJEPA2_MCJEPA_PARTIAL = "vjepa2-mcjepa-partial"

HAR_MODEL_IDS: tuple[str, ...] = (
    HAR_MODEL_DINOV2_PURO,
    HAR_MODEL_DINOV2_MCJEPA,
    HAR_MODEL_VJEPA2_PURO,
    HAR_MODEL_VJEPA2_MCJEPA_FROZEN,
    HAR_MODEL_VJEPA2_MCJEPA_PARTIAL,
)

HAR_BENCH_CAMERA_ID = "cam-har-bench"

HAR_CAMERA_IDS: tuple[str, ...] = (
    "cam-har-01",
    "cam-har-02",
    "cam-har-03",
    "cam-har-04",
    "cam-har-05",
)

# Internal checkpoint keys (interfaz / notebooks)
CKPT_KEY_DINOV2_PURO = "DINOv2_puro"
CKPT_KEY_DINOV2_MCJEPA = "DINOv2_to_MCJEPA"
CKPT_KEY_VJEPA2_PURO = "VJEPA2_puro"
CKPT_KEY_VJEPA2_MCJEPA_FROZEN = "VJEPA2_MCJEPA_frozen"
CKPT_KEY_VJEPA2_MCJEPA_PARTIAL = "VJEPA2_MCJEPA_partial_finetune"


@dataclass(frozen=True)
class HarModelSpec:
    model_id: str
    camera_id: str
    ckpt_key: str
    label: str
    display_name: str


HAR_MODELS: tuple[HarModelSpec, ...] = (
    HarModelSpec(
        HAR_MODEL_DINOV2_PURO,
        "cam-har-01",
        CKPT_KEY_DINOV2_PURO,
        "DINOv2 puro",
        "HAR — DINOv2 puro",
    ),
    HarModelSpec(
        HAR_MODEL_DINOV2_MCJEPA,
        "cam-har-02",
        CKPT_KEY_DINOV2_MCJEPA,
        "DINOv2 → MC-JEPA",
        "HAR — DINO→MC-JEPA",
    ),
    HarModelSpec(
        HAR_MODEL_VJEPA2_PURO,
        "cam-har-03",
        CKPT_KEY_VJEPA2_PURO,
        "V-JEPA2 puro",
        "HAR — V-JEPA2 puro",
    ),
    HarModelSpec(
        HAR_MODEL_VJEPA2_MCJEPA_FROZEN,
        "cam-har-04",
        CKPT_KEY_VJEPA2_MCJEPA_FROZEN,
        "V-JEPA2 MC frozen",
        "HAR — V-JEPA MC frozen",
    ),
    HarModelSpec(
        HAR_MODEL_VJEPA2_MCJEPA_PARTIAL,
        "cam-har-05",
        CKPT_KEY_VJEPA2_MCJEPA_PARTIAL,
        "V-JEPA2 MC partial",
        "HAR — V-JEPA MC finetune",
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
    return camera_id in _CAMERA_BY_ID or camera_id == HAR_BENCH_CAMERA_ID


def is_har_bench_camera(camera_id: str) -> bool:
    return camera_id == HAR_BENCH_CAMERA_ID


HAR_MODEL_COLORS: dict[str, str] = {
    HAR_MODEL_DINOV2_PURO: "#4FC3F7",
    HAR_MODEL_DINOV2_MCJEPA: "#81C784",
    HAR_MODEL_VJEPA2_PURO: "#FFB74D",
    HAR_MODEL_VJEPA2_MCJEPA_FROZEN: "#CE93D8",
    HAR_MODEL_VJEPA2_MCJEPA_PARTIAL: "#F06292",
}

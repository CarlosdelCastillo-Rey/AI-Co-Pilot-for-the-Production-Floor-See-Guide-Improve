"""Resolve ONNX model paths under vision-ops-backend/weights/."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def yunet_model_path() -> Path:
    return _backend_root() / "weights" / "face_detection_yunet" / "face_detection_yunet_2023mar.onnx"


def sface_model_path() -> Path:
    return _backend_root() / "weights" / "face_recognition_sface" / "face_recognition_sface_2021dec.onnx"


def faces_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "faces"


def owner_feature_path() -> Path:
    return faces_data_dir() / "owner.npz"


def enrollment_preview_path() -> Path:
    return faces_data_dir() / "enrollment_preview.jpg"


def models_ready() -> bool:
    return yunet_model_path().is_file() and sface_model_path().is_file()

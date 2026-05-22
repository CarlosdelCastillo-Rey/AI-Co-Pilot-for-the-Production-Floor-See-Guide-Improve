"""Resolve ONNX model paths under repo models/."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def yunet_model_path() -> Path:
    return repo_root() / "models" / "face_detection_yunet" / "face_detection_yunet_2023mar.onnx"


def sface_model_path() -> Path:
    return repo_root() / "models" / "face_recognition_sface" / "face_recognition_sface_2021dec.onnx"


def faces_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "faces"


def owner_feature_path() -> Path:
    return faces_data_dir() / "owner.npz"


def enrollment_preview_path() -> Path:
    return faces_data_dir() / "enrollment_preview.jpg"


def models_ready() -> bool:
    return yunet_model_path().is_file() and sface_model_path().is_file()

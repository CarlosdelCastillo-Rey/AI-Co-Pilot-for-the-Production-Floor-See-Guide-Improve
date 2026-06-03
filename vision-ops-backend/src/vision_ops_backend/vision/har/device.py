"""Device selection for HAR (lazy torch import)."""

from __future__ import annotations

_device: str | None = None


def har_device() -> str:
    global _device
    if _device is None:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False

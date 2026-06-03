"""Thread-safe transformers imports (5.x races on concurrent `from transformers import …`)."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_cached: dict[str, Any] | None = None


def get_transformers() -> dict[str, Any]:
    """Return AutoModel / AutoImageProcessor / AutoVideoProcessor with a single import lock."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor

        _cached = {
            "AutoModel": AutoModel,
            "AutoImageProcessor": AutoImageProcessor,
            "AutoVideoProcessor": AutoVideoProcessor,
        }
        return _cached


def preload_transformers() -> None:
    """Call once before parallel HAR live threads start."""
    get_transformers()

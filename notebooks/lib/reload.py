"""Reload lib modules after edits (Jupyter-friendly)."""

from __future__ import annotations

import importlib
import sys


_LIB_MODULES = (
    "lib.constants",
    "lib.paths",
    "lib.inhard",
    "lib.inhard_eda",
    "lib.embeddings",
    "lib.har_model",
    "lib.har_train",
    "lib.har_ensemble",
    "lib.inference",
    "lib.tracking",
    "lib.overlay",
    "lib.memory",
    "lib.eval_video",
    "lib.live_app",
    "lib.pipeline",
)


def reload_lib_modules() -> None:
    for name in _LIB_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])

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
    "lib.crop_extract",
    "lib.human_labels",
    "lib.hitl_ui",
    "lib.har_model",
    "lib.har_train",
    "lib.inference",
    "lib.tracking",
    "lib.overlay",
    "lib.memory",
    "lib.eval_video",
    "lib.live_app",
    "lib.pipeline",
    "lib.pipeline_cache",
    "lib.har_analysis",
    "lib.session_log",
)


def reload_lib_modules() -> None:
    for name in _LIB_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])

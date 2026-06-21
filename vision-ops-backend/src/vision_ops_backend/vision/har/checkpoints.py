"""Load har-research HAR checkpoints from HAR_CHECKPOINT_DIR."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import CKPT_KEY_POWERMEAN7, CKPT_KEY_V2_SSV2, HAR_MODELS
from vision_ops_backend.vision.har.device import torch_available

logger = logging.getLogger(__name__)


def _torch_ok() -> bool:
    return torch_available()


def har_research_root() -> Path:
    from vision_ops_backend.vision.paths import repo_root

    return repo_root() / "har-research"


def ensure_har_research_path() -> Path:
    root = har_research_root()
    path_str = str(root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return root


def checkpoint_dir() -> Path:
    raw = (settings.har_checkpoint_dir or "").strip()
    if not raw:
        return Path()
    path = Path(raw)
    if not path.is_absolute():
        from vision_ops_backend.vision.paths import repo_root

        path = repo_root() / path
    return path


class HarModelRegistry:
    """Lazy registry of har-research HarPredictor instances."""

    def __init__(self) -> None:
        self._loaded = False
        self.class_names: list[str] = []
        self.num_classes: int = 0
        self.predictors: dict[str, Any] = {}
        self._ready: dict[str, bool] = {}
        self._reasons: dict[str, str] = {}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not _torch_ok():
            for spec in HAR_MODELS:
                self._ready[spec.ckpt_key] = False
                self._reasons[spec.ckpt_key] = "torch not installed (uv sync --extra har)"
            return

        ckpt_dir = checkpoint_dir()
        if not ckpt_dir.is_dir():
            for spec in HAR_MODELS:
                self._ready[spec.ckpt_key] = False
                self._reasons[spec.ckpt_key] = f"checkpoint dir missing: {ckpt_dir}"
            return

        try:
            ensure_har_research_path()
            from lib.inference import HarPredictor
            from lib.powermean_ensemble import PowerMeanEnsemblePredictor
            from lib.ssv2_predictor import SSv2HarPredictor
        except Exception as exc:
            for spec in HAR_MODELS:
                self._ready[spec.ckpt_key] = False
                self._reasons[spec.ckpt_key] = f"har-research import failed: {exc}"
            return

        min_conf = float(settings.har_v2_min_confidence)
        for spec in HAR_MODELS:
            try:
                if spec.ckpt_key == CKPT_KEY_POWERMEAN7:
                    pm7_dir = ckpt_dir / CKPT_KEY_POWERMEAN7
                    if not pm7_dir.is_dir():
                        raise FileNotFoundError(f"powermean7 dir missing: {pm7_dir}")
                    predictor = PowerMeanEnsemblePredictor(pm7_dir, min_confidence=min_conf)
                elif spec.ckpt_key == CKPT_KEY_V2_SSV2:
                    pt = ckpt_dir / f"{spec.ckpt_key}.pt"
                    if not pt.is_file():
                        raise FileNotFoundError(f"checkpoint not found: {pt.name}")
                    predictor = SSv2HarPredictor(pt, min_confidence=min_conf)
                else:
                    pt = ckpt_dir / f"{spec.ckpt_key}.pt"
                    if not pt.is_file():
                        raise FileNotFoundError(f"checkpoint not found: {pt.name}")
                    predictor = HarPredictor(pt, min_confidence=min_conf)

                self.predictors[spec.ckpt_key] = predictor
                if not self.class_names:
                    self.class_names = list(predictor.class_names)
                    self.num_classes = len(self.class_names)
                self._ready[spec.ckpt_key] = True
                self._reasons[spec.ckpt_key] = "checkpoint OK"
                logger.info("Loaded HAR checkpoint %s (%s)", spec.model_id, spec.ckpt_key)
            except Exception as exc:
                logger.warning("%s load failed: %s", spec.ckpt_key, exc)
                self.predictors[spec.ckpt_key] = None
                self._ready[spec.ckpt_key] = False
                self._reasons[spec.ckpt_key] = str(exc)

    def model_status(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        out = []
        for spec in HAR_MODELS:
            out.append(
                {
                    "model_id": spec.model_id,
                    "camera_id": spec.camera_id,
                    "label": spec.label,
                    "ckpt_key": spec.ckpt_key,
                    "ready": self._ready.get(spec.ckpt_key, False),
                    "reason": self._reasons.get(spec.ckpt_key, ""),
                }
            )
        return out


_registry: HarModelRegistry | None = None


def get_registry() -> HarModelRegistry:
    global _registry
    if _registry is None:
        _registry = HarModelRegistry()
    return _registry

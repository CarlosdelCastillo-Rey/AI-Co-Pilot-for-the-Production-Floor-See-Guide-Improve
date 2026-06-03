"""Load Avance 4 checkpoints from HAR_CHECKPOINT_DIR."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import (
    CKPT_KEY_DINOV2_MCJEPA,
    CKPT_KEY_DINOV2_PURO,
    CKPT_KEY_VJEPA2_MCJEPA_FROZEN,
    CKPT_KEY_VJEPA2_MCJEPA_PARTIAL,
    CKPT_KEY_VJEPA2_PURO,
    HAR_MODELS,
)
from vision_ops_backend.vision.har.device import har_device, torch_available

logger = logging.getLogger(__name__)

def _torch_ok() -> bool:
    return torch_available()


def checkpoint_dir() -> Path:
    raw = (settings.har_checkpoint_dir or "").strip()
    if not raw:
        return Path()
    path = Path(raw)
    if not path.is_absolute():
        from vision_ops_backend.vision.paths import repo_root

        path = repo_root() / path
    return path


def _find_pt(ckpt_dir: Path, *candidates: str) -> Path | None:
    for name in candidates:
        for sub in ("", "embedding_classifiers", "vjepa2_mcjepa_experiments", "embeddings_parte1"):
            p = ckpt_dir / sub / name if sub else ckpt_dir / name
            if p.is_file():
                return p
    return None


class HarModelRegistry:
    """Lazy registry of loaded classifiers and encoders."""

    def __init__(self) -> None:
        self._loaded = False
        self.class_names: list[str] = []
        self.num_classes: int = 0
        self.dmc_enc: Any = None
        self.classifiers: dict[str, Any] = {}
        self._ready: dict[str, bool] = {}
        self._reasons: dict[str, str] = {}
        self.d_dino: int = 384
        self.d_vjepa2: int = 1024

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

        cfg_path = ckpt_dir / "config.json"
        if cfg_path.is_file():
            with cfg_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            self.class_names = list(cfg.get("class_names", []))
            self.num_classes = int(cfg.get("NUM_CLASSES", len(self.class_names)))

        import torch

        from vision_ops_backend.vision.har.models import (
            DinoToMCJEPAEncoder,
            EmbeddingMLPClassifier,
            MCJEPAHead,
        )

        device = har_device()

        enc_path = _find_pt(ckpt_dir, "dino_to_mcjepa_encoder.pt")
        if enc_path is not None:
            try:
                enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
                d_in = int(enc_ckpt.get("D_DINO", self.d_dino))
                self.dmc_enc = DinoToMCJEPAEncoder(d_in=d_in).to(device)
                self.dmc_enc.load_state_dict(enc_ckpt["state_dict"])
                self.dmc_enc.eval()
            except Exception as exc:
                logger.warning("dino_to_mcjepa_encoder load failed: %s", exc)
                self.dmc_enc = None

        for ckpt_key in (CKPT_KEY_DINOV2_PURO, CKPT_KEY_DINOV2_MCJEPA, CKPT_KEY_VJEPA2_PURO):
            pt = _find_pt(ckpt_dir, f"{ckpt_key}_classifier.pt")
            if pt is None:
                self.classifiers[ckpt_key] = None
                continue
            try:
                ck = torch.load(pt, map_location=device, weights_only=False)
                mlp = EmbeddingMLPClassifier(int(ck["input_dim"]), int(ck["num_classes"])).to(device)
                mlp.load_state_dict(ck["state_dict"])
                mlp.eval()
                self.classifiers[ckpt_key] = mlp
                if not self.class_names and ck.get("class_names"):
                    self.class_names = list(ck["class_names"])
                    self.num_classes = len(self.class_names)
            except Exception as exc:
                logger.warning("%s classifier load failed: %s", ckpt_key, exc)
                self.classifiers[ckpt_key] = None

        for ckpt_key, fname in (
            (CKPT_KEY_VJEPA2_MCJEPA_FROZEN, "VJEPA2_MCJEPA_frozen.pt"),
            (CKPT_KEY_VJEPA2_MCJEPA_PARTIAL, "VJEPA2_MCJEPA_partial_finetune.pt"),
        ):
            pt = _find_pt(ckpt_dir, fname)
            if pt is None:
                self.classifiers[ckpt_key] = None
                continue
            try:
                ck = torch.load(pt, map_location=device, weights_only=False)
                head = MCJEPAHead(
                    d_in=int(ck.get("D_VJEPA2", self.d_vjepa2)),
                    num_classes=int(ck.get("num_classes", self.num_classes or 14)),
                ).to(device)
                head.load_state_dict(ck["head_state_dict"])
                head.eval()
                self.classifiers[ckpt_key] = head
            except Exception as exc:
                logger.warning("%s head load failed: %s", ckpt_key, exc)
                self.classifiers[ckpt_key] = None

        self._refresh_ready()

    def _refresh_ready(self) -> None:
        for spec in HAR_MODELS:
            key = spec.ckpt_key
            if self.classifiers.get(key) is None:
                self._ready[key] = False
                self._reasons[key] = "classifier checkpoint not found"
                continue
            if key == CKPT_KEY_DINOV2_MCJEPA and self.dmc_enc is None:
                self._ready[key] = False
                self._reasons[key] = "missing dino_to_mcjepa_encoder.pt"
                continue
            self._ready[key] = True
            self._reasons[key] = "checkpoint OK"

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

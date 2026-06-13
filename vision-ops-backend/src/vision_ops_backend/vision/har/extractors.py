"""Lazy-loaded DINOv2 and V-JEPA2 feature extractors."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from vision_ops_backend.vision.har.device import har_device
from vision_ops_backend.vision.har.frames import IMAGE_SIZE
from vision_ops_backend.vision.har.transformers_imports import get_transformers, preload_transformers

logger = logging.getLogger(__name__)

DINO_MODEL_NAME = "facebook/dinov2-small"
VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

_dino_extractor: "DINOv2Extractor | None" = None
_vjepa_extractor: "VJEPA2TokenExtractor | None" = None
_extractor_lock = threading.Lock()


def _hf_login() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        logger.warning("HF login failed: %s", exc)


def _pil_images(rgb_frames: list[np.ndarray]):
    from PIL import Image

    return [Image.fromarray(f) for f in rgb_frames]


class DINOv2Extractor:
    def __call__(self, batch_frames: list[list[np.ndarray]]):
        return self.forward(batch_frames)

    def __init__(self, model_name: str):
        tfm = get_transformers()
        AutoImageProcessor = tfm["AutoImageProcessor"]
        AutoModel = tfm["AutoModel"]

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(har_device())
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, batch_frames: list[list[np.ndarray]]):
        import torch

        flat = _pil_images([img for vf in batch_frames for img in vf])
        with torch.inference_mode():
            inp = self.processor(images=flat, return_tensors="pt").to(har_device())
            out = self.model(**inp)
            feats = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out.pooler_output
            b, t = len(batch_frames), len(batch_frames[0])
            return feats.view(b, t, -1)


class VJEPA2TokenExtractor:
    def __call__(self, batch_frames: list[list[np.ndarray]]):
        return self.forward(batch_frames)

    def __init__(self, model):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _parse(outputs: Any):
        if hasattr(outputs, "last_hidden_state"):
            t = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            t = outputs[0]
        elif isinstance(outputs, dict):
            t = None
            for k in ("last_hidden_state", "x_norm_patchtokens"):
                if k in outputs:
                    t = outputs[k]
                    break
            if t is None:
                t = list(outputs.values())[0]
        else:
            t = outputs
        if t.ndim == 2:
            t = t.unsqueeze(1)
        elif t.ndim == 4:
            b, tt, n, d = t.shape
            t = t.reshape(b, tt * n, d)
        return t

    def _preprocess(self, batch_frames: list[list[np.ndarray]]):
        import torch

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
        clips = []
        for video_frames in batch_frames:
            arr = np.stack(video_frames).astype(np.float32) / 255.0
            clips.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
        video = torch.stack(clips).to(har_device())
        return (video - mean.to(har_device())) / std.to(har_device())

    def forward(self, batch_frames: list[list[np.ndarray]]):
        import torch

        video = self._preprocess(batch_frames)
        with torch.inference_mode():
            try:
                out = self.model(video, skip_predictor=True)
            except TypeError:
                out = self.model(video)
            return self._parse(out)


def preload_har_extractors() -> None:
    """Load har-research backbones once before parallel live inference threads."""
    try:
        from vision_ops_backend.vision.har.checkpoints import ensure_har_research_path, get_registry

        ensure_har_research_path()
        get_registry().ensure_loaded()
        from lib.embeddings import _load_vjepa
        from lib.dino_embeddings import _load_dinov2

        _load_vjepa()
        _load_dinov2()
    except Exception as exc:
        logger.warning("HAR v2 preload skipped: %s", exc)


def get_dino_extractor() -> DINOv2Extractor:
    global _dino_extractor
    if _dino_extractor is not None:
        return _dino_extractor
    with _extractor_lock:
        if _dino_extractor is None:
            _hf_login()
            preload_transformers()
            _dino_extractor = DINOv2Extractor(DINO_MODEL_NAME)
    return _dino_extractor


def get_vjepa_extractor() -> VJEPA2TokenExtractor:
    global _vjepa_extractor
    if _vjepa_extractor is not None:
        return _vjepa_extractor
    with _extractor_lock:
        if _vjepa_extractor is None:
            AutoModel = get_transformers()["AutoModel"]
            _hf_login()
            vj_enc = None
            for kwargs in (
                {"trust_remote_code": True},
                {"trust_remote_code": True, "ignore_mismatched_sizes": True},
            ):
                try:
                    vj_enc = AutoModel.from_pretrained(VJEPA2_MODEL_ID, **kwargs).to(har_device())
                    break
                except Exception as exc:
                    logger.warning("V-JEPA2 load attempt failed: %s", exc)
            if vj_enc is None:
                raise RuntimeError(f"Failed to load V-JEPA2 model {VJEPA2_MODEL_ID}")
            _vjepa_extractor = VJEPA2TokenExtractor(vj_enc)
    return _vjepa_extractor

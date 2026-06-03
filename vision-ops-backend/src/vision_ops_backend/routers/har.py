"""HAR activity recognition APIs (Avance 4 models)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.checkpoints import checkpoint_dir, get_registry
from vision_ops_backend.vision.har.constants import HAR_MODEL_IDS, is_har_camera, spec_for_model
from vision_ops_backend.vision.har.device import torch_available
from vision_ops_backend.vision.har.live_stream import get_har_live_manager
from vision_ops_backend.vision.har.probe_jobs import get_probe_all_job, start_probe_all_job
from vision_ops_backend.vision.har.shared_clip import shared_clip_metadata
from vision_ops_backend.vision.store import load_last_probe

router = APIRouter(prefix="/api/vision/har", tags=["vision-har"])


class HarProbeBody(BaseModel):
    clip_path: str | None = Field(None, description="Optional .mp4 override")
    reshuffle_videos: bool = Field(True, description="Reassign mock videos per camera before probe-all")


def _har_disabled() -> None:
    if not settings.har_enabled:
        raise HTTPException(status_code=503, detail="HAR probes disabled (HAR_ENABLED=false)")


def _require_torch() -> None:
    if not torch_available():
        raise HTTPException(
            status_code=503,
            detail="torch/transformers not installed — run: cd vision-ops-backend && uv sync --extra har",
        )


@router.get("/models")
def har_models() -> dict:
    _har_disabled()
    registry = get_registry()
    ckpt = checkpoint_dir()
    return {
        "enabled": settings.har_enabled,
        "checkpoint_dir": str(ckpt),
        "checkpoint_dir_exists": ckpt.is_dir(),
        "torch_available": torch_available(),
        "models": registry.model_status(),
    }


@router.get("/status")
def har_status() -> dict:
    _har_disabled()
    cameras: dict = {}
    for model_id in HAR_MODEL_IDS:
        spec = spec_for_model(model_id)
        if spec is None:
            continue
        probe = load_last_probe(spec.camera_id)
        entry: dict = {
            "model_id": model_id,
            "camera_id": spec.camera_id,
            "label": spec.label,
            "last_probe": probe is not None,
        }
        if probe:
            pred = probe.get("prediction") or {}
            entry["prediction"] = pred
            entry["source"] = probe.get("source")
            entry["backend"] = probe.get("backend")
        cameras[model_id] = entry
    meta = shared_clip_metadata()
    return {
        "ready": torch_available(),
        "shared_clip": meta,
        "models": cameras,
    }


@router.get("/live")
def har_live_all() -> dict:
    """Latest sliding-window predictions for all HAR mock cameras."""
    _har_disabled()
    manager = get_har_live_manager()
    return {"enabled": settings.har_live_enabled, "cameras": manager.all_states()}


@router.get("/live/{camera_id}")
def har_live_camera(camera_id: str) -> dict:
    _har_disabled()
    if not is_har_camera(camera_id):
        raise HTTPException(status_code=404, detail="Not a HAR camera")
    state = get_har_live_manager().get_state(camera_id)
    if state is None:
        raise HTTPException(status_code=503, detail="HAR live stream not running")
    return state


class HarPlaybackBody(BaseModel):
    playing: bool = Field(..., description="false = pause live decode + inference")


@router.post("/live/{camera_id}/playback")
def har_live_playback(camera_id: str, body: HarPlaybackBody) -> dict:
    """Sync backend live loop with UI play/pause."""
    _har_disabled()
    if not is_har_camera(camera_id):
        raise HTTPException(status_code=404, detail="Not a HAR camera")
    ok = get_har_live_manager().set_playback(camera_id, playing=body.playing)
    if not ok:
        raise HTTPException(status_code=503, detail="HAR live stream not running")
    return {"camera_id": camera_id, "playing": body.playing}


@router.get("/shared-clip")
def har_shared_clip(clip_path: str | None = None) -> dict:
    _har_disabled()
    return shared_clip_metadata(clip_path=clip_path)


@router.post("/{model_id}/probe")
def har_probe(model_id: str, body: HarProbeBody | None = None) -> dict:
    _har_disabled()
    _require_torch()
    if model_id not in HAR_MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"model_id must be one of {list(HAR_MODEL_IDS)}")
    clip = body.clip_path if body else None
    try:
        from vision_ops_backend.vision.har.probe_runner import run_har_camera_probe

        probe = run_har_camera_probe(model_id, clip_path=clip)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "probe": probe}


@router.post("/probe-all/async")
def har_probe_all_async(body: HarProbeBody | None = None) -> dict:
    """Start probe-all in background; poll GET /probe-all/jobs/{job_id} (avoids proxy timeouts)."""
    _har_disabled()
    _require_torch()
    clip = body.clip_path if body else None
    reshuffle = body.reshuffle_videos if body else True
    job_id = start_probe_all_job(clip_path=clip, reshuffle_videos=reshuffle)
    return {"status": "running", "job_id": job_id}


@router.get("/probe-all/jobs/{job_id}")
def har_probe_all_job(job_id: str) -> dict:
    _har_disabled()
    job = get_probe_all_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/probe-all")
def har_probe_all(body: HarProbeBody | None = None) -> dict:
    """Synchronous probe-all (may exceed HTTP timeouts; prefer /probe-all/async from the UI)."""
    _har_disabled()
    _require_torch()
    clip = body.clip_path if body else None
    try:
        from vision_ops_backend.vision.har.probe_runner import run_all_har_probes

        reshuffle = body.reshuffle_videos if body else True
        return run_all_har_probes(clip_path=clip, reshuffle_videos=reshuffle)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

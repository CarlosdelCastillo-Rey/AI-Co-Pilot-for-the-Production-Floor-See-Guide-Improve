"""HAR activity recognition APIs (Avance 4 models)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.checkpoints import checkpoint_dir, get_registry
from vision_ops_backend.vision.har.constants import HAR_BENCH_CAMERA_ID, HAR_MODEL_IDS, is_har_bench_camera, is_har_camera, spec_for_model
from vision_ops_backend.vision.har.har_bench import get_har_bench_manager
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


class HarPlaybackBody(BaseModel):
    playing: bool = Field(..., description="false = pause live decode + inference")


@router.post("/live/playback-all")
def har_live_playback_all(body: HarPlaybackBody) -> dict:
    """Pause or resume all HAR live streams (decode + model inference)."""
    _har_disabled()
    count = get_har_live_manager().set_playback_all(playing=body.playing)
    bench_ok = get_har_bench_manager().set_playback(playing=body.playing)
    return {
        "playing": body.playing,
        "cameras_updated": count,
        "bench_updated": bench_ok,
    }


@router.get("/live/{camera_id}")
def har_live_camera(camera_id: str) -> dict:
    _har_disabled()
    if is_har_bench_camera(camera_id):
        state = get_har_bench_manager().get_state()
        if state is None:
            raise HTTPException(status_code=503, detail="HAR bench stream not running")
        return state
    if not is_har_camera(camera_id):
        raise HTTPException(status_code=404, detail="Not a HAR camera")
    state = get_har_live_manager().get_state(camera_id)
    if state is None:
        raise HTTPException(status_code=503, detail="HAR live stream not running")
    return state


class HarBenchConfigBody(BaseModel):
    infer_every: int | None = Field(None, ge=1, le=120)
    buffer_frames: int | None = Field(None, ge=8, le=128)
    stream_fps: int | None = Field(None, ge=1, le=60)
    show_heatmap: bool | None = None
    show_yolo_boxes: bool | None = None
    top_k: int | None = Field(None, ge=1, le=10)
    ingest_logs: bool | None = None


class HarBenchModelBody(BaseModel):
    model_id: str


class HarBenchVideoBody(BaseModel):
    video: str = Field(..., description="Mock video filename from public/mock-videos")


@router.post("/live/{camera_id}/playback")
def har_live_playback(camera_id: str, body: HarPlaybackBody) -> dict:
    """Sync backend live loop with UI play/pause."""
    _har_disabled()
    if is_har_bench_camera(camera_id):
        ok = get_har_bench_manager().set_playback(playing=body.playing)
        if not ok:
            raise HTTPException(status_code=503, detail="HAR bench stream not running")
        return {"camera_id": camera_id, "playing": body.playing}
    if not is_har_camera(camera_id):
        raise HTTPException(status_code=404, detail="Not a HAR camera")
    ok = get_har_live_manager().set_playback(camera_id, playing=body.playing)
    if not ok:
        raise HTTPException(status_code=503, detail="HAR live stream not running")
    return {"camera_id": camera_id, "playing": body.playing}


@router.get("/bench")
def har_bench_snapshot() -> dict:
    """Sandbox bench: one camera, switchable model/video/hyperparameters."""
    _har_disabled()
    return get_har_bench_manager().snapshot()


@router.patch("/bench/config")
def har_bench_config(body: HarBenchConfigBody) -> dict:
    _har_disabled()
    stream = get_har_bench_manager().get_stream()
    if stream is None:
        raise HTTPException(status_code=503, detail="HAR bench stream not running")
    payload = body.model_dump(exclude_none=True)
    config = stream.update_config(**payload)
    return {"camera_id": HAR_BENCH_CAMERA_ID, "config": config}


@router.post("/bench/model")
def har_bench_model(body: HarBenchModelBody) -> dict:
    _har_disabled()
    if body.model_id not in HAR_MODEL_IDS:
        raise HTTPException(status_code=404, detail=f"model_id must be one of {list(HAR_MODEL_IDS)}")
    stream = get_har_bench_manager().get_stream()
    if stream is None:
        raise HTTPException(status_code=503, detail="HAR bench stream not running")
    stream.set_model(body.model_id)
    return {"camera_id": HAR_BENCH_CAMERA_ID, "model_id": body.model_id, "state": stream.get_state()}


@router.post("/bench/video")
def har_bench_video(body: HarBenchVideoBody) -> dict:
    _har_disabled()
    from pathlib import Path

    from vision_ops_backend.vision.mock_videos import list_mock_video_files, mock_videos_dir

    name = Path(body.video).name
    matches = [p for p in list_mock_video_files() if p.name == name]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found in {mock_videos_dir()}: {name}",
        )
    stream = get_har_bench_manager().get_stream()
    if stream is None:
        raise HTTPException(status_code=503, detail="HAR bench stream not running")
    stream.set_video(matches[0])
    return {"camera_id": HAR_BENCH_CAMERA_ID, "video": name, "state": stream.get_state()}


@router.post("/bench/reset")
def har_bench_reset() -> dict:
    _har_disabled()
    stream = get_har_bench_manager().get_stream()
    if stream is None:
        raise HTTPException(status_code=503, detail="HAR bench stream not running")
    session_id = stream.reset_session()
    return {"camera_id": HAR_BENCH_CAMERA_ID, "session_id": session_id, "state": stream.get_state()}


@router.get("/bench/live")
def har_bench_live() -> dict:
    _har_disabled()
    state = get_har_bench_manager().get_state()
    if state is None:
        raise HTTPException(status_code=503, detail="HAR bench stream not running")
    return state


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

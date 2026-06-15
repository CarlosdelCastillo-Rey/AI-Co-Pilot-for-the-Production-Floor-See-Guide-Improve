"""Multi-slot mock video wall — only visible slots decode + run HAR."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from vision_ops_backend.config import settings
from vision_ops_backend.vision.har.constants import (
    HAR_BENCH_CAMERA_ID,
    HAR_MODELS,
    MOCK_WALL_SLOT_COUNT,
    is_har_bench_camera,
    mock_slot_camera_id,
    mock_slot_index,
)
from vision_ops_backend.vision.har.har_bench import HarBenchStream
from vision_ops_backend.vision.mock_videos import list_mock_video_files, public_url_for_video

logger = logging.getLogger(__name__)


def visible_slot_indices(
    layout: str,
    *,
    full_view_index: int,
    video_count: int,
) -> list[int]:
    """Which slot indices should decode + infer (0..video_count-1 only)."""
    n = min(MOCK_WALL_SLOT_COUNT, max(video_count, 0))
    if n == 0:
        return []
    if layout == "full":
        idx = max(0, min(full_view_index, n - 1))
        return [idx]
    if layout == "dual":
        return list(range(min(2, n)))
    return list(range(min(4, n)))


class HarMockWallManager:
    """Up to four mock-video streams; playback/inference only on visible slots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[int, HarBenchStream] = {}
        self._layout = "dual"
        self._full_view_index = 0
        self._playing = False
        self._primary_slot = 0

    def start(self) -> None:
        if not settings.har_enabled or not settings.har_live_enabled:
            return
        threading.Thread(target=self._start_worker, daemon=True, name="har-mock-wall-boot").start()

    def _preload_models(self) -> None:
        try:
            from vision_ops_backend.vision.har.checkpoints import get_registry
            from vision_ops_backend.vision.har.extractors import preload_har_extractors

            get_registry().ensure_loaded()
            preload_har_extractors()
        except Exception as exc:
            logger.warning("HAR mock wall preload failed: %s", exc)

    def _ensure_streams(self, videos: list[Path] | None = None) -> int:
        """Create decode threads immediately; do not block on torch/HF preload."""
        pool = videos if videos is not None else list_mock_video_files()
        if not pool:
            return 0
        default_model = HAR_MODELS[0].model_id
        created = 0
        for i in range(min(MOCK_WALL_SLOT_COUNT, len(pool))):
            with self._lock:
                if i in self._streams:
                    continue
            stream = HarBenchStream(
                default_model,
                pool[i],
                camera_id=mock_slot_camera_id(i),
            )
            stream.start()
            with self._lock:
                self._streams[i] = stream
            created += 1
        return created

    def _start_worker(self) -> None:
        videos = list_mock_video_files()
        if not videos:
            logger.warning("HAR mock wall: no mock videos found")
            return
        n = self._ensure_streams(videos)
        logger.info("HAR mock wall started (%d slot(s))", n)
        threading.Thread(target=self._preload_models, daemon=True, name="har-mock-wall-preload").start()

    def stop(self) -> None:
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
        for stream in streams:
            stream.stop()

    def get_stream(self, slot: int | None = None) -> HarBenchStream | None:
        self._ensure_streams()
        with self._lock:
            if slot is not None:
                return self._streams.get(slot)
            return self._streams.get(self._primary_slot) or (
                self._streams.get(0) if self._streams else None
            )

    def get_stream_by_camera(self, camera_id: str) -> HarBenchStream | None:
        if is_har_bench_camera(camera_id):
            return self.get_stream()
        idx = mock_slot_index(camera_id)
        if idx is None:
            return None
        return self.get_stream(idx)

    def get_state(self, slot: int | None = None) -> dict[str, Any] | None:
        stream = self.get_stream(slot)
        return stream.get_state() if stream else None

    def set_playback(self, *, playing: bool) -> bool:
        if not self._ensure_streams():
            return False
        self._playing = playing
        self._apply_playback()
        return bool(self._streams)

    def sync(
        self,
        *,
        layout: str,
        playing: bool,
        model_id: str,
        active_video: str,
        full_view_index: int,
    ) -> dict[str, Any]:
        videos = list_mock_video_files()
        name_to_idx = {p.name: i for i, p in enumerate(videos[:MOCK_WALL_SLOT_COUNT])}
        primary = name_to_idx.get(Path(active_video).name, 0)

        with self._lock:
            self._layout = layout
            self._playing = playing
            self._full_view_index = max(0, full_view_index)
            self._primary_slot = primary

        for i, path in enumerate(videos[:MOCK_WALL_SLOT_COUNT]):
            stream = self.get_stream(i)
            if stream is None:
                stream = HarBenchStream(model_id, path, camera_id=mock_slot_camera_id(i))
                stream.start()
                with self._lock:
                    self._streams[i] = stream
            else:
                stream.set_model(model_id)
                if stream.video_path.name != path.name:
                    stream.set_video(path)

        self._apply_playback()
        return self.snapshot()

    def _apply_playback(self) -> None:
        videos = list_mock_video_files()
        visible = visible_slot_indices(
            self._layout,
            full_view_index=self._full_view_index,
            video_count=len(videos),
        )
        visible_set = set(visible)
        with self._lock:
            streams = dict(self._streams)
            playing = self._playing
        for i, stream in streams.items():
            active = playing and i in visible_set
            stream.set_playback_active(active)

    def patch_config(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            streams = list(self._streams.values())
        if not streams:
            return {}
        config = streams[0].update_config(**kwargs)
        for stream in streams[1:]:
            stream.update_config(**kwargs)
        return config

    def set_model(self, model_id: str) -> None:
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            stream.set_model(model_id)

    def set_video_by_name(self, video_path: Path) -> None:
        videos = list_mock_video_files()
        for i, p in enumerate(videos[:MOCK_WALL_SLOT_COUNT]):
            if p.name == video_path.name:
                stream = self.get_stream(i)
                if stream:
                    stream.set_video(p)
                with self._lock:
                    self._primary_slot = i
                break
        self._apply_playback()

    def reset_primary_session(self) -> str:
        stream = self.get_stream()
        if stream is None:
            return ""
        return stream.reset_session()

    def snapshot(self) -> dict[str, Any]:
        from vision_ops_backend.vision.har.checkpoints import get_registry

        self._ensure_streams()
        registry = get_registry()
        videos = [
            {"name": p.name, "url": public_url_for_video(p)}
            for p in list_mock_video_files()
        ]
        models = [
            {
                "model_id": spec.model_id,
                "label": spec.label,
                "ready": registry._ready.get(spec.ckpt_key, False),
            }
            for spec in HAR_MODELS
        ]
        primary = self.get_stream()
        state = primary.get_state() if primary else None
        base = settings.public_api_base.rstrip("/")
        slot_states: dict[str, dict[str, Any]] = {}
        with self._lock:
            primary_slot = self._primary_slot
            for i, stream in self._streams.items():
                slot_states[mock_slot_camera_id(i)] = stream.get_state()
        return {
            "enabled": settings.har_enabled and settings.har_live_enabled,
            "camera_id": HAR_BENCH_CAMERA_ID,
            "stream_url": f"{base}/api/cameras/{mock_slot_camera_id(primary_slot)}/stream",
            "state": state,
            "videos": videos,
            "models": models,
            "slots": slot_states,
            "layout": self._layout,
            "full_view_index": self._full_view_index,
            "primary_slot": primary_slot,
        }


_manager: HarMockWallManager | None = None


def get_mock_wall_manager() -> HarMockWallManager:
    global _manager
    if _manager is None:
        _manager = HarMockWallManager()
    return _manager

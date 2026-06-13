"""In-process HAR v2 session audit (no HTTP)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from vision_ops_backend.config import settings

logger = logging.getLogger(__name__)


def new_session_id() -> str:
    return f"har-{uuid.uuid4().hex[:12]}"


class HarV2SessionClient:
    """Session logger — mirrors har-research HarSessionLogger via in-process store."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        source: str = "bench",
        camera_id: str | None = None,
        video_name: str | None = None,
        model_id: str | None = None,
        model_tag: str | None = None,
        checkpoint_name: str | None = None,
        class_names: list[str] | None = None,
        hyperparams: dict[str, Any] | None = None,
        use_person_registry: bool = True,
    ) -> None:
        self.session_id = session_id or new_session_id()
        self.use_person_registry = use_person_registry
        self.video_name = video_name
        self._enabled = settings.har_v2_session_enabled
        if self._enabled:
            self._create_session(
                source=source,
                camera_id=camera_id,
                video_name=video_name,
                model_id=model_id,
                model_tag=model_tag,
                checkpoint_name=checkpoint_name,
                class_names=class_names or [],
                hyperparams=hyperparams or {},
            )

    def _create_session(self, **kwargs: Any) -> None:
        try:
            from vision_ops_alerting.db.session import SessionLocal
            from vision_ops_alerting.services.har_session_store import create_audit_session

            with SessionLocal() as db:
                create_audit_session(db, session_id=self.session_id, **kwargs)
                db.commit()
        except Exception as exc:
            logger.warning("HAR v2 session create failed: %s", exc)

    def log_inference(
        self,
        *,
        track_id: int,
        frame_idx: int,
        bbox: list[int] | None,
        prediction: dict[str, Any],
        label_changed: bool = False,
        uncertain: bool = False,
        infer_ms: float | None = None,
        crop_jpeg: bytes | None = None,
        frame_jpeg: bytes | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        try:
            from vision_ops_alerting.db.session import SessionLocal
            from vision_ops_alerting.services.har_session_store import record_session_event

            with SessionLocal() as db:
                result = record_session_event(
                    db,
                    session_id=self.session_id,
                    track_id=track_id,
                    frame_idx=frame_idx,
                    bbox=bbox,
                    prediction=prediction,
                    label_changed=label_changed,
                    uncertain=uncertain,
                    infer_ms=infer_ms,
                    crop_jpeg=crop_jpeg,
                    frame_jpeg=frame_jpeg,
                    embedding=embedding,
                    use_person_registry=self.use_person_registry,
                    video_name=self.video_name,
                )
                db.commit()
                return result
        except Exception as exc:
            logger.warning("HAR v2 event log failed: %s", exc)
        return None

    def finalize(self) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        try:
            from vision_ops_alerting.db.session import SessionLocal
            from vision_ops_alerting.services.har_session_store import finalize_audit_session

            with SessionLocal() as db:
                result = finalize_audit_session(db, self.session_id)
                db.commit()
                return result
        except Exception as exc:
            logger.warning("HAR v2 session finalize failed: %s", exc)
        return None

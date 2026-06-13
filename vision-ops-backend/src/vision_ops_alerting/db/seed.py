from __future__ import annotations

import json

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import Camera


def seed_if_empty(db: Session) -> None:
    """No demo timeline/rules seed — data comes from live HAR and user actions."""
    return


def seed_cameras_if_empty(db: Session) -> None:
    """Fresh DB: HAR cameras are added via seed_har_cameras_if_missing."""
    if db.query(Camera).count() > 0:
        return


def disable_legacy_cameras(db: Session) -> None:
    """Hide unused industrial demo cameras (superseded by HAR mock feeds)."""
    for cam_id in ("cam-01", "cam-02", "cam-03", "cam-har-03", "cam-har-04", "cam-har-05"):
        row = db.get(Camera, cam_id)
        if row:
            row.enabled = False


def seed_har_cameras_if_missing(db: Session) -> None:
    from vision_ops_alerting.services.mock_videos import assign_videos_to_har_cameras, public_url_for_video

    video_map = assign_videos_to_har_cameras()
    har_specs = [
        ("cam-har-01", "HAR — V-JEPA v2", "v2-vjepa"),
        ("cam-har-02", "HAR — DINOv2", "v2-dinov2"),
    ]
    for cam_id, name, model in har_specs:
        row = db.get(Camera, cam_id)
        mock_path = video_map.get(cam_id)
        mock_cfg = (
            json.dumps(
                {
                    "mockVideoFile": mock_path.name if mock_path else None,
                    "mockVideoUrl": public_url_for_video(mock_path) if mock_path else None,
                }
            )
            if mock_path
            else None
        )
        if row:
            if mock_cfg and not row.config_json:
                row.config_json = mock_cfg
            continue
        image_url = public_url_for_video(mock_path) if mock_path else ""
        db.add(
            Camera(
                id=cam_id,
                name=name,
                location="HAR Lab / Mock video (live)",
                image_url=image_url,
                source_type="mock_video",
                status="live",
                zone="HAR LIVE",
                inference_model=model,
                enabled=True,
                config_json=mock_cfg,
            )
        )

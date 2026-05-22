"""Mock industrial camera cards (cam-01, cam-02) enriched with /api/vision probe results."""

from __future__ import annotations

from typing import Any

from vision_ops_backend.vision import CAM_ASSEMBLY, CAM_WAREHOUSE
from vision_ops_backend.vision.probe_runner import ensure_camera_still
from vision_ops_backend.vision.store import (
    heatmap_available,
    load_last_probe,
    overlays_for_camera,
    preview_available,
    still_available,
)

_MOCK_META: dict[str, dict[str, Any]] = {
    CAM_ASSEMBLY: {
        "name": "Camera 01 - Assembly",
        "location": "Main Hall / Line 4",
        "image": (
            "https://lh3.googleusercontent.com/aida-public/AB6AXuCtdXg1qgVaATzDFV4GlsmN6CkUoyf1Z5phhagAyhKszH_SM-XO_97YtvK6_rhFO1EC5ny-HEVEIP1Wz2oRu_LYR5IOJVdWCFu0csqXHHopNJFR5fD0-ooCwFJKB6q8aDm0yPLzbPKtGYY7AQThGRta6LJSy3krV7Ze8hd6UnLyT7J6eiI11S6664PLbZ9IWYxq4SeOpkEwSm2g-eCVrZOwtq7YjtLw8HV8C_23jAB7xWqoV3X1prHnLVBcL0GHSMS0ayxsVG6QrqY"
        ),
        "coords_base": "42.3601° N, 71.0589° W",
    },
    CAM_WAREHOUSE: {
        "name": "Camera 02 - Warehouse",
        "location": "Loading Dock / Zone B",
        "image": (
            "https://lh3.googleusercontent.com/aida-public/AB6AXuD0WaxmzB30i4UvnXB1kC5UAjuno45jZ0-lYANMKEPRwQpqZH639_Ac7yPq9EJwxynwUc8jWfLtP6TuMgSHCc4R8QV2j8GXrckY0OSBfzsbliQXwp7qGaM_dgRn_CJ_-YN2M84FIR_4mTkNuSeOUqYZv-zFb-PGGtCtruaN4-mtsBu_sa6AvIDb6JnHCMjexxBix8FdInNk_8IbvGQsjiq1a0uDuXJABCY-cv8XDEYCM9YPSBwMnKCs_vAm8Ksn_xYUDxWv9393I"
        ),
        "coords_base": "42.3610° N, 71.0595° W",
    },
}


def build_industrial_camera(camera_id: str) -> dict[str, Any]:
    ensure_camera_still(camera_id)
    meta = _MOCK_META[camera_id]
    probe = load_last_probe(camera_id)
    overlays = overlays_for_camera(camera_id)
    if not overlays and camera_id == CAM_ASSEMBLY:
        overlays = [
            {
                "type": "machine",
                "label": "Run Vision Lab probe for DINO heatmap",
                "top": "30%",
                "left": "40%",
                "width": "20%",
                "height": "25%",
                "variant": "tertiary",
            }
        ]
    if not overlays and camera_id == CAM_WAREHOUSE:
        overlays = [
            {
                "type": "forklift",
                "label": "Run Vision Lab — V-JEPA probe",
                "top": "60%",
                "left": "20%",
                "width": "25%",
                "height": "30%",
                "variant": "tertiary",
            }
        ]

    vision_tag = ""
    if probe:
        backend = probe.get("backend", "")
        if camera_id == CAM_ASSEMBLY:
            vision_tag = f" | DINO {backend}" if backend else " | DINO OK"
        else:
            score = probe.get("anomaly_score")
            if score is not None:
                vision_tag = f" | V-JEPA {max(0.0, float(score)):.2f}"
            else:
                vision_tag = " | V-JEPA OK"

    image_url = meta["image"]
    if still_available(camera_id):
        image_url = f"/api/vision/artifacts/{camera_id}/still"

    payload: dict[str, Any] = {
        "id": camera_id,
        "name": meta["name"],
        "location": meta["location"],
        "image": image_url,
        "status": "live",
        "coords": f"{meta['coords_base']} | VISION{vision_tag}",
        "overlays": overlays,
        "visionProbe": probe,
    }
    if preview_available(camera_id):
        payload["previewUrl"] = f"/api/vision/artifacts/{camera_id}/preview"
    if heatmap_available(camera_id):
        payload["heatmapUrl"] = f"/api/vision/artifacts/{camera_id}/overlay"
    return payload


def list_industrial_cameras() -> list[dict[str, Any]]:
    return [build_industrial_camera(CAM_ASSEMBLY), build_industrial_camera(CAM_WAREHOUSE)]

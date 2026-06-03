from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import AlertRule, AnalyticsDaily, AnalyticsHeatmap, Camera, Event, new_id


def seed_if_empty(db: Session) -> None:
    if db.query(AlertRule).count() > 0:
        return

    rules = [
        AlertRule(
            id="rule-1",
            icon="polyline",
            title="Geofencing Intrusion (North Dock)",
            description="Triggers if unauthorized personnel enter Zone A-14 after 22:00.",
            zone="ZONE A-14",
            case_type="user_left_position",
            severity="CRITICAL",
            enabled=True,
        ),
        AlertRule(
            id="rule-2",
            icon="groups",
            title="Crowd Density Threshold",
            description="Main Lobby monitoring. Trigger when person count exceeds 15.",
            zone="MAIN LOBBY",
            case_type="unknown",
            severity="WARNING",
            enabled=True,
        ),
        AlertRule(
            id="rule-3",
            icon="warning",
            title="PPE Compliance Check",
            description="Automated hard-hat and high-vis vest detection at Gate 3.",
            zone="GATE 3",
            case_type="unknown",
            severity="DISABLED",
            enabled=False,
        ),
        AlertRule(
            id="rule-4",
            icon="fire_extinguisher",
            title="Fire/Smoke Detection",
            description="Thermal imaging integration for early fire detection in Warehouse B.",
            zone="WHSE B",
            case_type="forklift_in_zone",
            severity="CRITICAL",
            enabled=True,
        ),
        AlertRule(
            id="rule-5",
            icon="schedule",
            title="Operator Idle Detection",
            description="Alert when operator idle exceeds 5 minutes on assembly line.",
            zone="LINE 4",
            case_type="user_not_working",
            severity="WARNING",
            enabled=True,
        ),
    ]
    db.add_all(rules)

    now = datetime.now(timezone.utc)
    events = [
        Event(
            id="evt-1",
            rule_id="rule-4",
            site_id="site-01",
            line_id="line-4",
            camera_id="cam-01",
            case_type="forklift_in_zone",
            severity="critical",
            title="Conveyor 2 Blockage Detected",
            description=(
                "Flow stopped for 3:12 due to material buildup at Transfer Point Alpha. "
                "Automated e-stop triggered. Operator recalibration required before restart."
            ),
            meta_json=json.dumps([{"icon": "person", "text": "J. Miller"}, {"icon": "location_on", "text": "Zone 4 / Floor 2"}]),
            thumbnail_url="https://lh3.googleusercontent.com/aida-public/AB6AXuBp3bQPzQYw62zX_2pzW3Le1-ZFn9mAobM6DAzL7jaK90xOBNPRzHhfUesgAadlpjhz9ex-CPUQ5sujRcXQAIgBKm_a9sfLVCFlKWtFal_Ps3Ql17YDYsuq1rxGtAqG9lxV813sMd2V513vwwM-x8gSQiQs4E05Vl_Jue-RdN-Bgwigs3Gzmxx05yuZnqXQfPmQv-p3s0a5zanhbJrj6VS7CXsn3uaciPGI4A99y25oo14uwZeMFrSi3ZHR_s-8iJfIBR5cvjRpftM",
            clip_duration_sec=14,
            occurred_at=now.replace(hour=14, minute=23, second=12),
        ),
        Event(
            id="evt-2",
            site_id="site-01",
            line_id="line-a",
            camera_id="cam-02",
            case_type="unknown",
            severity="warning",
            title="Thermal Deviation: Motor M-09",
            description=(
                "Operating temperature exceeded threshold of 85°C. Current reading: 92.4°C. "
                "Cooling sequence initiated manually. Scheduling maintenance for next shift."
            ),
            meta_json=json.dumps([{"icon": "sensors", "text": "Heat Sensor HS-41"}]),
            thumbnail_url="https://lh3.googleusercontent.com/aida-public/AB6AXuCnsrmrUduGm-QCjh10fyqvme8ZMCbKRjwABPesR4oxCruGBfqNCi8SUe8X2TVRiF052P7j7ocQUD_ulTVZFPgvNErB3NZhu8bMos0pAljzxOEeKp0mF7_2YiTO1fNesT5MfbEyN4omirInz5n7s-GZfd9_YxgIfi0LR6OLVg4_zDhWPyY5uBxxKy3RSjg3GN9oNUgXPmGUoxS7APDt78Q7icUBckGq9rCyR07l19id848lmVgAbTquisW7ZDDz7cEbsSUdwgSkPGA",
            clip_duration_sec=8,
            occurred_at=now.replace(hour=13, minute=45, second=5),
        ),
        Event(
            id="evt-3",
            site_id="site-01",
            line_id="line-b",
            camera_id="cam-01",
            case_type="unknown",
            severity="info",
            title="Operator Handoff: Shift B",
            description=(
                "Standard procedure completed. All systems verified at 98.4% efficiency. "
                "Inventory levels marked as sufficient for the next 8-hour cycle."
            ),
            meta_json=json.dumps([{"icon": "sync_alt", "text": "Auto-logged"}]),
            thumbnail_url="https://lh3.googleusercontent.com/aida-public/AB6AXuA8QSJuWijkQiH0Uh59JTlgi2pmkroNvjYpqGsPSWBQhVy75kVNp29LsbVxQkkcOhr0_IjPjn1wvj7BhNLU-wKALbVVHiFd1KUJWO73VC_a8ygEtXF1PNwZVDH7Tk4y3nkmIryRMuV-lozrr-d-Ruv5Zfzf7vOtXbw8iHfBZWSB-VALVnyaBxmbV3boSV7Ht9yFVnJP13RoXH4IchlJ6WexsoAo_gS76TxFoCPJPGZ-JBuSl57Kl3PeIyaFzhF52BANco8nFQ166do",
            clip_duration_sec=32,
            occurred_at=now.replace(hour=12, minute=10, second=44),
        ),
        Event(
            id="evt-4",
            rule_id="rule-1",
            site_id="site-01",
            line_id="line-c",
            camera_id="cam-02",
            case_type="user_left_position",
            severity="critical",
            title="Unidentified Personnel in Restricted Zone",
            description=(
                "Vision AI detected unauthorized human presence in Robot Cell G-2. "
                "Safety lockdown engaged. Supervisor notification sent via secure link."
            ),
            meta_json=json.dumps([{"icon": "visibility", "text": "Vision Node 12"}]),
            thumbnail_url="https://lh3.googleusercontent.com/aida-public/AB6AXuDzVywH68hQHg65TTMaExnZY0KmdNovr8Et6ZtjbO4O0uCheM1Y1cK3yDjdY1LdmwOYCpZ8d3UGHokqIfh2EsPI7Y939h06x8lJqzumKo_-_wusiCZL6Ze3hbOXpn0ie1l5LH6LF52MyCcCUV3t_8ajOi4a-3ZSU1jKSbO1oJ51Ky-7aXwNZal7Zcezfqn00xCQNSu756VMRgpyfjIM2GAQqPF7i1Hc9cb6LrNVpFlxAyDITx3nLSG3zYqeyEjYl9gMB8wygpgGckw",
            clip_duration_sec=45,
            occurred_at=now.replace(hour=9, minute=12, second=0),
        ),
    ]
    db.add_all(events)

    date_str = now.strftime("%Y-%m-%d")
    db.add(
        AnalyticsDaily(
            id=new_id("daily"),
            event_date=date_str,
            shift="morning",
            camera_id=None,
            incident_count=2,
            uptime_pct=94.2,
            flow_efficiency_pct=94.2,
        )
    )

    grid = {
        "width": 10,
        "height": 10,
        "cells": [[0.1 + (i + j) * 0.05 for j in range(10)] for i in range(10)],
        "hotspots": [{"x": 5, "y": 5, "severity": "critical", "label": "Zone A-12"}],
    }
    db.add(
        AnalyticsHeatmap(
            id=new_id("heatmap"),
            camera_id="cam-01",
            event_date=date_str,
            shift="morning",
            grid_json=json.dumps(grid),
            anomaly_count=3,
            sensors_active=42,
        )
    )


def seed_cameras_if_empty(db: Session) -> None:
    """Fresh DB: no legacy cam-01/02/03 — HAR cameras are added via seed_har_cameras_if_missing."""
    if db.query(Camera).count() > 0:
        return


def disable_legacy_cameras(db: Session) -> None:
    """Hide unused industrial demo cameras (superseded by HAR mock feeds)."""
    for cam_id in ("cam-01", "cam-02", "cam-03"):
        row = db.get(Camera, cam_id)
        if row is not None:
            row.enabled = False


def seed_har_cameras_if_missing(db: Session) -> None:
    """Add Avance 4 HAR mock cameras when absent (safe on existing DBs)."""
    from vision_ops_alerting.services.mock_videos import assign_videos_to_har_cameras, public_url_for_video

    har_defaults = [
        ("cam-har-01", "HAR — DINOv2 puro", "dinov2_puro", 10),
        ("cam-har-02", "HAR — DINO→MC-JEPA", "dinov2_mcjepa", 11),
        ("cam-har-03", "HAR — V-JEPA2 puro", "vjepa2_puro", 12),
        ("cam-har-04", "HAR — V-JEPA MC frozen", "vjepa2_mcjepa_frozen", 13),
        ("cam-har-05", "HAR — V-JEPA MC finetune", "vjepa2_mcjepa_partial", 14),
    ]
    video_map = assign_videos_to_har_cameras()
    for cam_id, name, model, order in har_defaults:
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
        if row is not None:
            if mock_cfg and not row.config_json:
                row.config_json = mock_cfg
                row.location = "HAR Lab / Mock video"
            continue
        image_url = public_url_for_video(mock_path) if mock_path else ""
        db.add(
            Camera(
                id=cam_id,
                name=name,
                location="HAR Lab / Mock video",
                zone="HAR",
                source_type="mock_video",
                coords="42.3605°N · 71.0592°W",
                inference_model=model,
                inference_task="activity",
                image_url=image_url,
                backend_camera_id=cam_id,
                status="live",
                sort_order=order,
                config_json=mock_cfg,
            )
        )

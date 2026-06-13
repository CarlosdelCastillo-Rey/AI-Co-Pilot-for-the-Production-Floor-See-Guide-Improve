from __future__ import annotations

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import PlantConfig

DEFAULT_PLANT: dict[str, float | int | str] = {
    "site_name": "VisionOps Plant",
    "line_cost_per_minute": 125.0,
    "material_cost_per_unit": 18.5,
    "target_cycle_sec": 45.0,
    "shift_hours": 8.0,
    "uptime_critical_penalty": 2.5,
    "uptime_warning_penalty": 0.8,
    "uptime_floor_pct": 85.0,
    "uptime_ceiling_pct": 99.9,
    "performance_floor_pct": 50.0,
    "performance_ceiling_pct": 100.0,
    "quality_floor_pct": 70.0,
    "quality_ceiling_pct": 100.0,
    "default_clip_duration_sec": 60,
    "downtime_critical_threshold_pct": 0.7,
    "inference_base_per_camera": 400,
    "inference_probe_bonus": 200,
    "inference_event_multiplier": 12,
    "inference_min_per_camera": 300,
}

KPI_DEFINITIONS: list[dict] = [
    {
        "id": "cameras_online",
        "label": "Cameras Online",
        "formula": "live_cameras / total_enabled_cameras",
        "description": "Count of cameras reporting live status vs total enabled cameras in the registry.",
        "settingsKeys": [],
    },
    {
        "id": "inferences_per_min",
        "label": "Inferences / min",
        "formula": "max(min_per_camera × online, base × online + probe_bonus × probes + event_multiplier × events_today)",
        "description": "Estimated vision throughput based on online cameras, active probes, and today's event volume.",
        "settingsKeys": [
            "inferenceBasePerCamera",
            "inferenceProbeBonus",
            "inferenceEventMultiplier",
            "inferenceMinPerCamera",
        ],
    },
    {
        "id": "events_today",
        "label": "Events Today",
        "formula": "count(events where occurred_at is today)",
        "description": "All alert events logged today (any severity). Delta compares today vs 7-day daily average.",
        "settingsKeys": [],
    },
    {
        "id": "avg_edge_latency",
        "label": "Avg Edge Latency",
        "formula": "avg(health_metric_samples.latency_ms for vision_backend)",
        "description": "Mean backend probe latency from health samples. Delta vs previous 5-sample window.",
        "settingsKeys": [],
    },
    {
        "id": "uptime",
        "label": "Uptime Rate",
        "formula": "analytics_daily → health metrics → camera ratio → 99.9 − (critical×penalty_c + warning×penalty_w)",
        "description": "Line availability estimate. Uses stored daily value first, then health telemetry, then live camera ratio, then event-penalty fallback.",
        "settingsKeys": ["uptimeCriticalPenalty", "uptimeWarningPenalty", "uptimeFloorPct", "uptimeCeilingPct"],
    },
    {
        "id": "oee",
        "label": "OEE Composite",
        "formula": "(Availability / 100) × (Performance / 100) × (Quality / 100) × 100",
        "description": "Overall Equipment Effectiveness for the selected shift and camera scope.",
        "settingsKeys": ["shiftHours", "targetCycleSec"],
    },
    {
        "id": "oee_availability",
        "label": "OEE Availability",
        "formula": "uptime − (resolved_downtime_seconds / 60 / shift_hours)",
        "description": "Running time adjusted by supervisor-logged downtime from resolved timeline incidents.",
        "settingsKeys": ["shiftHours"],
    },
    {
        "id": "oee_performance",
        "label": "OEE Performance",
        "formula": "100 − (critical+warning events / (total events + 10)) × 100, clamped",
        "description": "Cycle-pace proxy from alert frequency vs baseline event volume.",
        "settingsKeys": ["performanceFloorPct", "performanceCeilingPct", "targetCycleSec"],
    },
    {
        "id": "oee_quality",
        "label": "OEE Quality",
        "formula": "100 − (defect case types / total events) × 100, clamped",
        "description": "Quality proxy from vision-detected operator, zone, and defect case types.",
        "settingsKeys": ["qualityFloorPct", "qualityCeilingPct"],
    },
    {
        "id": "coq",
        "label": "Cost of Quality",
        "formula": "(downtime_min × line_cost_per_min) + (scrap_units × material_cost_per_unit)",
        "description": "Financial impact from supervisor-logged downtime and scrap on resolved timeline incidents.",
        "settingsKeys": ["lineCostPerMinute", "materialCostPerUnit"],
    },
    {
        "id": "flow_efficiency",
        "label": "Flow Efficiency",
        "formula": "(performance + quality) / 2",
        "description": "Material and operator flow proxy derived from OEE performance and quality factors.",
        "settingsKeys": ["targetCycleSec"],
    },
    {
        "id": "flow_efficiency_trend",
        "label": "Flow Efficiency Trend",
        "formula": "today_flow − yesterday_flow",
        "description": "Day-over-day change in computed flow efficiency.",
        "settingsKeys": [],
    },
    {
        "id": "downtime_by_station",
        "label": "Est. Downtime per Station",
        "formula": "sum(clip_duration_sec or default_clip_duration) per camera/line for critical+warning events",
        "description": "Estimated minutes tied to recent alert clip durations, grouped by camera or line.",
        "settingsKeys": ["defaultClipDurationSec", "downtimeCriticalThresholdPct"],
    },
    {
        "id": "pareto",
        "label": "Pareto Root Causes",
        "formula": "count(events by industrial_reason_code) / total tagged closures",
        "description": "Share of resolved incidents by standardized reason code logged on Timeline.",
        "settingsKeys": [],
    },
    {
        "id": "open_critical",
        "label": "Open Critical Queue",
        "formula": "count(events where resolution_status=OPEN and severity=critical)",
        "description": "Incidents awaiting supervisor triage that still block the line.",
        "settingsKeys": [],
    },
    {
        "id": "heatmap_anomalies",
        "label": "Heatmap Anomalies",
        "formula": "stored grid anomalies OR count(critical+warning events for camera today)",
        "description": "Anomaly count from vision heatmap data or live event fallback for the selected camera.",
        "settingsKeys": [],
    },
    {
        "id": "incident_count",
        "label": "Incident Count",
        "formula": "count(critical events today)",
        "description": "Critical-severity events for the current calendar day.",
        "settingsKeys": [],
    },
    {
        "id": "avg_ack_time",
        "label": "Avg Time to Acknowledge",
        "formula": "avg(acknowledged_at − occurred_at)",
        "description": "Mean supervisor response time for acknowledged incidents today.",
        "settingsKeys": [],
    },
]


def get_plant_config(db: Session) -> PlantConfig:
    row = db.get(PlantConfig, "default")
    if row:
        return row
    row = PlantConfig(id="default", site_name="VisionOps Plant")
    db.add(row)
    db.flush()
    return row


def plant_to_dict(config: PlantConfig) -> dict:
    return {
        "siteName": config.site_name or "VisionOps Plant",
        "lineCostPerMinute": config.line_cost_per_minute,
        "materialCostPerUnit": config.material_cost_per_unit,
        "targetCycleSec": config.target_cycle_sec,
        "shiftHours": config.shift_hours,
        "uptimeCriticalPenalty": config.uptime_critical_penalty,
        "uptimeWarningPenalty": config.uptime_warning_penalty,
        "uptimeFloorPct": config.uptime_floor_pct,
        "uptimeCeilingPct": config.uptime_ceiling_pct,
        "performanceFloorPct": config.performance_floor_pct,
        "performanceCeilingPct": config.performance_ceiling_pct,
        "qualityFloorPct": config.quality_floor_pct,
        "qualityCeilingPct": config.quality_ceiling_pct,
        "defaultClipDurationSec": config.default_clip_duration_sec,
        "downtimeCriticalThresholdPct": config.downtime_critical_threshold_pct,
        "inferenceBasePerCamera": config.inference_base_per_camera,
        "inferenceProbeBonus": config.inference_probe_bonus,
        "inferenceEventMultiplier": config.inference_event_multiplier,
        "inferenceMinPerCamera": config.inference_min_per_camera,
        "updatedAt": config.updated_at.isoformat() if config.updated_at else None,
        "updatedBy": config.updated_by,
    }


def update_plant_config(db: Session, data: dict, *, updated_by: str | None = None) -> PlantConfig:
    config = get_plant_config(db)
    field_map = {
        "siteName": "site_name",
        "lineCostPerMinute": "line_cost_per_minute",
        "materialCostPerUnit": "material_cost_per_unit",
        "targetCycleSec": "target_cycle_sec",
        "shiftHours": "shift_hours",
        "uptimeCriticalPenalty": "uptime_critical_penalty",
        "uptimeWarningPenalty": "uptime_warning_penalty",
        "uptimeFloorPct": "uptime_floor_pct",
        "uptimeCeilingPct": "uptime_ceiling_pct",
        "performanceFloorPct": "performance_floor_pct",
        "performanceCeilingPct": "performance_ceiling_pct",
        "qualityFloorPct": "quality_floor_pct",
        "qualityCeilingPct": "quality_ceiling_pct",
        "defaultClipDurationSec": "default_clip_duration_sec",
        "downtimeCriticalThresholdPct": "downtime_critical_threshold_pct",
        "inferenceBasePerCamera": "inference_base_per_camera",
        "inferenceProbeBonus": "inference_probe_bonus",
        "inferenceEventMultiplier": "inference_event_multiplier",
        "inferenceMinPerCamera": "inference_min_per_camera",
    }
    for api_key, col in field_map.items():
        if api_key in data and data[api_key] is not None:
            setattr(config, col, data[api_key])
    if updated_by:
        config.updated_by = updated_by
    db.flush()
    return config

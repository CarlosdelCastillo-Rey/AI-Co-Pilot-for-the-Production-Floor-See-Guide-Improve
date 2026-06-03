"""Purge runtime alerting data while keeping cameras, users, and configuration."""

from __future__ import annotations

from sqlalchemy.orm import Session

from vision_ops_alerting.db.models import (
    AlertDelivery,
    AnalyticsDaily,
    AnalyticsHeatmap,
    Event,
    HarActivityLog,
    HarInferenceResult,
    HarInferenceRun,
    HarWatchSession,
    HealthMetricSample,
)


def purge_dynamic_alerting_data(db: Session) -> dict[str, int]:
    """Delete events, deliveries, analytics rollups, HAR logs, and telemetry samples."""
    counts: dict[str, int] = {}

    # FK order: HAR logs → sessions → deliveries → events → inference → analytics → health
    counts["har_activity_logs"] = db.query(HarActivityLog).delete(synchronize_session=False)
    counts["har_watch_sessions"] = db.query(HarWatchSession).delete(synchronize_session=False)
    counts["alert_deliveries"] = db.query(AlertDelivery).delete(synchronize_session=False)
    counts["events"] = db.query(Event).delete(synchronize_session=False)
    counts["har_inference_results"] = db.query(HarInferenceResult).delete(synchronize_session=False)
    counts["har_inference_runs"] = db.query(HarInferenceRun).delete(synchronize_session=False)
    counts["analytics_heatmaps"] = db.query(AnalyticsHeatmap).delete(synchronize_session=False)
    counts["analytics_daily"] = db.query(AnalyticsDaily).delete(synchronize_session=False)
    counts["health_metric_samples"] = db.query(HealthMetricSample).delete(synchronize_session=False)

    db.commit()
    return counts

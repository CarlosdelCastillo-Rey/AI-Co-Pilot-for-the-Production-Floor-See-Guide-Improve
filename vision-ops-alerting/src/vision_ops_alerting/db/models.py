from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(64), default="Supervisor")
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    icon: Mapped[str] = mapped_column(String(64), default="warning")
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    zone: Mapped[str] = mapped_column(String(128))
    case_type: Mapped[str] = mapped_column(String(64), default="unknown")
    severity: Mapped[str] = mapped_column(String(16))  # CRITICAL | WARNING | DISABLED
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_email: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_telegram: Mapped[bool] = mapped_column(Boolean, default=False)
    email_template_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("email_templates.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    events: Mapped[list[Event]] = relationship(back_populates="rule")


class EmailTemplateRecord(Base):
    __tablename__ = "email_templates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    slug: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(256))
    case_type: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(128))
    severity_level: Mapped[str] = mapped_column(String(16))  # warning | critical | info
    headline: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(String(512))
    footer_reason: Mapped[str] = mapped_column(String(128))
    layout: Mapped[str] = mapped_column(String(32), default="standard")
    snapshot_url: Mapped[str | None] = mapped_column(Text)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rule_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("alert_rules.id"), nullable=True)
    site_id: Mapped[str | None] = mapped_column(String(64))
    line_id: Mapped[str | None] = mapped_column(String(64))
    camera_id: Mapped[str | None] = mapped_column(String(64))
    case_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16))  # critical | warning | info | normal
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    actor_type: Mapped[str | None] = mapped_column(String(32))
    actor_track_id: Mapped[str | None] = mapped_column(String(64))
    actor_name: Mapped[str | None] = mapped_column(String(128))
    evidence_json: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[str | None] = mapped_column(Text)
    meta_json: Mapped[str | None] = mapped_column(Text)
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    clip_url: Mapped[str | None] = mapped_column(Text)
    clip_duration_sec: Mapped[int | None] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_status: Mapped[str] = mapped_column(String(32), default="OPEN", index=True)
    downtime_caused_seconds: Mapped[int] = mapped_column(Integer, default=0)
    scrap_caused_units: Mapped[int] = mapped_column(Integer, default=0)
    closure_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    industrial_reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    hidden_from_panel: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    rule: Mapped[AlertRule | None] = relationship(back_populates="events")
    deliveries: Mapped[list[AlertDelivery]] = relationship(back_populates="event")


class AlertDelivery(Base):
    __tablename__ = "alert_deliveries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), ForeignKey("events.id"), index=True)
    channel: Mapped[str] = mapped_column(String(32), default="email")
    template_id: Mapped[str] = mapped_column(String(64))
    to_emails_json: Mapped[str] = mapped_column(Text)
    message_ids_json: Mapped[str | None] = mapped_column(Text)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16))  # sent | failed | dry_run
    error_message: Mapped[str | None] = mapped_column(Text)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    event: Mapped[Event] = relationship(back_populates="deliveries")


class AnalyticsHeatmap(Base):
    __tablename__ = "analytics_heatmaps"
    __table_args__ = (UniqueConstraint("camera_id", "event_date", "shift", name="uq_heatmap_camera_date_shift"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    event_date: Mapped[str] = mapped_column(String(16), index=True)
    shift: Mapped[str] = mapped_column(String(16))
    grid_json: Mapped[str] = mapped_column(Text)
    anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    sensors_active: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalyticsDaily(Base):
    __tablename__ = "analytics_daily"
    __table_args__ = (UniqueConstraint("event_date", "shift", "camera_id", name="uq_daily_date_shift_camera"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_date: Mapped[str] = mapped_column(String(16), index=True)
    shift: Mapped[str] = mapped_column(String(16))
    camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    incident_count: Mapped[int] = mapped_column(Integer, default=0)
    uptime_pct: Mapped[float] = mapped_column(Float, default=94.0)
    flow_efficiency_pct: Mapped[float] = mapped_column(Float, default=94.0)
    availability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    performance_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    oee_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    downtime_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    scrap_units: Mapped[int | None] = mapped_column(Integer, nullable=True)
    coq_total_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IndustrialReasonCode(Base):
    __tablename__ = "industrial_reason_codes"

    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(256))
    category: Mapped[str] = mapped_column(String(128))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class PlantConfig(Base):
    __tablename__ = "plant_config"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default="default")
    site_name: Mapped[str] = mapped_column(String(256), default="VisionOps Plant")
    line_cost_per_minute: Mapped[float] = mapped_column(Float, default=125.0)
    material_cost_per_unit: Mapped[float] = mapped_column(Float, default=18.5)
    target_cycle_sec: Mapped[float] = mapped_column(Float, default=45.0)
    shift_hours: Mapped[float] = mapped_column(Float, default=8.0)
    uptime_critical_penalty: Mapped[float] = mapped_column(Float, default=2.5)
    uptime_warning_penalty: Mapped[float] = mapped_column(Float, default=0.8)
    uptime_floor_pct: Mapped[float] = mapped_column(Float, default=85.0)
    uptime_ceiling_pct: Mapped[float] = mapped_column(Float, default=99.9)
    performance_floor_pct: Mapped[float] = mapped_column(Float, default=50.0)
    performance_ceiling_pct: Mapped[float] = mapped_column(Float, default=100.0)
    quality_floor_pct: Mapped[float] = mapped_column(Float, default=70.0)
    quality_ceiling_pct: Mapped[float] = mapped_column(Float, default=100.0)
    default_clip_duration_sec: Mapped[int] = mapped_column(Integer, default=60)
    downtime_critical_threshold_pct: Mapped[float] = mapped_column(Float, default=0.7)
    inference_base_per_camera: Mapped[int] = mapped_column(Integer, default=400)
    inference_probe_bonus: Mapped[int] = mapped_column(Integer, default=200)
    inference_event_multiplier: Mapped[int] = mapped_column(Integer, default=12)
    inference_min_per_camera: Mapped[int] = mapped_column(Integer, default=300)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(256))
    zone: Mapped[str | None] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(String(16), default="rtsp")  # rtsp | onvif | webcam
    stream_url: Mapped[str | None] = mapped_column(Text)
    coords: Mapped[str | None] = mapped_column(String(128))
    inference_model: Mapped[str | None] = mapped_column(String(64))  # dinov3 | vjepa2 | yolov8 | none
    inference_task: Mapped[str | None] = mapped_column(String(128))
    image_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="offline")  # live | offline
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    backend_camera_id: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class HealthMetricSample(Base):
    __tablename__ = "health_metric_samples"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    value_pct: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[str | None] = mapped_column(String(256))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class HarInferenceRun(Base):
    """One HAR probe batch or single-model run on a shared clip (for historics / watch data)."""

    __tablename__ = "har_inference_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_type: Mapped[str] = mapped_column(String(16), default="batch")  # batch | single
    clip_source: Mapped[str] = mapped_column(String(512))
    clip_path: Mapped[str | None] = mapped_column(Text)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    shared_clip: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | partial | error
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    meta_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    results: Mapped[list["HarInferenceResult"]] = relationship(
        "HarInferenceResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class HarInferenceResult(Base):
    """Per-model activity prediction within a HAR run."""

    __tablename__ = "har_inference_results"
    __table_args__ = (UniqueConstraint("run_id", "model_id", name="uq_har_result_run_model"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("har_inference_runs.id", ondelete="CASCADE"),
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    predicted_label: Mapped[str | None] = mapped_column(String(256))
    class_index: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float, index=True)
    backend: Mapped[str | None] = mapped_column(String(256))
    device: Mapped[str | None] = mapped_column(String(32))
    top_k_json: Mapped[str | None] = mapped_column(Text)
    overlay_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error_message: Mapped[str | None] = mapped_column(Text)
    probed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    run: Mapped["HarInferenceRun"] = relationship("HarInferenceRun", back_populates="results")


HAR_PRIMARY_ACTION_LABEL = "Assemble system"


class HarWatchSession(Base):
    """Groups live HAR logs per camera + video loop / watch period."""

    __tablename__ = "har_watch_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    video_name: Mapped[str | None] = mapped_column(String(512))
    clip_url: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_label: Mapped[str | None] = mapped_column(String(128))
    hyperparams_json: Mapped[str | None] = mapped_column(Text)
    meta_json: Mapped[str | None] = mapped_column(Text)

    logs: Mapped[list["HarActivityLog"]] = relationship(
        "HarActivityLog",
        back_populates="session",
    )


class HarActivityLog(Base):
    """Append-only integral HAR activity log per camera (live + probe)."""

    __tablename__ = "har_activity_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("har_watch_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(16), default="live")  # live | probe
    frame_index: Mapped[int | None] = mapped_column(Integer)
    video_offset_sec: Mapped[float | None] = mapped_column(Float)
    predicted_label: Mapped[str | None] = mapped_column(String(256), index=True)
    class_index: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    top_k_json: Mapped[str | None] = mapped_column(Text)
    is_primary_action: Mapped[bool] = mapped_column(Boolean, default=False)
    person_count: Mapped[int] = mapped_column(Integer, default=0)
    detections_json: Mapped[str | None] = mapped_column(Text)
    actor_type: Mapped[str | None] = mapped_column(String(32))
    actor_track_id: Mapped[str | None] = mapped_column(String(64))
    actor_name: Mapped[str | None] = mapped_column(String(128))
    backend: Mapped[str | None] = mapped_column(String(256))
    device: Mapped[str | None] = mapped_column(String(32))
    infer_ms: Mapped[float | None] = mapped_column(Float)
    snapshot_url: Mapped[str | None] = mapped_column(Text)
    video_name: Mapped[str | None] = mapped_column(String(512), index=True)
    clip_url: Mapped[str | None] = mapped_column(Text)
    model_label: Mapped[str | None] = mapped_column(String(128))
    hyperparams_json: Mapped[str | None] = mapped_column(Text)
    promoted_to_event_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("events.id", ondelete="SET NULL"),
        nullable=True,
    )

    session: Mapped["HarWatchSession | None"] = relationship("HarWatchSession", back_populates="logs")

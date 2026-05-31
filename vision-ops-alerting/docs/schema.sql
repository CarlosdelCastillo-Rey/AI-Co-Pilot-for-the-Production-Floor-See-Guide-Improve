-- VisionOps local SQLite schema (reference)
-- Auto-created by SQLAlchemy on startup; see db/models.py

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS alert_rules (
    id              TEXT PRIMARY KEY,
    icon            TEXT NOT NULL DEFAULT 'warning',
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    zone            TEXT NOT NULL,
    case_type       TEXT NOT NULL DEFAULT 'unknown',
    severity        TEXT NOT NULL CHECK (severity IN ('CRITICAL', 'WARNING', 'DISABLED')),
    enabled         INTEGER NOT NULL DEFAULT 1,
    notify_email    INTEGER NOT NULL DEFAULT 1,
    notify_telegram INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id                TEXT PRIMARY KEY,
    rule_id           TEXT REFERENCES alert_rules(id),
    site_id           TEXT,
    line_id           TEXT,
    camera_id         TEXT,
    case_type         TEXT NOT NULL,
    severity          TEXT NOT NULL,
    title             TEXT NOT NULL,
    description       TEXT NOT NULL,
    actor_type        TEXT,
    actor_track_id    TEXT,
    actor_name        TEXT,
    evidence_json     TEXT,
    context_json      TEXT,
    meta_json         TEXT,
    thumbnail_url     TEXT,
    clip_url          TEXT,
    clip_duration_sec INTEGER,
    occurred_at       TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_occurred_at ON events(occurred_at);
CREATE INDEX IF NOT EXISTS ix_events_camera_id ON events(camera_id);
CREATE INDEX IF NOT EXISTS ix_events_severity ON events(severity);

CREATE TABLE IF NOT EXISTS alert_deliveries (
    id               TEXT PRIMARY KEY,
    event_id         TEXT NOT NULL REFERENCES events(id),
    channel          TEXT NOT NULL DEFAULT 'email',
    template_id      TEXT NOT NULL,
    to_emails_json   TEXT NOT NULL,
    message_ids_json TEXT,
    dry_run          INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,
    error_message    TEXT,
    sent_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alert_deliveries_event_id ON alert_deliveries(event_id);
CREATE INDEX IF NOT EXISTS ix_alert_deliveries_sent_at ON alert_deliveries(sent_at);

CREATE TABLE IF NOT EXISTS analytics_heatmaps (
    id              TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    shift           TEXT NOT NULL,
    grid_json       TEXT NOT NULL,
    anomaly_count   INTEGER NOT NULL DEFAULT 0,
    sensors_active  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    UNIQUE (camera_id, event_date, shift)
);

CREATE TABLE IF NOT EXISTS analytics_daily (
    id                   TEXT PRIMARY KEY,
    event_date           TEXT NOT NULL,
    shift                TEXT NOT NULL,
    camera_id            TEXT,
    incident_count       INTEGER NOT NULL DEFAULT 0,
    uptime_pct           REAL NOT NULL DEFAULT 94.0,
    flow_efficiency_pct  REAL NOT NULL DEFAULT 94.0,
    created_at           TEXT NOT NULL,
    UNIQUE (event_date, shift, camera_id)
);

CREATE TABLE IF NOT EXISTS health_metric_samples (
    id           TEXT PRIMARY KEY,
    service      TEXT NOT NULL,
    status       TEXT NOT NULL,
    latency_ms   REAL,
    value_pct    REAL NOT NULL DEFAULT 0,
    detail       TEXT,
    recorded_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_health_metric_service ON health_metric_samples(service);
CREATE INDEX IF NOT EXISTS ix_health_metric_recorded ON health_metric_samples(recorded_at);

CREATE TABLE IF NOT EXISTS cameras (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    location            TEXT NOT NULL,
    zone                TEXT,
    source_type         TEXT NOT NULL DEFAULT 'rtsp',
    stream_url          TEXT,
    coords              TEXT,
    inference_model     TEXT,
    inference_task      TEXT,
    image_url           TEXT,
    status              TEXT NOT NULL DEFAULT 'offline',
    enabled             INTEGER NOT NULL DEFAULT 1,
    sort_order          INTEGER NOT NULL DEFAULT 0,
    backend_camera_id   TEXT,
    config_json         TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_cameras_sort_order ON cameras(sort_order);

CREATE TABLE IF NOT EXISTS email_templates (
    id              TEXT PRIMARY KEY,
    slug            TEXT NOT NULL,
    name            TEXT NOT NULL,
    case_type       TEXT NOT NULL,
    category        TEXT NOT NULL,
    severity_level  TEXT NOT NULL,
    headline        TEXT NOT NULL,
    body            TEXT NOT NULL,
    subject         TEXT NOT NULL,
    footer_reason   TEXT NOT NULL,
    layout          TEXT NOT NULL DEFAULT 'standard',
    snapshot_url    TEXT,
    is_builtin      INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_email_templates_case_type ON email_templates(case_type);

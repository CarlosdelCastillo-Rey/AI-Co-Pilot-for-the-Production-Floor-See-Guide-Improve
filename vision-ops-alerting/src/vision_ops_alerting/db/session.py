from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from vision_ops_alerting.config import settings
from vision_ops_alerting.db.models import Base


def _ensure_sqlite_dir(url: str) -> None:
    if url.startswith("sqlite:///"):
        path = Path(url.replace("sqlite:///", ""))
        path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.database_url)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    if settings.database_url.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _migrate_sqlite_schema() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    from sqlalchemy import text

    with engine.connect() as conn:
        rule_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alert_rules)"))}
        if "email_template_id" not in rule_cols:
            conn.execute(text("ALTER TABLE alert_rules ADD COLUMN email_template_id TEXT"))
        if "updated_by" not in rule_cols:
            conn.execute(text("ALTER TABLE alert_rules ADD COLUMN updated_by TEXT"))

        event_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(events)"))}
        event_migrations = [
            ("acknowledged_at", "TEXT"),
            ("acknowledged_by", "TEXT"),
            ("resolution_status", "TEXT DEFAULT 'OPEN'"),
            ("downtime_caused_seconds", "INTEGER DEFAULT 0"),
            ("scrap_caused_units", "INTEGER DEFAULT 0"),
            ("closure_notes", "TEXT"),
            ("industrial_reason_code", "TEXT"),
            ("resolved_at", "TEXT"),
            ("resolved_by", "TEXT"),
            ("hidden_from_panel", "INTEGER DEFAULT 0"),
        ]
        for col, col_type in event_migrations:
            if col not in event_cols:
                conn.execute(text(f"ALTER TABLE events ADD COLUMN {col} {col_type}"))

        daily_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(analytics_daily)"))}
        daily_migrations = [
            ("availability_pct", "REAL"),
            ("performance_pct", "REAL"),
            ("quality_pct", "REAL"),
            ("oee_pct", "REAL"),
            ("downtime_minutes", "REAL"),
            ("scrap_units", "INTEGER"),
            ("coq_total_usd", "REAL"),
        ]
        for col, col_type in daily_migrations:
            if col not in daily_cols:
                conn.execute(text(f"ALTER TABLE analytics_daily ADD COLUMN {col} {col_type}"))

        plant_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(plant_config)"))}
        plant_migrations = [
            ("site_name", "TEXT DEFAULT 'VisionOps Plant'"),
            ("shift_hours", "REAL DEFAULT 8.0"),
            ("uptime_critical_penalty", "REAL DEFAULT 2.5"),
            ("uptime_warning_penalty", "REAL DEFAULT 0.8"),
            ("uptime_floor_pct", "REAL DEFAULT 85.0"),
            ("uptime_ceiling_pct", "REAL DEFAULT 99.9"),
            ("performance_floor_pct", "REAL DEFAULT 50.0"),
            ("performance_ceiling_pct", "REAL DEFAULT 100.0"),
            ("quality_floor_pct", "REAL DEFAULT 70.0"),
            ("quality_ceiling_pct", "REAL DEFAULT 100.0"),
            ("default_clip_duration_sec", "INTEGER DEFAULT 60"),
            ("downtime_critical_threshold_pct", "REAL DEFAULT 0.7"),
            ("inference_base_per_camera", "INTEGER DEFAULT 400"),
            ("inference_probe_bonus", "INTEGER DEFAULT 200"),
            ("inference_event_multiplier", "INTEGER DEFAULT 12"),
            ("inference_min_per_camera", "INTEGER DEFAULT 300"),
        ]
        for col, col_type in plant_migrations:
            if col not in plant_cols:
                conn.execute(text(f"ALTER TABLE plant_config ADD COLUMN {col} {col_type}"))
        if "updated_by" not in plant_cols:
            conn.execute(text("ALTER TABLE plant_config ADD COLUMN updated_by TEXT"))

        user_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(users)"))}
        if user_cols and "role" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'Supervisor'"))
            conn.execute(
                text(
                    "UPDATE users SET role = 'Ops Lead' WHERE email = 'admin@visionops.local'"
                )
            )

        conn.commit()

        # Backfill lifecycle defaults on existing rows
        conn.execute(
            text("UPDATE events SET resolution_status = 'OPEN' WHERE resolution_status IS NULL")
        )
        conn.execute(
            text("UPDATE events SET downtime_caused_seconds = 0 WHERE downtime_caused_seconds IS NULL")
        )
        conn.execute(
            text("UPDATE events SET scrap_caused_units = 0 WHERE scrap_caused_units IS NULL")
        )
        conn.execute(
            text(
                "UPDATE events SET hidden_from_panel = 0 WHERE hidden_from_panel IS NULL"
            )
        )
        conn.commit()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_schema()
    from vision_ops_alerting.services.har_activity_store import prune_old_logs

    with SessionLocal() as db:
        prune_old_logs(db)
    from vision_ops_alerting.db.seed import (
        disable_legacy_cameras,
        seed_cameras_if_empty,
        seed_har_cameras_if_missing,
        seed_if_empty,
    )
    from vision_ops_alerting.services.email_templates import ensure_builtin_templates
    from vision_ops_alerting.services.industrial_seed import ensure_industrial_defaults
    from vision_ops_alerting.services.rule_dispatch import ensure_default_action_rules
    from vision_ops_alerting.services.users import ensure_default_admin

    with SessionLocal() as db:
        seed_cameras_if_empty(db)
        disable_legacy_cameras(db)
        seed_har_cameras_if_missing(db)
        ensure_builtin_templates(db)
        ensure_default_action_rules(db)
        ensure_industrial_defaults(db)
        ensure_default_admin(db)
        db.commit()
    if not settings.seed_db:
        return
    with SessionLocal() as db:
        seed_if_empty(db)
        db.commit()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

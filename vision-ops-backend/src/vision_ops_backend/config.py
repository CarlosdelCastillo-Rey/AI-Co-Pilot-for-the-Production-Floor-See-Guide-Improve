"""Unified VisionOps application settings (vision + ops)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# vision-ops-backend/ project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_database_url() -> str:
    db_path = PROJECT_ROOT / "data" / "vision_ops.db"
    return f"sqlite:///{db_path}"


class _VisionSettings(BaseSettings):
    """Vision / inference layer (unprefixed env vars)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    webcam_enabled: bool = False
    camera_index: int = 0
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    mjpeg_fps: int = 12
    public_api_base: str = "http://localhost:8000"
    camera_id: str = "webcam-0"

    face_enabled: bool = False
    owner_name: str = "You"
    face_match_threshold: float = 0.5
    face_detect_every_n_frames: int = 3

    vision_enabled: bool = True

    har_enabled: bool = True
    har_checkpoint_dir: str = "har-research/checkpoints"
    har_shared_clip_path: str = ""
    har_persist_enabled: bool = True
    har_activity_ingest_enabled: bool = True
    har_auto_probe_on_start: bool = False
    har_live_enabled: bool = True
    har_live_infer_every: int = 16
    har_live_buffer_frames: int = 32
    har_live_stream_fps: int = 12
    har_live_show_heatmap: bool = True
    har_live_show_yolo_boxes: bool = True
    har_live_show_robo_hormiga: bool = False
    har_live_per_person_mode: bool = True
    har_exclude_labels: str = ""
    har_v2_session_enabled: bool = True
    har_v2_min_confidence: float = 0.25


class _OpsSettings(BaseSettings):
    """Operational layer (SQLite, alerts, advisor) — VISIONOPS_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONOPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dry_run: bool = True
    mailersend_api_token: str = ""
    from_email: str = ""
    from_name: str = "VisionOps"
    to_email: str = ""
    to_name: str = "Recipient"
    telegram_bot_token: str = ""
    telegram_chat_ids: str = ""
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    advisor_temperature: float = 0.92
    camera_advisor_temperature: float = 0.95
    database_url: str = default_database_url()
    seed_db: bool = False
    alert_snapshots_enabled: bool = True
    auth_secret: str = "visionops-dev-secret-change-me"
    auth_token_hours: int = 72
    seed_admin_email: str = "admin@visionops.local"
    seed_admin_password: str = "admin123"
    seed_admin_name: str = "Plant Supervisor"
    seed_admin_role: str = "Ops Lead"
    har_log_retention_days: int = 7
    har_primary_action_label: str = "Assemble system"
    har_promote_non_assembly: bool = True
    har_low_confidence_threshold: float = 0.15
    har_promote_cooldown_sec: int = 300
    har_email_enabled: bool = False
    har_notify_enabled: bool = True
    har_notify_confidence_threshold: float = 0.3
    har_notify_cooldown_sec: int = 10
    har_ingest_heartbeat_sec: int = 60
    har_ingest_confidence_delta: float = 0.05
    har_session_artifacts_dir: str = ""
    har_reid_auto_link: bool = True
    har_reid_match_threshold: float = 0.75


class Settings(_VisionSettings, _OpsSettings):
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def har_exclude_label_list(self) -> list[str]:
        return [label.strip() for label in self.har_exclude_labels.split(",") if label.strip()]

    @property
    def public_base_url(self) -> str:
        return self.public_api_base.rstrip("/")

    @property
    def session_artifacts_path(self) -> Path:
        if self.har_session_artifacts_dir.strip():
            return Path(self.har_session_artifacts_dir).expanduser().resolve()
        return PROJECT_ROOT / "data" / "har_sessions"

    @property
    def to_emails(self) -> list[str]:
        return [e.strip() for e in self.to_email.split(",") if e.strip()]

    def to_recipients(self) -> list[dict[str, str]]:
        return [{"email": email, "name": self.to_name} for email in self.to_emails]

    @property
    def telegram_chat_id_list(self) -> list[str]:
        return [c.strip() for c in self.telegram_chat_ids.split(",") if c.strip()]


settings = Settings()

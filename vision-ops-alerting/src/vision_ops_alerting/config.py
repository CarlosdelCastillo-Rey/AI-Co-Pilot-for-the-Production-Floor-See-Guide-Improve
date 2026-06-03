from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# vision-ops-alerting/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_database_url() -> str:
    db_path = PROJECT_ROOT / "data" / "vision_ops.db"
    return f"sqlite:///{db_path}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALERTING_", env_file=".env", extra="ignore")

    # Service
    cors_origins: str = "http://localhost:3000"
    # Default safe behavior: don't send email unless explicitly enabled.
    dry_run: bool = True

    # MailerSend
    mailersend_api_token: str = ""
    from_email: str = ""
    from_name: str = "VisionOps"
    # Comma-separated list: a@x.com,b@y.com
    to_email: str = ""
    to_name: str = "Recipient"

    # Strands / local model
    ollama_model: str = "llama3.1"
    advisor_temperature: float = 0.75

    # SQLite database (absolute path under vision-ops-alerting/data/)
    database_url: str = default_database_url()

    # If true, insert demo rows when alert_rules is empty (dev/demo only)
    seed_db: bool = False

    # Vision backend for health probes
    vision_backend_url: str = "http://localhost:8000"

    # Simple email/password auth (change secret in production)
    auth_secret: str = "visionops-dev-secret-change-me"
    auth_token_hours: int = 72
    seed_admin_email: str = "admin@visionops.local"
    seed_admin_password: str = "admin123"
    seed_admin_name: str = "Plant Supervisor"
    seed_admin_role: str = "Ops Lead"

    # HAR integral activity logs
    har_log_retention_days: int = 7
    har_primary_action_label: str = "Assemble system"
    har_promote_non_assembly: bool = True
    har_low_confidence_threshold: float = 0.15
    har_promote_cooldown_sec: int = 300
    har_email_enabled: bool = False
    har_ingest_heartbeat_sec: int = 60
    har_ingest_confidence_delta: float = 0.05

    @property
    def to_emails(self) -> list[str]:
        return [e.strip() for e in self.to_email.split(",") if e.strip()]

    def to_recipients(self) -> list[dict[str, str]]:
        return [{"email": email, "name": self.to_name} for email in self.to_emails]


settings = Settings()


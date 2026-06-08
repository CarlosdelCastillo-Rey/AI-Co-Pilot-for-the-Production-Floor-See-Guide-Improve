"""Application settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Physical webcam (MacBook camera). Off by default — mock HAR cameras use files only.
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
    har_checkpoint_dir: str = "notebooks/Avance 4. Modelos alternativos/Checkpoints"
    har_shared_clip_path: str = ""
    har_persist_enabled: bool = True
    har_activity_ingest_enabled: bool = True
    alerting_api_url: str = "http://localhost:8001"
    har_auto_probe_on_start: bool = False
    har_live_enabled: bool = True
    har_live_infer_every: int = 16
    har_live_buffer_frames: int = 32
    har_live_stream_fps: int = 12
    har_live_show_heatmap: bool = True
    har_live_show_yolo_boxes: bool = True
    # Comma-separated InHARD labels to mask at inference (e.g. "Assemble system")
    har_exclude_labels: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def har_exclude_label_list(self) -> list[str]:
        return [label.strip() for label in self.har_exclude_labels.split(",") if label.strip()]


settings = Settings()

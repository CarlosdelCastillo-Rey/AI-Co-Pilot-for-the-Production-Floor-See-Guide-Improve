"""Application settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    camera_index: int = 0
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    mjpeg_fps: int = 12
    public_api_base: str = "http://localhost:8000"
    camera_id: str = "webcam-0"

    face_enabled: bool = True
    owner_name: str = "You"
    face_match_threshold: float = 0.5
    face_detect_every_n_frames: int = 3

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

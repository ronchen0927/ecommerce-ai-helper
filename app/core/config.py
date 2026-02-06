from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Ecommerce Visual Pro"
    debug: bool = False

    # Storage
    storage_type: str = "local"  # "local" or "gcs"
    local_storage_path: str = "storage"
    gcs_bucket_name: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # AI Service endpoints (placeholders for now)
    rmbg_api_url: str = ""
    flux_api_url: str = ""
    iclight_api_url: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

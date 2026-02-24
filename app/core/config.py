"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Ecommerce Visual Pro"
    debug: bool = False

    # Storage
    storage_type: str = "local"  # "local" or "gcs"
    local_storage_path: str = "storage"
    gcs_bucket_name: str = ""
    gcs_project_id: str = ""
    gcs_credentials_path: str = ""  # Path to service account JSON

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_acks_late: bool = True  # Acknowledge after task completes
    celery_worker_prefetch_multiplier: int = 1  # For long-running tasks
    celery_task_time_limit: int = 600  # 10 minutes max
    celery_result_expires: int = 3600  # 1 hour

    # AI Service endpoints
    rmbg_api_url: str = ""
    rmbg_api_key: str = ""
    flux_api_url: str = ""
    flux_api_key: str = ""
    iclight_api_url: str = ""
    iclight_api_key: str = ""
    ai_api_timeout: float = 180.0
    ai_api_max_retries: int = 3

    # Authentication
    auth_enabled: bool = False
    api_keys: str = ""  # Comma-separated list of valid API keys
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate Limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100  # Requests per window
    rate_limit_window_seconds: int = 60  # Window size

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_api_keys_list(self) -> list[str]:
        """Parse comma-separated API keys into list."""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

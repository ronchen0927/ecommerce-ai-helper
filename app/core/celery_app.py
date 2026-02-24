"""Celery application configuration for async task processing."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ecommerce_visual_pro",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.image_processing"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="Asia/Taipei",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    # Reliability settings
    task_acks_late=settings.celery_task_acks_late,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_time_limit - 30,  # 30s grace period
    result_expires=settings.celery_result_expires,
    # Retry settings
    task_reject_on_worker_lost=True,
    task_default_retry_delay=30,  # 30 seconds
    task_max_retries=3,
    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_pool_limit=settings.redis_max_connections,
    # Result settings
    result_extended=True,  # Store task args/kwargs in result
)

# SSL configuration for rediss:// URLs (e.g., Upstash)
if settings.celery_broker_url.startswith("rediss://"):
    celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": "none"}

if settings.celery_result_backend.startswith("rediss://"):
    celery_app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": "none"}

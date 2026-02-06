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
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Taipei",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,  # For long-running tasks
)

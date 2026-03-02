import os
from celery import Celery
from src.core.config import settings

# Initialize Celery app with Redis broker and backend
celery_app = Celery(
    "ingestion_orchestrator",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.orchestration.tasks"]
)

# Optional configuration settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=False
)

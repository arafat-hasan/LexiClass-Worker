"""Celery application configuration."""

from pathlib import Path

from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger

from .core.config import get_settings
from .core.logging import setup_logging
from .core.queue_config import QUEUE_CONFIGS, TASK_QUEUES, TASK_ROUTES

# Load settings
settings = get_settings()

# Configure logging
setup_logging()

# Create Celery app
app = Celery("lexiclass_worker")

# Configure Celery
app.conf.update(
    broker_url=settings.celery.broker_url,
    result_backend=settings.celery.result_backend,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.celery.task_time_limit,
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    task_queues=TASK_QUEUES,
    task_routes=TASK_ROUTES,
    # Add task-specific settings from queue configs
    task_annotations={
        f"lexiclass_worker.tasks.{queue.name}_task": {
            "rate_limit": queue.rate_limit,
            "retry_backoff": True,  # Enable exponential backoff
            "retry_backoff_max": queue.retry_policy["interval_max"],
            "retry_jitter": True,  # Add random jitter to retry delays
            "max_retries": queue.retry_policy["max_retries"],
            "retry_delay": queue.retry_policy["interval_start"],
            "retry_kwargs": {"max_delay": queue.retry_policy["interval_max"]},
        } for queue in QUEUE_CONFIGS.values()
    },
    # Add result settings
    task_ignore_result=False,  # We want to store task results
    result_expires=86400,  # Results expire after 24 hours
)

# Create storage directories
storage_base = Path(settings.storage.base_path)
(storage_base / settings.storage.models_dir).mkdir(parents=True, exist_ok=True)
(storage_base / settings.storage.indexes_dir).mkdir(parents=True, exist_ok=True)

# Configure Celery logging
@after_setup_logger.connect
@after_setup_task_logger.connect
def setup_celery_logging(*args, **kwargs):
    """Configure Celery to use our logging setup."""
    setup_logging()

# Auto-discover tasks
app.autodiscover_tasks(["lexiclass_worker.tasks"])
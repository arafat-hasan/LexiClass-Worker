"""Celery application configuration."""

import logging
from pathlib import Path

from celery import Celery
from celery.signals import after_setup_logger, after_setup_task_logger, worker_ready
from dotenv import load_dotenv
import redis

from .core.config import get_settings
from .core.logging import setup_logging
from .core.database import initialize_database
# Import queue configuration from Core - single source of truth
from lexiclass_core.queue_config import QUEUE_CONFIGS, TASK_QUEUES, TASK_ROUTES

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

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


def check_redis_connection(redis_url: str, connection_type: str) -> bool:
    """Check if Redis is accessible.

    Args:
        redis_url: Redis connection URL
        connection_type: Type of connection (broker/result_backend)

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Parse Redis URL and create connection
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=5)
        # Test connection with PING
        response = client.ping()
        if response:
            logger.info(
                f"✓ Redis {connection_type} connection successful",
                extra={
                    "redis_url": redis_url,
                    "connection_type": connection_type,
                    "status": "connected"
                }
            )
            return True
        else:
            logger.error(
                f"✗ Redis {connection_type} connection failed: PING returned False",
                extra={
                    "redis_url": redis_url,
                    "connection_type": connection_type,
                    "status": "failed"
                }
            )
            return False
    except redis.ConnectionError as e:
        logger.error(
            f"✗ Redis {connection_type} connection error: {str(e)}",
            extra={
                "redis_url": redis_url,
                "connection_type": connection_type,
                "status": "connection_error",
                "error": str(e)
            }
        )
        return False
    except Exception as e:
        logger.error(
            f"✗ Redis {connection_type} unexpected error: {str(e)}",
            extra={
                "redis_url": redis_url,
                "connection_type": connection_type,
                "status": "error",
                "error": str(e)
            }
        )
        return False
    finally:
        try:
            client.close()
        except:
            pass


@worker_ready.connect
def check_connections(**kwargs):
    """Check all connections when worker is ready."""
    logger.info("=" * 60)
    logger.info("Checking service connections...")
    logger.info("=" * 60)

    # Initialize database using Core's session factory
    try:
        initialize_database()
        logger.info("✓ Database initialized using Core session factory")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")

    # Check Redis broker connection
    broker_connected = check_redis_connection(settings.celery.broker_url, "broker")

    # Check Redis result backend connection
    result_backend_connected = check_redis_connection(
        settings.celery.result_backend,
        "result_backend"
    )

    # Check if worker uses local filesystem storage (no database)
    logger.info(
        "✓ File storage configured",
        extra={
            "storage_type": "filesystem",
            "base_path": str(settings.storage.base_path),
            "models_dir": settings.storage.models_dir,
            "indexes_dir": settings.storage.indexes_dir,
            "status": "configured"
        }
    )

    # Summary
    logger.info("=" * 60)
    if broker_connected and result_backend_connected:
        logger.info("✓ All connections successful - Worker ready to process tasks")
    else:
        logger.warning("⚠ Some connections failed - Worker may not function properly")
    logger.info("=" * 60)

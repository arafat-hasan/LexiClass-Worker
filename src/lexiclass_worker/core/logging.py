"""Logging configuration for the worker service."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from .config import get_settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def __init__(self):
        super().__init__()
        self.settings = get_settings()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": self.settings.environment,
        }

        # Add extra fields if available
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if available
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            log_data.update({
                "exception_type": exc_type.__name__ if exc_type else None,
                "exception_message": str(exc_value) if exc_value else None,
                "stack_trace": self.formatException(record.exc_info),
            })

        # Add task-specific context if available
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "project_id"):
            log_data["project_id"] = record.project_id

        return json.dumps(log_data)


class TaskContextFilter(logging.Filter):
    """Filter that adds task context to log records."""

    def __init__(self, task_id: Optional[str] = None, project_id: Optional[int] = None):
        super().__init__()
        self.task_id = task_id
        self.project_id = project_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add task context to the log record."""
        if self.task_id:
            record.task_id = self.task_id
        if self.project_id:
            record.project_id = self.project_id
        return True


def setup_logging() -> None:
    """Configure logging based on settings."""
    settings = get_settings()
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=settings.log_level,
        handlers=handlers,
        force=True,  # Override existing configuration
    )

    # Set levels for third-party loggers
    logging.getLogger("celery").setLevel(logging.WARNING)
    logging.getLogger("kombu").setLevel(logging.WARNING)
    logging.getLogger("amqp").setLevel(logging.WARNING)

    # Create logger for our application
    logger = logging.getLogger("lexiclass_worker")
    logger.setLevel(settings.log_level)


def get_task_logger(task_id: str, project_id: Optional[int] = None) -> logging.Logger:
    """Get a logger with task context."""
    logger = logging.getLogger(f"lexiclass_worker.task.{task_id}")
    logger.addFilter(TaskContextFilter(task_id, project_id))
    return logger

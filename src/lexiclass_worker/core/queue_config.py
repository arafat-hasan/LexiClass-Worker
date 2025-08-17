"""Shared queue configuration for LexiClass services."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class QueueName(str, Enum):
    """Task queue names."""

    DEFAULT = "default"
    INDEXING = "indexing"
    TRAINING = "training"
    PREDICTION = "prediction"


class QueueConfig(BaseModel):
    """Queue configuration."""

    name: QueueName
    routing_key: str
    priority: int = 0
    rate_limit: Optional[str] = None  # e.g., "100/s", "1000/m"
    retry_policy: Dict[str, int] = {
        "max_retries": 3,
        "interval_start": 0,
        "interval_step": 60,  # 1 minute
        "interval_max": 3600,  # 1 hour
    }
    queue_arguments: Dict[str, int] = {"x-max-priority": 10}


# Queue configurations
QUEUE_CONFIGS = {
    QueueName.DEFAULT: QueueConfig(
        name=QueueName.DEFAULT,
        routing_key="default",
        priority=0,
    ),
    QueueName.INDEXING: QueueConfig(
        name=QueueName.INDEXING,
        routing_key="indexing",
        priority=1,
        rate_limit="100/m",  # 100 tasks per minute
    ),
    QueueName.TRAINING: QueueConfig(
        name=QueueName.TRAINING,
        routing_key="training",
        priority=2,
        rate_limit="10/m",  # 10 tasks per minute
        retry_policy={
            "max_retries": 2,
            "interval_start": 0,
            "interval_step": 300,  # 5 minutes
            "interval_max": 7200,  # 2 hours
        },
    ),
    QueueName.PREDICTION: QueueConfig(
        name=QueueName.PREDICTION,
        routing_key="prediction",
        priority=3,
        rate_limit="1000/m",  # 1000 tasks per minute
    ),
}

# Task routing patterns
TASK_INDEXING = 'lexiclass_worker.tasks.index_documents_task'
TASK_TRAINING = 'lexiclass_worker.tasks.train_model_task'
TASK_PREDICTION = 'lexiclass_worker.tasks.predict_documents_task'

# Convert queue configs to Celery task_queues format
TASK_QUEUES = {
    queue.name: {
        'routing_key': queue.routing_key,
        'queue_arguments': queue.queue_arguments,
        'rate_limit': queue.rate_limit,
    } for queue in QUEUE_CONFIGS.values()
}

# Task routing configuration
TASK_ROUTES = {
    TASK_INDEXING: {'queue': QueueName.INDEXING},
    TASK_TRAINING: {'queue': QueueName.TRAINING},
    TASK_PREDICTION: {'queue': QueueName.PREDICTION}
}
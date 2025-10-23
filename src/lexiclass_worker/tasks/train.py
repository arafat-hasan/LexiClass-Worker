"""Training task implementation."""

from pathlib import Path
from typing import Dict, List, Optional

from lexiclass.io import DocumentLoader, load_labels
from pydantic import Field

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.exceptions import DocumentError, ModelError


class TrainModelInput(TaskInput):
    """Input schema for model training task."""
    project_id: str = Field(..., description="Unique project identifier")
    labels_path: str = Field(..., description="Path to labels TSV file")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to train on"
    )


class TrainModelOutput(TaskOutput):
    """Output schema for model training task."""
    model_path: Optional[str] = None
    num_labels: Optional[int] = None
    metrics: Optional[Dict] = None


class TrainModelTask(MLTaskBase):
    """Task for training document classification models."""

    name = "lexiclass_worker.tasks.train_model_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return TrainModelInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return TrainModelOutput


@app.task(
    base=TrainModelTask,
    bind=True,
    retry_backoff=True,
    max_retries=3,
)
def train_model_task(self, **kwargs) -> dict:
    """Train a new model for document classification."""
    logger = self.get_task_logger(
        self.request.id,
        kwargs.get("project_id"),
    )
    
    try:
        # Validate input
        input_data = self.validate_input(kwargs)
        settings = get_settings()

        logger.info(
            "Starting model training",
            extra={"input": input_data.model_dump()},
        )

        # Get storage paths
        model_path = settings.get_model_path(input_data.project_id)
        index_path = settings.get_index_path(input_data.project_id)

        # Load labels and train
        try:
            labels = load_labels(input_data.labels_path)
        except Exception as e:
            raise DocumentError(
                "Failed to load labels",
                details={
                    "labels_path": str(input_data.labels_path),
                    "error": str(e),
                },
            ) from e

        if input_data.document_ids:
            labels = {k: v for k, v in labels.items() if k in input_data.document_ids}
            if not labels:
                raise DocumentError(
                    "No valid documents found for training",
                    details={
                        "requested_ids": input_data.document_ids,
                    },
                )

        # Train model
        try:
            self.classifier.load_index(index_path)
            metrics = self.classifier.train(labels)
            self.classifier.save_model(model_path, index_path=index_path)
        except Exception as e:
            raise ModelError(
                "Model training failed",
                details={
                    "num_labels": len(labels),
                    "model_path": str(model_path),
                    "error": str(e),
                },
            ) from e

        logger.info(
            "Model training completed",
            extra={
                "metrics": metrics,
                "num_labels": len(labels),
            },
        )

        # Prepare and validate output
        output_data = {
            "status": "completed",
            "project_id": input_data.project_id,
            "model_path": str(model_path),
            "num_labels": len(labels),
            "metrics": metrics,
        }
        validated_output = self.validate_output(output_data)
        return validated_output.model_dump()

    except Exception as e:
        logger.exception("Unexpected error during training")
        raise
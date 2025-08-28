"""Base classes and utilities for ML tasks."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from celery import Task
from lexiclass.classifier import SVMDocumentClassifier
from pydantic import BaseModel, ConfigDict, ValidationError

from .exceptions import LexiClassWorkerError, TaskValidationError
from .logging import get_task_logger


class TaskInput(BaseModel):
    """Base class for task input validation."""
    model_config = ConfigDict(extra="forbid")


class TaskOutput(BaseModel):
    """Base class for task output validation."""
    model_config = ConfigDict(extra="forbid")

    status: str
    project_id: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class MLTaskBase(Task, ABC):
    """Abstract base class for ML tasks with common functionality."""

    _classifier: Optional[SVMDocumentClassifier] = None
    abstract = True

    def __init__(self):
        self.logger = logging.getLogger(f"lexiclass_worker.task.{self.name}")

    def get_task_logger(self, task_id: str, project_id: Optional[str] = None) -> logging.Logger:
        """Get a logger with task context."""
        return get_task_logger(task_id, project_id)

    @property
    def classifier(self) -> SVMDocumentClassifier:
        """Get or create the classifier instance."""
        if self._classifier is None:
            self._classifier = SVMDocumentClassifier()
        return self._classifier

    def validate_input(self, input_data: Dict[str, Any]) -> TaskInput:
        """Validate task input data using the task's input schema."""
        print("DEBUG: Validating input data:", input_data)
        print("DEBUG: Using schema:", self.input_schema)
        try:
            return self.input_schema(**input_data)
        except ValidationError as e:
            error_details = {"validation_errors": e.errors()}
            print("DEBUG: Validation error details:", error_details)
            raise TaskValidationError(
                "Invalid task input",
                details=error_details,
            )

    def validate_output(self, output_data: Dict[str, Any]) -> TaskOutput:
        """Validate task output data using the task's output schema."""
        try:
            return self.output_schema(**output_data)
        except ValidationError as e:
            raise TaskValidationError(
                "Invalid task output",
                details={"validation_errors": e.errors()},
            )

    @property
    @abstractmethod
    def input_schema(self) -> type[TaskInput]:
        """Get the Pydantic schema for task input validation."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[TaskOutput]:
        """Get the Pydantic schema for task output validation."""
        pass

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with structured logging."""
        logger = self.get_task_logger(
            task_id,
            kwargs.get("project_id", "unknown"),
        )

        if isinstance(exc, LexiClassWorkerError):
            error_code = exc.code
            error_details = exc.details
        else:
            error_code = "INTERNAL_ERROR"
            error_details = {}

        logger.error(
            f"Task failed: {str(exc)}",
            extra={
                "error_code": error_code,
                "error_details": error_details,
                "args": args,
                "kwargs": kwargs,
            },
            exc_info=einfo,
        )

        return self.validate_output({
            "status": "failed",
            "project_id": kwargs.get("project_id", "unknown"),
            "error": str(exc),
            "error_code": error_code,
            "error_details": error_details,
        })

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success with structured logging."""
        logger = self.get_task_logger(
            task_id,
            kwargs.get("project_id", "unknown"),
        )
        logger.info(
            "Task completed successfully",
            extra={
                "task_result": retval,
            },
        )
        return retval

    def __call__(self, *args, **kwargs):
        """Execute task with error handling."""
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            # Ensure all exceptions are converted to our custom format
            if not isinstance(e, LexiClassWorkerError):
                e = LexiClassWorkerError(
                    str(e),
                    details={"original_error": e.__class__.__name__},
                )
            raise e
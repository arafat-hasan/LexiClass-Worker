"""Prediction task implementation."""

from pathlib import Path
from typing import Dict, List, Optional

from lexiclass.io import DocumentLoader
from pydantic import Field

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings


class PredictDocumentsInput(TaskInput):
    """Input schema for document prediction task."""
    project_id: str = Field(..., description="Unique project identifier")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific document IDs to predict"
    )


class PredictDocumentsOutput(TaskOutput):
    """Output schema for document prediction task."""
    predictions: Dict[str, tuple[str, float]]
    num_documents: int


class PredictDocumentsTask(MLTaskBase):
    """Task for running document predictions."""

    name = "lexiclass_worker.tasks.predict_documents_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return PredictDocumentsInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return PredictDocumentsOutput


@app.task(base=PredictDocumentsTask, bind=True)
def predict_documents_task(self, **kwargs) -> dict:
    """Run prediction on documents.

    Args:
        project_id: Unique project identifier
        document_ids: Optional list of specific document IDs to predict

    Returns:
        Dict containing predictions for each document
    """
    # Validate input
    input_data = self.validate_input(kwargs)
    settings = get_settings()

    # Get storage paths
    model_path = settings.get_model_path(input_data.project_id)
    index_path = settings.get_index_path(input_data.project_id)

    # Load model and index
    self.classifier.load_model(model_path, index_path=index_path)

    # Get documents to predict
    if input_data.document_ids:
        docs = DocumentLoader.load_documents_by_ids(index_path, input_data.document_ids)
    else:
        docs = DocumentLoader.load_documents_from_index(index_path)

    # Run prediction
    predictions = self.classifier.predict(docs)

    # Prepare and validate output
    output_data = {
        "status": "completed",
        "project_id": input_data.project_id,
        "predictions": predictions,
        "num_documents": len(docs),
    }
    return self.validate_output(output_data)

"""Indexing task implementation."""

from pathlib import Path
from typing import Optional

from lexiclass.io import DocumentLoader
from pydantic import Field

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings


class IndexDocumentsInput(TaskInput):
    """Input schema for document indexing task."""
    project_id: str = Field(..., description="Unique project identifier")
    documents_path: Path = Field(..., description="Directory containing document files")
    is_incremental: bool = Field(
        default=False,
        description="Whether to update existing index"
    )


class IndexDocumentsOutput(TaskOutput):
    """Output schema for document indexing task."""
    index_path: str
    is_incremental: bool
    num_documents: Optional[int] = None


class IndexDocumentsTask(MLTaskBase):
    """Task for building document indexes."""

    name = "lexiclass_worker.tasks.index_documents_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return IndexDocumentsInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return IndexDocumentsOutput


@app.task(base=IndexDocumentsTask, bind=True)
def index_documents_task(self, **kwargs) -> dict:
    """Build or update the document index.

    Args:
        project_id: Unique project identifier
        documents_path: Directory containing document text files
        is_incremental: Whether to update existing index

    Returns:
        Dict containing indexing results
    """
    # Validate input
    input_data = self.validate_input(kwargs)
    settings = get_settings()

    # Get storage paths
    index_path = settings.get_index_path(input_data.project_id)

    def stream_factory():
        return DocumentLoader.iter_documents_from_directory(input_data.documents_path)

    # Count documents for reporting
    num_documents = sum(1 for _ in stream_factory())

    if input_data.is_incremental and self._classifier is not None:
        # Update existing index
        self.classifier.update_index(
            index_path=index_path,
            document_stream_factory=stream_factory,
        )
    else:
        # Build new index
        self.classifier.build_index(
            index_path=index_path,
            document_stream_factory=stream_factory,
        )

    # Prepare and validate output
    output_data = {
        "status": "completed",
        "project_id": input_data.project_id,
        "index_path": str(index_path),
        "is_incremental": input_data.is_incremental,
        "num_documents": num_documents,
    }
    return self.validate_output(output_data)

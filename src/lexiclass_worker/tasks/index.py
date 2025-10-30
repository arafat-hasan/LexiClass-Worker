"""Indexing task implementation."""

import asyncio
from pathlib import Path
from typing import Optional

from lexiclass.io import DocumentLoader
from pydantic import Field

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import IndexStatus, update_document_status, get_document_ids_by_project


class IndexDocumentsInput(TaskInput):
    """Input schema for document indexing task."""
    project_id: int = Field(..., description="Unique project identifier")
    documents_path: str = Field(..., description="Directory containing document files")
    is_incremental: bool = Field(
        default=False,
        description="Whether to update existing index"
    )


class IndexDocumentsOutput(TaskOutput):
    """Output schema for document indexing task."""
    index_path: Optional[str] = None
    is_incremental: Optional[bool] = None
    num_documents: Optional[int] = None


class IndexDocumentsTask(MLTaskBase):
    """Task for building document indexes."""

    name = "lexiclass_worker.tasks.index.index_documents_task"

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
    # Debug input
    print("DEBUG: Received task kwargs:", kwargs)
    
    # Validate input
    try:
        input_data = self.validate_input(kwargs)
    except Exception as e:
        print("DEBUG: Validation error details:", str(e))
        raise
    settings = get_settings()

    # Get storage paths
    index_path = settings.get_index_path(input_data.project_id)
    documents_path = Path(input_data.documents_path)

    def stream_factory():
        return DocumentLoader.iter_documents_from_directory(str(documents_path))

    # Count documents for reporting
    num_documents = sum(1 for _ in stream_factory())

    # Build new index (incremental updates not yet supported)
    self.classifier.build_index(
        index_path=str(index_path),
        document_stream_factory=stream_factory,
    )

    # Update document statuses in database
    try:
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Get all document IDs for this project
            document_ids = loop.run_until_complete(get_document_ids_by_project(input_data.project_id))

            # Update all documents to 'indexed' status
            loop.run_until_complete(
                update_document_status(
                    project_id=input_data.project_id,
                    document_ids=document_ids,
                    status=IndexStatus.INDEXED
                )
            )
            print(f"Updated {len(document_ids)} documents to 'indexed' status")
        finally:
            loop.close()
    except Exception as e:
        print(f"Warning: Failed to update document statuses: {e}")
        # Continue even if status update fails - indexing was successful

    # Prepare and validate output
    output_data = {
        "status": "completed",
        "project_id": input_data.project_id,
        "index_path": str(index_path),
        "is_incremental": input_data.is_incremental,
        "num_documents": num_documents,
    }
    validated_output = self.validate_output(output_data)
    return validated_output.model_dump()

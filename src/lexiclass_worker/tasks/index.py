"""Indexing task implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from lexiclass.io import DocumentLoader
from pydantic import Field

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import (
    IndexStatus,
    update_document_status,
    get_document_ids_by_project,
    update_project_index_status,
    update_indexing_status,
)

logger = logging.getLogger(__name__)


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
    # Validate input
    input_data = self.validate_input(kwargs)
    settings = get_settings()

    # Get storage paths
    index_path = settings.get_index_path(input_data.project_id)
    documents_path = Path(input_data.documents_path)

    def stream_factory():
        return DocumentLoader.iter_documents_from_directory(str(documents_path))

    # Count documents for reporting
    num_documents = sum(1 for _ in stream_factory())
    logger.info(
        "Starting indexing for project %d with %d documents from %s",
        input_data.project_id, num_documents, documents_path
    )

    # Build new index (incremental updates not yet supported)
    # Create document index if not exists
    if self.classifier.document_index is None:
        from lexiclass.index import DocumentIndex
        self.classifier.document_index = DocumentIndex()

    # Track indexing success/failure
    indexing_succeeded = False
    error_message = None

    try:
        logger.info("Building document index for project %d", input_data.project_id)

        # Determine token cache path
        token_cache_path = str(index_path) + '.tokens.jsonl.gz'

        # Build index with token caching but without final save
        self.classifier.document_index.build_index(
            feature_extractor=self.classifier.feature_extractor,
            tokenizer=self.classifier.tokenizer,
            index_path=None,  # Don't auto-save yet (we'll save manually after validation)
            document_stream_factory=stream_factory,
            token_cache_path=token_cache_path,  # Explicit cache path
        )
        self.classifier.index_built = True

        # Check if the index is empty (0 features) - do this BEFORE attempting to save
        num_features = self.classifier.feature_extractor.num_features()
        if num_features == 0:
            raise ValueError(
                f"Document indexing failed: No valid features extracted from {num_documents} documents. "
                "This usually happens when: (1) documents are too short or contain only common words, "
                "(2) all tokens were filtered out as stop words, or (3) documents have insufficient "
                "vocabulary diversity. Please ensure documents contain meaningful text content with "
                "at least 50 characters and varied vocabulary."
            )

        logger.info(
            "Index built successfully with %d features from %d documents",
            num_features,
            num_documents
        )

        # Only save if we have valid features
        self.classifier.document_index.save_index(str(index_path))
        logger.info("Document index saved to %s", index_path)

        # Save the feature extractor alongside the index
        import pickle
        extractor_path = str(index_path) + '.extractor'
        with open(extractor_path, 'wb') as f:
            pickle.dump(self.classifier.feature_extractor, f, protocol=2)
        logger.info("Feature extractor saved to %s", extractor_path)

        # Mark indexing as successful
        indexing_succeeded = True

    except ValueError as e:
        # Validation errors - clear user-facing messages
        error_message = str(e)
        logger.error("Document indexing validation failed: %s", error_message)
        raise
    except ZeroDivisionError as e:
        # Gensim's ZeroDivisionError when trying to save with 0 features
        error_message = (
            f"Document indexing failed: No valid features extracted from {num_documents} documents. "
            "This usually happens when: (1) documents are too short or contain only common words, "
            "(2) all tokens were filtered out as stop words, or (3) documents have insufficient "
            "vocabulary diversity. Please ensure documents contain meaningful text content with "
            "at least 50 characters and varied vocabulary."
        )
        logger.error("Document indexing failed with ZeroDivisionError: %s", error_message)
        raise ValueError(error_message) from e
    except Exception as e:
        # Unexpected errors - wrap with context
        error_message = (
            f"Failed to build document index: {str(e)}. "
            f"Please check that documents are properly formatted and contain valid text."
        )
        logger.error("Unexpected error during document indexing: %s", error_message, exc_info=True)
        raise ValueError(error_message) from e
    finally:
        # ALWAYS update status - whether indexing succeeded or failed
        # This ensures database reflects true state
        logger.info(
            "Updating indexing status: project=%d, success=%s, documents=%d",
            input_data.project_id, indexing_succeeded, num_documents
        )

        try:
            # Create new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Update both document and project statuses atomically
                loop.run_until_complete(
                    update_indexing_status(
                        project_id=input_data.project_id,
                        success=indexing_succeeded,
                        error_message=error_message,
                        num_documents=num_documents,
                    )
                )
                logger.info(
                    "Successfully updated indexing status for project %d: %s",
                    input_data.project_id,
                    "INDEXED" if indexing_succeeded else "FAILED"
                )
            finally:
                loop.close()
        except Exception as e:
            # Critical error - status update failed
            # This should propagate and fail the task
            logger.error(
                "CRITICAL: Failed to update indexing status for project %d: %s",
                input_data.project_id, str(e), exc_info=True
            )
            # Re-raise to fail the task properly
            # Celery will handle retries
            raise

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

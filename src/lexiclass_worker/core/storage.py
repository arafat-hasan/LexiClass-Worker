"""Storage management for models and indexes."""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backend implementations."""

    def save_model(self, model_path: Path, index_path: Path) -> bool:
        """Save model and associated index files."""
        ...

    def load_model(self, model_path: Path, index_path: Path) -> bool:
        """Load model and associated index files."""
        ...

    def delete_model(self, project_id: str) -> bool:
        """Delete model and associated files."""
        ...


class LocalStorageBackend:
    """Local filesystem storage implementation."""

    def save_model(self, model_path: Path, index_path: Path) -> bool:
        """Save model and index files to local storage."""
        try:
            # Ensure parent directories exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(
                "Failed to prepare storage paths",
                extra={
                    "model_path": str(model_path),
                    "index_path": str(index_path),
                    "error": str(e),
                },
                exc_info=True,
            )
            return False

    def load_model(self, model_path: Path, index_path: Path) -> bool:
        """Check if model and index files exist."""
        return model_path.exists() and index_path.exists()

    def delete_model(self, project_id: str) -> bool:
        """Delete model and associated files."""
        try:
            from .config import get_settings

            settings = get_settings()
            model_path = settings.get_model_path(project_id)
            index_path = settings.get_index_path(project_id)

            # Delete model file
            if model_path.exists():
                model_path.unlink()

            # Delete index directory
            if index_path.parent.exists():
                for file in index_path.parent.glob("*"):
                    file.unlink()
                index_path.parent.rmdir()

            return True
        except Exception as e:
            logger.error(
                "Failed to delete model files",
                extra={
                    "project_id": project_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            return False

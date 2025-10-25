"""Configuration management for the worker service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class CeleryConfig(BaseModel):
    """Celery-specific configuration."""

    broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL for the Celery message broker"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="URL for the Celery result backend"
    )
    task_time_limit: int = Field(
        default=3600,
        description="Maximum runtime for tasks in seconds"
    )
    worker_prefetch_multiplier: int = Field(
        default=1,
        description="Number of tasks to prefetch"
    )


class StorageConfig(BaseModel):
    """Storage configuration for models and indexes."""

    base_path: Path = Field(
        default=Path("/tmp/lexiclass"),
        description="Base path for storing models and indexes"
    )
    models_dir: str = Field(
        default="models",
        description="Directory name for model storage"
    )
    indexes_dir: str = Field(
        default="indexes",
        description="Directory name for index storage"
    )


class Settings(BaseSettings):
    """Global settings for the worker service."""
    model_config = SettingsConfigDict(
        env_prefix="LEXICLASS_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    # Service configuration
    environment: str = Field(
        default="development",
        description="Deployment environment"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Logging format (json or text)"
    )

    # Database configuration
    database_uri: str = Field(
        default="postgresql+asyncpg://lexiclass:lexiclass@localhost/lexiclass",
        description="Database connection URI"
    )

    # Component configurations
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @property
    def DATABASE_URI(self) -> str:
        """Get database URI (for compatibility with API)."""
        return self.database_uri

    def get_model_path(self, project_id: str, field_id: Optional[str] = None) -> Path:
        """Get the path for storing model files.

        Args:
            project_id: Project ID
            field_id: Optional field ID for field-specific models

        Returns:
            Path to model file
        """
        if field_id:
            # Field-specific model path
            return (
                self.storage.base_path
                / project_id
                / self.storage.models_dir
                / field_id
                / "model.pkl"
            )
        else:
            # Legacy path for backward compatibility
            return (
                self.storage.base_path
                / project_id
                / self.storage.models_dir
                / "model.pkl"
            )

    def get_index_path(self, project_id: str) -> Path:
        """Get the path for storing index files.

        Note: Index is shared across all fields in a project.
        """
        return (
            self.storage.base_path
            / project_id
            / self.storage.indexes_dir
            / "index"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

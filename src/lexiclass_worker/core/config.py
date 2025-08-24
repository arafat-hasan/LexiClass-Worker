"""Configuration management for the worker service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CeleryConfig(BaseSettings):
    """Celery-specific configuration."""
    model_config = SettingsConfigDict(env_prefix="LEXICLASS_CELERY_")

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


class StorageConfig(BaseSettings):
    """Storage configuration for models and indexes."""
    model_config = SettingsConfigDict(env_prefix="LEXICLASS_STORAGE_")

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
        env_file=".env",
        env_file_encoding="utf-8",
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

    # Component configurations
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def get_model_path(self, project_id: str) -> Path:
        """Get the path for storing model files."""
        return (
            self.storage.base_path
            / self.storage.models_dir
            / project_id
            / "model.pkl"
        )

    def get_index_path(self, project_id: str) -> Path:
        """Get the path for storing index files."""
        return (
            self.storage.base_path
            / self.storage.indexes_dir
            / project_id
            / "index"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

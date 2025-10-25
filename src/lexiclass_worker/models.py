"""Database models - imported from LexiClass API repository.

This module re-exports models from the API to maintain single source of truth.
The API owns all database model definitions.
"""

# Import all models from the API
from lexiclass_api.models import (
    Base,
    Document,
    DocumentLabel,
    Field,
    FieldClass,
    IndexStatus,
    Model,
    ModelStatus,
    Prediction,
    Project,
)

__all__ = [
    "Base",
    "Document",
    "DocumentLabel",
    "Field",
    "FieldClass",
    "IndexStatus",
    "Model",
    "ModelStatus",
    "Prediction",
    "Project",
]

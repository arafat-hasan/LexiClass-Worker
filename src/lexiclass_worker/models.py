"""Database models - imported from LexiClass Core repository.

This module re-exports models from the Core package to maintain single source of truth.
The Core package owns all database model definitions.
"""

# Import all models from the Core package
from lexiclass_core.models import (
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

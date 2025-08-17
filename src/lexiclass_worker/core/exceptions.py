"""Custom exceptions for the worker service."""

from typing import Any, Dict, Optional


class LexiClassWorkerError(Exception):
    """Base exception for all worker errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class TaskValidationError(LexiClassWorkerError):
    """Raised when task input/output validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            details=details,
        )


class StorageError(LexiClassWorkerError):
    """Raised when storage operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code="STORAGE_ERROR",
            details=details,
        )


class ModelError(LexiClassWorkerError):
    """Raised when ML model operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code="MODEL_ERROR",
            details=details,
        )


class DocumentError(LexiClassWorkerError):
    """Raised when document operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            code="DOCUMENT_ERROR",
            details=details,
        )

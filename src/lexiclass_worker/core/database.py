"""Database connection and utilities for the worker - uses Core's session factory."""

import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Import models from Core - single source of truth
from lexiclass_core.models import Document, IndexStatus, Project, ProjectIndexStatus
from lexiclass_core.db.session import get_db_session, init_db
from lexiclass_core.db import session as session_module

from .config import get_settings

logger = logging.getLogger(__name__)

# Thread lock for lazy initialization (Double-Check Locking pattern)
_init_lock = threading.Lock()
_initialized = False


def initialize_database() -> None:
    """Initialize database connection using Core's session factory.

    This function is idempotent - safe to call multiple times.
    Uses Singleton pattern to ensure only one session factory per process.

    Raises:
        Exception: If database initialization fails
    """
    global _initialized

    # Fast path: already initialized
    if _initialized and session_module.AsyncSessionFactory is not None:
        logger.debug("Database already initialized, skipping")
        return

    # Slow path: need to initialize
    with _init_lock:
        # Double-check after acquiring lock
        if _initialized and session_module.AsyncSessionFactory is not None:
            logger.debug("Database already initialized by another thread, skipping")
            return

        try:
            settings = get_settings()
            logger.info(f"Initializing database session factory with URI: {settings.DATABASE_URI}")
            init_db(str(settings.DATABASE_URI))
            _initialized = True
            logger.info("Database session factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
            raise


def ensure_db_initialized() -> None:
    """Ensure database is initialized (lazy initialization fallback).

    This provides a safety net if worker initialization failed.
    Uses lazy initialization pattern with thread safety.

    Raises:
        RuntimeError: If database initialization fails
    """
    if session_module.AsyncSessionFactory is None:
        logger.warning(
            "Database not initialized - attempting lazy initialization. "
            "This should have been done in worker_process_init signal!"
        )
        try:
            initialize_database()
        except Exception as e:
            error_msg = (
                f"Lazy database initialization failed: {str(e)}. "
                "This indicates a worker configuration problem. "
                "Database should be initialized during worker startup."
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


async def update_document_status(
    project_id: int,
    document_ids: List[str],
    status: IndexStatus
) -> None:
    """Update document index status in the database.

    Args:
        project_id: Project ID
        document_ids: List of document IDs to update
        status: New index status
    """
    # Ensure database is initialized (lazy initialization fallback)
    ensure_db_initialized()

    async with get_db_session() as session:
        stmt = (
            update(Document)
            .where(Document.project_id == project_id)
            .where(Document.id.in_(document_ids))
            .values(index_status=status)
        )
        result = await session.execute(stmt)

        logger.info(
            f"Updated {result.rowcount} documents to status '{status.value}' "
            f"for project {project_id}"
        )


async def get_document_ids_by_project(project_id: int) -> List[str]:
    """Get all document IDs for a project.

    Args:
        project_id: Project ID

    Returns:
        List of document IDs
    """
    # Ensure database is initialized (lazy initialization fallback)
    ensure_db_initialized()

    try:
        async with get_db_session() as session:
            stmt = select(Document.id).where(Document.project_id == project_id)
            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch document IDs: {e}")
        raise


async def update_project_index_status(
    project_id: int,
    index_status: str,
) -> None:
    """Update project index status and last_indexed_at timestamp.

    Args:
        project_id: Project ID
        index_status: New index status (e.g., 'indexed', 'failed')
    """
    # Ensure database is initialized (lazy initialization fallback)
    ensure_db_initialized()

    async with get_db_session() as session:
        stmt = (
            update(Project)
            .where(Project.id == project_id)
            .values(
                index_status=index_status,
                last_indexed_at=datetime.now(timezone.utc)
            )
        )
        result = await session.execute(stmt)

        logger.info(
            f"Updated project {project_id} index_status to '{index_status}' "
            f"and set last_indexed_at timestamp"
        )


async def get_project_index_status(project_id: int) -> Optional[ProjectIndexStatus]:
    """Get current index status of a project.

    Args:
        project_id: Project ID

    Returns:
        Current ProjectIndexStatus or None
    """
    # Ensure database is initialized (lazy initialization fallback)
    ensure_db_initialized()

    async with get_db_session() as session:
        stmt = select(Project.index_status).where(Project.id == project_id)
        result = await session.execute(stmt)
        status = result.scalar_one_or_none()
        return status


async def update_indexing_status(
    project_id: int,
    success: bool,
    error_message: Optional[str] = None,
    num_documents: Optional[int] = None,
    enable_idempotency: bool = True,
) -> None:
    """Update indexing status for both documents and project atomically.

    This function ensures atomic updates with proper state transitions:
    - On success: Documents → INDEXED, Project → INDEXED
    - On failure: Documents → FAILED, Project → FAILED

    Implements:
    - Command Pattern: Encapsulates status update operations
    - Transaction Script: Atomic updates within single transaction
    - Idempotency: Safe to call multiple times with same result

    Args:
        project_id: Project ID
        success: Whether indexing succeeded
        error_message: Error message if indexing failed
        num_documents: Number of documents indexed (for logging)
        enable_idempotency: Whether to check for idempotent updates (default: True)

    Raises:
        Exception: If database update fails
    """
    # Ensure database is initialized (lazy initialization fallback)
    ensure_db_initialized()

    # Determine target statuses based on success/failure
    if success:
        doc_status = IndexStatus.INDEXED
        project_status = ProjectIndexStatus.INDEXED
        log_level = "info"
    else:
        doc_status = IndexStatus.FAILED
        project_status = ProjectIndexStatus.FAILED
        log_level = "error"

    async with get_db_session() as session:
        # Idempotency check - avoid unnecessary updates
        if enable_idempotency and success:
            current_status = await get_project_index_status(project_id)
            if current_status == ProjectIndexStatus.INDEXED:
                logger.info(
                    f"Project {project_id} already indexed (idempotent operation), skipping update"
                )
                return

        # Get all document IDs for this project
        stmt = select(Document.id).where(Document.project_id == project_id)
        result = await session.execute(stmt)
        document_ids = [row[0] for row in result.fetchall()]

        if not document_ids:
            logger.warning(
                f"No documents found for project {project_id}, skipping document status update"
            )
        else:
            # Update all documents to target status
            stmt = (
                update(Document)
                .where(Document.project_id == project_id)
                .where(Document.id.in_(document_ids))
                .values(index_status=doc_status)
            )
            doc_result = await session.execute(stmt)

            if log_level == "info":
                logger.info(
                    f"Updated {doc_result.rowcount} documents to status '{doc_status.value}' "
                    f"for project {project_id}"
                )
            else:
                logger.error(
                    f"Updated {doc_result.rowcount} documents to status '{doc_status.value}' "
                    f"for project {project_id}"
                )

        # Update project index status and timestamp
        stmt = (
            update(Project)
            .where(Project.id == project_id)
            .values(
                index_status=project_status,
                last_indexed_at=datetime.now(timezone.utc)
            )
        )
        proj_result = await session.execute(stmt)

        # Log the update
        log_message = (
            f"Updated project {project_id} index_status to '{project_status.value}' "
            f"and set last_indexed_at timestamp. "
            f"Documents affected: {len(document_ids)}"
        )
        if error_message:
            log_message += f". Error: {error_message}"

        if log_level == "info":
            logger.info(log_message)
        else:
            logger.error(log_message)

        # Auto-commit happens via context manager
        # If any error occurs, auto-rollback happens

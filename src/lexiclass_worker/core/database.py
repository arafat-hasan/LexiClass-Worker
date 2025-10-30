"""Database connection and utilities for the worker - uses Core's session factory."""

import logging
from typing import List

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Import models from Core - single source of truth
from lexiclass_core.models import Document, IndexStatus
from lexiclass_core.db.session import get_db_session, init_db

from .config import get_settings

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    """Initialize database connection using Core's session factory."""
    settings = get_settings()
    init_db(str(settings.DATABASE_URI))


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
    try:
        async with get_db_session() as session:
            stmt = (
                update(Document)
                .where(Document.project_id == project_id)
                .where(Document.id.in_(document_ids))
                .values(index_status=status)
            )
            result = await session.execute(stmt)
            await session.commit()

            logger.info(
                f"Updated {result.rowcount} documents to status '{status.value}' "
                f"for project {project_id}"
            )
    except Exception as e:
        logger.error(f"Failed to update document statuses: {e}")
        raise


async def get_document_ids_by_project(project_id: int) -> List[str]:
    """Get all document IDs for a project.

    Args:
        project_id: Project ID

    Returns:
        List of document IDs
    """
    try:
        async with get_db_session() as session:
            stmt = select(Document.id).where(Document.project_id == project_id)
            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch document IDs: {e}")
        raise

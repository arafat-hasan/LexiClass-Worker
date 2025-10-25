"""Database connection and utilities for the worker."""

import logging
from contextlib import asynccontextmanager
from typing import List, AsyncGenerator

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Import models from Core - single source of truth
from lexiclass_core.models import Base, Document, IndexStatus

from .config import get_settings

logger = logging.getLogger(__name__)


# Create database engine
def get_engine():
    """Get database engine."""
    settings = get_settings()
    return create_async_engine(
        str(settings.DATABASE_URI),
        echo=False,
        pool_pre_ping=True,
    )


# Create session factory
def get_session_factory():
    """Get session factory."""
    engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def update_document_status(
    project_id: str,
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


async def get_document_ids_by_project(project_id: str) -> List[str]:
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

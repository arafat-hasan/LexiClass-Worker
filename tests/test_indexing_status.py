"""Integration tests for indexing status updates.

Tests verify that document and project index_status are correctly updated
in both success and failure scenarios.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from lexiclass_core.models import Base, Document, IndexStatus, Project, ProjectIndexStatus
from lexiclass_core.db.session import init_db

from lexiclass_worker.core.database import (
    update_indexing_status,
    get_project_index_status,
    update_document_status,
    update_project_index_status,
)


# Test database URL - use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    SessionFactory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Initialize the global session factory for database.py functions
    from lexiclass_core.db import session as session_module
    session_module.AsyncSessionFactory = SessionFactory

    async with SessionFactory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_project(test_db_session: AsyncSession) -> Project:
    """Create a test project."""
    project = Project(
        name="Test Project",
        description="Test project for indexing status tests",
        status="created",
        config={},
    )
    test_db_session.add(project)
    await test_db_session.commit()
    await test_db_session.refresh(project)
    return project


@pytest.fixture
async def test_documents(
    test_db_session: AsyncSession,
    test_project: Project
) -> list[Document]:
    """Create test documents."""
    documents = []
    for i in range(5):
        doc = Document(
            project_id=test_project.id,
            content_path=f"/fake/path/doc_{i}.txt",
            doc_metadata={"source": f"test_{i}"},
            index_status=IndexStatus.PENDING,
        )
        documents.append(doc)
        test_db_session.add(doc)

    await test_db_session.commit()
    for doc in documents:
        await test_db_session.refresh(doc)

    return documents


class TestIndexingStatusUpdates:
    """Test suite for indexing status updates."""

    @pytest.mark.asyncio
    async def test_successful_indexing_updates_status(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test that successful indexing updates documents and project to INDEXED."""
        # Act: Update status for successful indexing
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
            num_documents=len(test_documents),
        )

        # Assert: All documents should be INDEXED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        assert len(documents) == 5
        for doc in documents:
            assert doc.index_status == IndexStatus.INDEXED

        # Assert: Project should be INDEXED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.INDEXED
        assert test_project.last_indexed_at is not None

        # Assert: last_indexed_at should be recent (within last minute)
        time_diff = datetime.now(timezone.utc) - test_project.last_indexed_at
        assert time_diff.total_seconds() < 60

    @pytest.mark.asyncio
    async def test_failed_indexing_updates_status(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test that failed indexing updates documents and project to FAILED."""
        error_msg = "No valid features extracted from documents"

        # Act: Update status for failed indexing
        await update_indexing_status(
            project_id=test_project.id,
            success=False,
            error_message=error_msg,
            num_documents=len(test_documents),
        )

        # Assert: All documents should be FAILED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        assert len(documents) == 5
        for doc in documents:
            assert doc.index_status == IndexStatus.FAILED

        # Assert: Project should be FAILED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.FAILED
        assert test_project.last_indexed_at is not None

    @pytest.mark.asyncio
    async def test_idempotent_success_update(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test that updating to INDEXED twice is idempotent."""
        # First update
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
            num_documents=len(test_documents),
        )

        await test_db_session.refresh(test_project)
        first_timestamp = test_project.last_indexed_at

        # Wait a bit to ensure timestamp would change if update happens
        import time
        time.sleep(0.1)

        # Second update (should be idempotent)
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
            num_documents=len(test_documents),
            enable_idempotency=True,
        )

        # Assert: Timestamp should NOT have changed (idempotent)
        await test_db_session.refresh(test_project)
        assert test_project.last_indexed_at == first_timestamp

    @pytest.mark.asyncio
    async def test_status_update_with_no_documents(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
    ):
        """Test that status update works even with no documents."""
        # Act: Update status for project with no documents
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
            num_documents=0,
        )

        # Assert: Project should still be updated
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.INDEXED
        assert test_project.last_indexed_at is not None

    @pytest.mark.asyncio
    async def test_get_project_index_status(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
    ):
        """Test getting project index status."""
        # Initially should be None
        status = await get_project_index_status(test_project.id)
        assert status is None

        # Update to INDEXED
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
        )

        # Should now be INDEXED
        status = await get_project_index_status(test_project.id)
        assert status == ProjectIndexStatus.INDEXED

    @pytest.mark.asyncio
    async def test_atomic_transaction_rollback(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test that transaction rollback works on error."""
        # This test verifies that if an error occurs during status update,
        # the entire transaction is rolled back

        # Mock to force an error during project update
        with patch(
            'lexiclass_worker.core.database.update',
            side_effect=Exception("Database error")
        ):
            with pytest.raises(Exception, match="Database error"):
                await update_indexing_status(
                    project_id=test_project.id,
                    success=True,
                )

        # Assert: Documents should still be PENDING (rollback occurred)
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.PENDING

    @pytest.mark.asyncio
    async def test_partial_document_update(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test updating status when some documents are already indexed."""
        # Set first 2 documents to INDEXED
        for i in range(2):
            test_documents[i].index_status = IndexStatus.INDEXED
        await test_db_session.commit()

        # Update all to INDEXED
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
        )

        # Assert: All documents should be INDEXED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.INDEXED

    @pytest.mark.asyncio
    async def test_status_transition_from_in_progress(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test status transition from IN_PROGRESS to INDEXED/FAILED."""
        # Set project to IN_PROGRESS
        test_project.index_status = ProjectIndexStatus.IN_PROGRESS
        await test_db_session.commit()

        # Update to INDEXED
        await update_indexing_status(
            project_id=test_project.id,
            success=True,
        )

        # Assert: Should transition to INDEXED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.INDEXED


class TestBackwardCompatibility:
    """Test backward compatibility with old status update functions."""

    @pytest.mark.asyncio
    async def test_update_document_status_still_works(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
    ):
        """Test that old update_document_status function still works."""
        document_ids = [doc.id for doc in test_documents]

        await update_document_status(
            project_id=test_project.id,
            document_ids=document_ids,
            status=IndexStatus.INDEXED,
        )

        # Assert: All documents should be updated
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.INDEXED

    @pytest.mark.asyncio
    async def test_update_project_index_status_still_works(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
    ):
        """Test that old update_project_index_status function still works."""
        await update_project_index_status(
            project_id=test_project.id,
            index_status="indexed",
        )

        # Assert: Project should be updated
        await test_db_session.refresh(test_project)
        # Note: old function uses string, new model uses enum
        # The value should still work due to enum.value comparison
        assert test_project.index_status == ProjectIndexStatus.INDEXED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

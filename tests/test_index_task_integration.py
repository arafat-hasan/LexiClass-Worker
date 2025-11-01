"""End-to-end integration tests for the indexing task.

These tests verify the entire indexing workflow including status updates.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from lexiclass_core.models import Base, Document, IndexStatus, Project, ProjectIndexStatus

from lexiclass_worker.tasks.index import index_documents_task


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
async def test_db_session(test_db_engine):
    """Create test database session."""
    SessionFactory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Initialize the global session factory
    from lexiclass_core.db import session as session_module
    session_module.AsyncSessionFactory = SessionFactory

    async with SessionFactory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def temp_documents_dir():
    """Create temporary directory with test documents."""
    temp_dir = tempfile.mkdtemp()

    # Create some test documents
    test_docs = [
        ("doc1.txt", "This is a test document about machine learning and artificial intelligence."),
        ("doc2.txt", "Natural language processing is an important field in computer science."),
        ("doc3.txt", "Deep learning models require large amounts of training data."),
        ("doc4.txt", "Text classification is a common task in machine learning applications."),
        ("doc5.txt", "Support vector machines are effective for document classification tasks."),
    ]

    for filename, content in test_docs:
        file_path = Path(temp_dir) / filename
        file_path.write_text(content)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
async def test_project(test_db_session: AsyncSession) -> Project:
    """Create a test project."""
    project = Project(
        name="Test Project",
        description="Test project for indexing",
        status="created",
        config={},
        index_status=ProjectIndexStatus.IN_PROGRESS,  # Simulating API setting this
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


class TestIndexTaskIntegration:
    """Integration tests for the index_documents_task."""

    @pytest.mark.asyncio
    async def test_successful_indexing_updates_all_statuses(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
        temp_documents_dir: str,
    ):
        """Test that successful indexing updates all statuses correctly."""
        # Arrange: Mock the settings to use our temp directory
        with patch('lexiclass_worker.tasks.index.get_settings') as mock_settings:
            settings_mock = MagicMock()
            settings_mock.get_index_path.return_value = Path(temp_documents_dir) / "index"
            mock_settings.return_value = settings_mock

            # Create a mock task instance
            task_instance = MagicMock()
            task_instance.validate_input = lambda x: type('obj', (object,), x)
            task_instance.validate_output = lambda x: type('obj', (object,), {**x, 'model_dump': lambda: x})
            task_instance.classifier = MagicMock()
            task_instance.classifier.document_index = None

            # Mock the classifier methods to succeed
            mock_index = MagicMock()
            mock_extractor = MagicMock()
            mock_extractor.num_features.return_value = 100  # Success case
            task_instance.classifier.feature_extractor = mock_extractor
            task_instance.classifier.tokenizer = MagicMock()

            with patch('lexiclass_worker.tasks.index.DocumentIndex') as mock_index_class:
                mock_index_class.return_value = mock_index

                # Act: Run the indexing task
                result = index_documents_task(
                    task_instance,
                    project_id=test_project.id,
                    documents_path=temp_documents_dir,
                    is_incremental=False,
                )

        # Assert: Task should have completed
        assert result['status'] == 'completed'
        assert result['project_id'] == test_project.id

        # Assert: All documents should be INDEXED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.INDEXED

        # Assert: Project should be INDEXED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.INDEXED
        assert test_project.last_indexed_at is not None

    @pytest.mark.asyncio
    async def test_failed_indexing_updates_to_failed_status(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
        temp_documents_dir: str,
    ):
        """Test that failed indexing (0 features) updates status to FAILED."""
        # Arrange: Mock the settings
        with patch('lexiclass_worker.tasks.index.get_settings') as mock_settings:
            settings_mock = MagicMock()
            settings_mock.get_index_path.return_value = Path(temp_documents_dir) / "index"
            mock_settings.return_value = settings_mock

            # Create a mock task instance
            task_instance = MagicMock()
            task_instance.validate_input = lambda x: type('obj', (object,), x)
            task_instance.validate_output = lambda x: type('obj', (object,), {**x, 'model_dump': lambda: x})
            task_instance.classifier = MagicMock()
            task_instance.classifier.document_index = None

            # Mock the classifier to return 0 features (failure case)
            mock_index = MagicMock()
            mock_extractor = MagicMock()
            mock_extractor.num_features.return_value = 0  # Failure case
            task_instance.classifier.feature_extractor = mock_extractor
            task_instance.classifier.tokenizer = MagicMock()

            with patch('lexiclass_worker.tasks.index.DocumentIndex') as mock_index_class:
                mock_index_class.return_value = mock_index

                # Act: Run the indexing task (should fail)
                with pytest.raises(ValueError, match="No valid features extracted"):
                    index_documents_task(
                        task_instance,
                        project_id=test_project.id,
                        documents_path=temp_documents_dir,
                        is_incremental=False,
                    )

        # Assert: All documents should be FAILED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.FAILED

        # Assert: Project should be FAILED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.FAILED
        assert test_project.last_indexed_at is not None

    @pytest.mark.asyncio
    async def test_indexing_exception_updates_to_failed(
        self,
        test_db_session: AsyncSession,
        test_project: Project,
        test_documents: list[Document],
        temp_documents_dir: str,
    ):
        """Test that unexpected exceptions during indexing update status to FAILED."""
        # Arrange: Mock the settings
        with patch('lexiclass_worker.tasks.index.get_settings') as mock_settings:
            settings_mock = MagicMock()
            settings_mock.get_index_path.return_value = Path(temp_documents_dir) / "index"
            mock_settings.return_value = settings_mock

            # Create a mock task instance
            task_instance = MagicMock()
            task_instance.validate_input = lambda x: type('obj', (object,), x)
            task_instance.classifier = MagicMock()
            task_instance.classifier.document_index = None

            # Mock to raise an exception during indexing
            with patch('lexiclass_worker.tasks.index.DocumentIndex') as mock_index_class:
                mock_index_class.return_value.build_index.side_effect = Exception("Unexpected error")

                # Act: Run the indexing task (should fail)
                with pytest.raises(ValueError, match="Failed to build document index"):
                    index_documents_task(
                        task_instance,
                        project_id=test_project.id,
                        documents_path=temp_documents_dir,
                        is_incremental=False,
                    )

        # Assert: All documents should be FAILED
        stmt = select(Document).where(Document.project_id == test_project.id)
        result = await test_db_session.execute(stmt)
        documents = result.scalars().all()

        for doc in documents:
            assert doc.index_status == IndexStatus.FAILED

        # Assert: Project should be FAILED
        await test_db_session.refresh(test_project)
        assert test_project.index_status == ProjectIndexStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

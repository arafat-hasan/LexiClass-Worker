# LexiClass Worker - Development Guide

Complete guide for setting up a development environment and contributing to LexiClass Worker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Development Environment](#development-environment)
4. [Running Locally](#running-locally)
5. [Testing](#testing)
6. [Code Quality](#code-quality)
7. [Debugging](#debugging)
8. [Making Changes](#making-changes)
9. [Best Practices](#best-practices)

---

## Prerequisites

### Required Software

- **Python 3.11+**: `python --version`
- **PostgreSQL 14+**: `psql --version`
- **Redis 6+**: `redis-cli --version`
- **Git**: `git --version`
- **Docker** (optional): `docker --version`
- **Docker Compose** (optional): `docker-compose --version`

### System Dependencies

**macOS**:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 postgresql@14 redis git

# Start services
brew services start postgresql@14
brew services start redis
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv postgresql-14 redis-server git

# Start services
sudo systemctl start postgresql
sudo systemctl start redis
```

---

## Initial Setup

### 1. Clone Repository

```bash
# Clone the worker repository
git clone <lexiclass-worker-repo>
cd LexiClass-Worker

# Clone sibling repositories (for local development)
cd ..
git clone <lexiclass-repo>
git clone <lexiclass-core-repo>
```

### 2. Create Virtual Environment

```bash
cd LexiClass-Worker

# Create virtual environment
python3.11 -m venv .venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Upgrade pip and tools
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install worker in editable mode with dev dependencies
pip install -e ".[dev]"

# Install LexiClass library (if developing locally)
cd ../LexiClass
pip install -e .
cd ../LexiClass-Worker

# Install LexiClass-Core (if developing locally)
cd ../LexiClass-Core
pip install -e .
cd ../LexiClass-Worker
```

### 4. Install Optional Plugins

```bash
# For spaCy tokenizer
pip install spacy
python -m spacy download en_core_web_sm

# For XGBoost classifier
pip install xgboost

# For transformer models
pip install sentence-transformers transformers torch

# For SentencePiece tokenizer
pip install sentencepiece
```

---

## Development Environment

### Database Setup

```bash
# Create PostgreSQL database
createdb lexiclass_dev

# Or manually
psql postgres
CREATE DATABASE lexiclass_dev;
\q

# Apply migrations (if using alembic)
alembic upgrade head
```

### Redis Setup

```bash
# Start Redis (if not running)
redis-server

# Verify
redis-cli ping
# Should return: PONG
```

### Environment Configuration

Create `.env` file in project root:

```bash
# Development environment
LEXICLASS_ENVIRONMENT=development

# Database (use async driver)
DATABASE_URI=postgresql+asyncpg://localhost/lexiclass_dev

# Celery
CELERY__BROKER_URL=redis://localhost:6379/0
CELERY__RESULT_BACKEND=redis://localhost:6379/0
CELERY__TASK_TIME_LIMIT=3600
CELERY__WORKER_PREFETCH_MULTIPLIER=1

# Storage
STORAGE__BASE_PATH=./data
STORAGE__MODELS_DIR=models
STORAGE__INDEXES_DIR=indexes

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LEXICLASS_LOG_LEVEL=DEBUG

# LexiClass library settings
LEXICLASS_LOCALE=en
LEXICLASS_RANDOM_SEED=42
```

### Directory Structure

```bash
# Create storage directories
mkdir -p data/models data/indexes data/documents
```

---

## Running Locally

### Start Worker

```bash
# Activate virtual environment
source .venv/bin/activate

# Start worker with debug logging
celery -A lexiclass_worker.celery worker \
    --loglevel=debug \
    --concurrency=2

# With hot-reload (for development)
watchmedo auto-restart \
    --directory=./src \
    --pattern='*.py' \
    --recursive \
    -- celery -A lexiclass_worker.celery worker --loglevel=debug
```

### Using Docker

```bash
# Start all services (worker, redis, postgres)
docker compose up --build

# Start in background
docker compose up -d

# View logs
docker compose logs -f worker

# Restart worker after code changes
docker compose restart worker

# Stop all
docker compose down
```

### Test Task Submission

```python
# In Python REPL or script
from lexiclass_worker.tasks import train_field_model_task

# Submit a test task
result = train_field_model_task.delay(
    field_id=1,
    project_id=1,
    tokenizer="icu",
    feature_extractor="bow",
    classifier="svm"
)

# Check status
print(result.state)  # PENDING, STARTED, SUCCESS, FAILURE

# Get result (blocks until complete)
print(result.get(timeout=300))
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/lexiclass_worker --cov-report=html

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_tasks.py

# Specific test
pytest tests/unit/test_tasks.py::test_training_task

# With verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Test Database

Create separate test database:

```bash
createdb lexiclass_test

# Set in test environment
export DATABASE_URI=postgresql+asyncpg://localhost/lexiclass_test
```

### Writing Tests

**Unit Test Example**:
```python
# tests/unit/test_tasks.py
import pytest
from lexiclass_worker.tasks.field_train import TrainFieldModelInput

def test_train_input_validation():
    # Valid input
    input_data = TrainFieldModelInput(
        field_id=1,
        project_id=1,
        tokenizer="spacy"
    )
    assert input_data.tokenizer == "spacy"

    # Invalid input
    with pytest.raises(ValueError):
        TrainFieldModelInput(
            field_id="invalid",  # Should be int
            project_id=1
        )
```

**Integration Test Example**:
```python
# tests/integration/test_training_flow.py
import pytest
from lexiclass_worker.tasks import train_field_model_task

@pytest.mark.asyncio
async def test_full_training_flow(test_db, test_documents):
    # Setup test data
    field = await create_test_field(test_db)
    await create_test_labels(test_db, field.id)

    # Run training task
    result = train_field_model_task.apply(
        kwargs={
            "field_id": field.id,
            "project_id": field.project_id,
        }
    )

    # Verify results
    assert result.status == "SUCCESS"
    assert result.result["model_version"] == 1
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from lexiclass_core.db import init_db
from lexiclass_worker.core.config import get_settings

@pytest.fixture
async def test_db():
    """Setup test database."""
    settings = get_settings()
    init_db(str(settings.DATABASE_URI))
    yield
    # Cleanup

@pytest.fixture
def test_documents():
    """Create test documents."""
    return {
        "doc1": "This is a test document",
        "doc2": "Another test document",
    }
```

---

## Code Quality

### Formatting

```bash
# Format with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Both in one command
black src/ tests/ && isort src/ tests/
```

### Linting

```bash
# Flake8
flake8 src/ tests/

# Pylint
pylint src/lexiclass_worker

# Combine checks
flake8 src/ tests/ && pylint src/lexiclass_worker
```

### Type Checking

```bash
# MyPy
mypy src/

# Strict mode
mypy --strict src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

---

## Debugging

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG
LEXICLASS_LOG_LEVEL=DEBUG

# Or in worker command
celery -A lexiclass_worker.celery worker --loglevel=debug
```

### Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()

# In async functions
import asyncio
import pdb
pdb.set_trace()
```

### VS Code Debugging

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Celery Worker",
      "type": "python",
      "request": "launch",
      "module": "celery",
      "args": [
        "-A", "lexiclass_worker.celery",
        "worker",
        "--loglevel=debug",
        "--pool=solo"
      ],
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  ]
}
```

### Inspect Running Tasks

```bash
# Active tasks
celery -A lexiclass_worker.celery inspect active

# Registered tasks
celery -A lexiclass_worker.celery inspect registered

# Worker stats
celery -A lexiclass_worker.celery inspect stats

# Query task result
celery -A lexiclass_worker.celery result <task-id>
```

### Redis Inspection

```bash
# Connect to Redis
redis-cli

# List queues
KEYS *

# Queue length
LLEN celery

# View tasks in queue
LRANGE celery 0 -1

# View task result
GET celery-task-meta-<task-id>
```

---

## Making Changes

### Development Workflow

1. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests for new code
   - Update documentation

3. **Run Tests**
   ```bash
   pytest
   black src/ tests/
   isort src/ tests/
   mypy src/
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: Add new feature"
   ```

5. **Push**
   ```bash
   git push origin feature/my-feature
   ```

### Adding a New Task

1. **Create task file**:
   ```bash
   touch src/lexiclass_worker/tasks/my_task.py
   ```

2. **Define schemas**:
   ```python
   from pydantic import Field as PydanticField
   from ..core.base import TaskInput, TaskOutput

   class MyTaskInput(TaskInput):
       param1: int
       param2: str = "default"

   class MyTaskOutput(TaskOutput):
       result: dict
   ```

3. **Implement task**:
   ```python
   @app.task(base=MyTaskBase, bind=True)
   def my_task(self, **kwargs):
       input_data = self.validate_input(kwargs)
       # Implementation
       return self.validate_output(result)
   ```

4. **Export task**:
   ```python
   # In tasks/__init__.py
   from .my_task import my_task
   __all__ = [..., "my_task"]
   ```

5. **Add tests**:
   ```python
   # tests/unit/test_my_task.py
   def test_my_task():
       # Test implementation
       pass
   ```

### Updating Dependencies

```bash
# Update requirements
pip install --upgrade <package>

# Freeze requirements
pip freeze > requirements.txt

# Or use pip-tools
pip-compile pyproject.toml
```

---

## Best Practices

### Code Style

1. **Follow PEP 8**
2. **Use type hints**:
   ```python
   def my_function(arg: str) -> dict:
       pass
   ```

3. **Write docstrings**:
   ```python
   def my_function(arg: str) -> dict:
       """Short description.

       Args:
           arg: Description

       Returns:
           Description
       """
       pass
   ```

4. **Keep functions small** (< 50 lines)
5. **One responsibility per function**

### Testing

1. **Test coverage > 80%**
2. **Unit tests for all functions**
3. **Integration tests for workflows**
4. **Test edge cases**
5. **Mock external dependencies**

### Git Commits

Use conventional commits:
```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Refactor code
chore: Update dependencies
```

### Documentation

1. **Update README** for major changes
2. **Add docstrings** to all public functions
3. **Update CHANGELOG**
4. **Add examples** for new features

---

## Common Development Tasks

### Reset Database

```bash
# Drop and recreate
dropdb lexiclass_dev && createdb lexiclass_dev

# Run migrations
alembic upgrade head
```

### Clear Redis Queue

```bash
# Connect to Redis
redis-cli

# Clear all
FLUSHALL

# Or specific queue
DEL celery
```

### Rebuild Docker

```bash
# Remove all containers and volumes
docker compose down -v

# Rebuild and start
docker compose up --build
```

### Generate Sample Data

```python
# scripts/generate_sample_data.py
import asyncio
from lexiclass_core.db import get_db_session
from lexiclass_core.models import Project, Field

async def create_sample_data():
    async with get_db_session() as session:
        # Create project
        project = Project(name="Test Project")
        session.add(project)
        await session.commit()

if __name__ == "__main__":
    asyncio.run(create_sample_data())
```

---

## Troubleshooting Development Issues

### Import Errors

```bash
# Verify installation
pip show lexiclass-worker

# Reinstall in editable mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
pg_isready

# Check connection
psql lexiclass_dev -c "SELECT 1"

# Verify DATABASE_URI
echo $DATABASE_URI
```

### Redis Connection Errors

```bash
# Check Redis is running
redis-cli ping

# Check port
redis-cli -p 6379 ping
```

### Worker Not Picking Up Tasks

```bash
# Check worker is running
celery -A lexiclass_worker.celery inspect active

# Check queue
redis-cli LLEN celery

# Restart worker
docker compose restart worker
```

---

## Resources

- **Python Docs**: https://docs.python.org/3/
- **Celery Docs**: https://docs.celeryproject.org/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org/
- **Pytest Docs**: https://docs.pytest.org/

---

**Last Updated**: 2025-11-02

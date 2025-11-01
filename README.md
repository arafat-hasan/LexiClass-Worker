# LexiClass Worker

**Asynchronous ML Task Executor for the LexiClass Document Classification Platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

LexiClass Worker is a **Celery-based asynchronous task executor** that powers the machine learning operations in the LexiClass distributed document classification platform. It picks tasks from Redis queues and executes ML operations (training, prediction, evaluation) using the LexiClass ML library.

### Key Responsibilities

- 🔄 **Execute ML Tasks**: Index documents, train models, run predictions, and evaluate performance
- 📦 **Task Orchestration**: Manage task lifecycle, status updates, and error handling
- 💾 **Result Persistence**: Store model files to disk and metadata to PostgreSQL
- 🔌 **Plugin Management**: Support flexible ML pipelines using LexiClass plugins
- 📊 **Database Integration**: Use LexiClass-Core models for data persistence
- ⚡ **Async Processing**: Handle long-running ML operations without blocking the API

### Position in the Architecture

```
┌─────────────┐
│ LexiClass   │  Pure ML Library (tokenization, training, prediction)
│  (Library)  │  - 11 plugins (tokenizers, features, classifiers)
└─────────────┘  - Stateless, no database dependencies
       ↑
       │ Uses for ML operations
       │
┌──────┴──────────────────────────────────────────────┐
│           LexiClass Worker (This Repo)              │
│  - Celery tasks for async ML execution              │
│  - Task schemas (Pydantic validation)               │
│  - Result persistence to database/disk              │
│  - Status tracking and error handling               │
└──────┬──────────────────────────────────────────────┘
       ↑
       │ Depends on for models/schemas
       │
┌──────┴──────────────────────────────────────────────┐
│            LexiClass-Core (Foundation)               │
│  - ORM Models (SQLAlchemy)                          │
│  - Pydantic Schemas                                 │
│  - Database session management                      │
│  - Shared configuration and utilities               │
└─────────────────────────────────────────────────────┘
```

### How It Works

1. **API** submits a task to Redis queue (e.g., train model)
2. **Worker** picks up the task from Redis
3. **Worker** validates input using Pydantic schemas
4. **Worker** calls LexiClass library to perform ML operations
5. **Worker** stores results using LexiClass-Core models
6. **Worker** updates task status in Redis and database
7. **API** polls for results and returns to client

---

## Quick Start

### Prerequisites

- Python 3.11+
- Redis server
- PostgreSQL database
- Access to LexiClass and LexiClass-Core packages

### Installation

```bash
# Clone the repository
git clone <lexiclass-worker-repo>
cd LexiClass-Worker

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file:

```bash
# Database
DATABASE_URI=postgresql+asyncpg://user:password@localhost:5432/lexiclass

# Redis
CELERY__BROKER_URL=redis://localhost:6379/0
CELERY__RESULT_BACKEND=redis://localhost:6379/0

# Storage
STORAGE__BASE_PATH=/data/lexiclass

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Running the Worker

```bash
# Start Celery worker
celery -A lexiclass_worker.celery worker --loglevel=info

# With multiple workers
celery -A lexiclass_worker.celery worker --loglevel=info --concurrency=4

# With specific queues
celery -A lexiclass_worker.celery worker --loglevel=info -Q training,indexing,prediction
```

### Running with Docker

```bash
# Development
docker compose up --build

# Production
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose logs -f worker
```

---

## Available Tasks

### Training Task
Train a classification model on labeled documents.

```python
from lexiclass_worker.tasks import train_field_model_task

result = train_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",           # icu, spacy, sentencepiece, huggingface
    feature_extractor="tfidf",   # bow, tfidf, fasttext, sbert
    classifier="svm",            # svm, xgboost, transformer
    locale="en"
)
```

### Prediction Task
Run predictions on documents using a trained model.

```python
from lexiclass_worker.tasks import predict_field_documents_task

result = predict_field_documents_task.delay(
    field_id=123,
    project_id=456,
    document_ids=[789, 790, 791],
    tokenizer="spacy",           # Must match training
    feature_extractor="tfidf",   # Must match training
    locale="en"
)
```

### Evaluation Task
Evaluate model performance on test data.

```python
from lexiclass_worker.tasks import evaluate_field_model_task

result = evaluate_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",           # Must match training
    feature_extractor="tfidf",   # Must match training
    locale="en"
)
```

---

## Plugin System

The worker leverages the LexiClass plugin ecosystem for flexible ML pipelines:

### Tokenizers
- **icu** (default) - Fast, locale-aware tokenization
- **spacy** - Linguistic features, stop word filtering
- **sentencepiece** - Trainable subword tokenization
- **huggingface** - Access to 1000+ pre-trained tokenizers

### Feature Extractors
- **bow** (default) - Simple bag-of-words
- **tfidf** - TF-IDF weighting for better relevance
- **fasttext** - Subword embeddings
- **sbert** - Sentence-BERT transformer embeddings

### Classifiers
- **svm** (default) - Fast linear SVM
- **xgboost** - Gradient boosting for higher accuracy
- **transformer** - Fine-tuned BERT/RoBERTa for maximum accuracy

### Recommended Combinations

| Use Case | Tokenizer | Features | Classifier | Speed | Accuracy |
|----------|-----------|----------|------------|-------|----------|
| **Quick Prototype** | icu | bow | svm | ⚡⚡⚡ | ~88% |
| **Production** | spacy | tfidf | svm | ⚡⚡⚡ | ~92% |
| **High Accuracy** | spacy | tfidf | xgboost | ⚡⚡ | ~94% |
| **State-of-the-Art** | huggingface | sbert | transformer | ⚡ | ~96% |

---

## Features

### Task Management
- ✅ **Pydantic Validation**: Input/output schema validation for all tasks
- ✅ **Error Handling**: Comprehensive error handling with retry logic
- ✅ **Status Tracking**: Real-time task status updates
- ✅ **Priority Queues**: Separate queues for different task types

### Database Integration
- ✅ **Async Sessions**: Efficient async database operations
- ✅ **Worker-level Init**: Database initialized once per worker process
- ✅ **Lazy Fallback**: Automatic recovery if initialization fails
- ✅ **Core Models**: Uses LexiClass-Core ORM models

### ML Operations
- ✅ **Plugin Support**: All 11 LexiClass plugins supported
- ✅ **Model Versioning**: Automatic version management
- ✅ **Dynamic Paths**: Model paths generated from metadata
- ✅ **Dual Storage**: Results in both database and disk

### Monitoring & Logging
- ✅ **Structured Logging**: JSON or text format
- ✅ **Task Tracking**: Correlation IDs for request tracing
- ✅ **Health Checks**: Redis and database connection monitoring
- ✅ **Performance Metrics**: Task duration and success rates

---

## Project Structure

```
src/lexiclass_worker/
├── core/                      # Core components
│   ├── base.py               # Base task classes
│   ├── config.py             # Configuration management
│   ├── database.py           # Database session management
│   ├── exceptions.py         # Custom exceptions
│   ├── logging.py            # Logging configuration
│   └── storage.py            # Storage interface
├── tasks/                     # Task implementations
│   ├── field_train.py        # Training task
│   ├── field_predict.py      # Prediction task
│   ├── field_evaluate.py     # Evaluation task
│   ├── index.py              # Indexing task (legacy)
│   ├── train.py              # Training task (legacy)
│   └── predict.py            # Prediction task (legacy)
├── models.py                  # Re-export Core models
├── celery.py                  # Celery app configuration
└── __init__.py               # Package initialization

tests/                         # Test suite
├── unit/                     # Unit tests
├── integration/              # Integration tests
└── conftest.py               # Pytest configuration

docs/                          # Documentation
├── ARCHITECTURE.md           # System architecture
├── DEVELOPMENT.md            # Development guide
├── DEPLOYMENT.md             # Deployment guide
├── TASKS_REFERENCE.md        # Task API reference
├── CONFIGURATION.md          # Configuration options
└── TROUBLESHOOTING.md        # Common issues
```

---

## Dependencies

### Required Packages
- **lexiclass** - ML library for document classification
- **lexiclass-core** - Shared models, schemas, and utilities
- **celery[redis]** - Distributed task queue
- **pydantic** - Data validation
- **sqlalchemy[asyncio]** - Async ORM
- **asyncpg** - PostgreSQL async driver

### Optional Plugins
```bash
# For better accuracy
pip install spacy xgboost
python -m spacy download en_core_web_sm

# For state-of-the-art models
pip install sentence-transformers transformers torch

# For GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src/lexiclass_worker --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With verbose output
pytest -v

# Stop on first failure
pytest -x
```

---

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t lexiclass-worker:latest .

# Run with docker-compose
docker compose -f docker-compose.prod.yml up -d

# Scale workers
docker compose -f docker-compose.prod.yml up -d --scale worker=4

# Check health
docker compose ps
```

### Environment Variables

See [CONFIGURATION.md](docs/CONFIGURATION.md) for complete list.

Key variables:
- `DATABASE_URI` - PostgreSQL connection string
- `CELERY__BROKER_URL` - Redis broker URL
- `CELERY__RESULT_BACKEND` - Redis result backend URL
- `STORAGE__BASE_PATH` - Base path for file storage
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

---

## Monitoring

### Health Checks

```bash
# Check worker status
celery -A lexiclass_worker.celery inspect active

# Check registered tasks
celery -A lexiclass_worker.celery inspect registered

# Check stats
celery -A lexiclass_worker.celery inspect stats
```

### Celery Flower (Optional)

```bash
# Install flower
pip install flower

# Start flower
celery -A lexiclass_worker.celery flower --port=5555

# Access at http://localhost:5555
```

### Logs

```bash
# Docker logs
docker compose logs -f worker

# Filter by level
docker compose logs worker | grep ERROR

# Save to file
docker compose logs worker > worker.log
```

---

## Documentation

- 📚 **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and dependencies
- 🛠️ **[Development Guide](docs/DEVELOPMENT.md)** - Setup and testing
- 🚀 **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- 📖 **[Tasks Reference](docs/TASKS_REFERENCE.md)** - Complete task API
- ⚙️ **[Configuration](docs/CONFIGURATION.md)** - All config options
- 🔧 **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues

### Additional Resources

- **[Plugin Usage Guide](PLUGIN_USAGE_GUIDE.md)** - Plugin combinations and examples
- **[Refactoring Summary](REFACTORING_SUMMARY.md)** - Recent changes and migration guide
- **[Database Session Management](docs/DATABASE_SESSION_MANAGEMENT.md)** - DB architecture

---

## Integration

### With LexiClass API

The API service submits tasks and polls for results:

```python
# In API service
from lexiclass_worker.tasks import train_field_model_task

# Submit task
task = train_field_model_task.delay(
    field_id=field_id,
    project_id=project_id,
    tokenizer=request.tokenizer,
    feature_extractor=request.feature_extractor,
    classifier=request.classifier,
    locale=request.locale
)

# Return task ID to client
return {"task_id": task.id}

# Client polls for status
task_result = AsyncResult(task_id, app=celery_app)
status = task_result.state
result = task_result.result
```

### Queue Configuration

Tasks are routed to specific queues:

- **indexing** - Document indexing tasks
- **training** - Model training tasks
- **prediction** - Prediction tasks

Queue configuration is shared via `lexiclass_core.queue_config`.

---

## Troubleshooting

### Worker Not Starting

```bash
# Check Redis connection
redis-cli ping

# Check database connection
psql $DATABASE_URI -c "SELECT 1"

# Check logs
docker compose logs worker
```

### Tasks Failing

```bash
# Check task errors
celery -A lexiclass_worker.celery inspect active

# Purge failed tasks
celery -A lexiclass_worker.celery purge

# Restart worker
docker compose restart worker
```

### Performance Issues

```bash
# Increase concurrency
celery -A lexiclass_worker.celery worker --concurrency=8

# Monitor resource usage
docker stats

# Check queue depth
redis-cli llen celery
```

For more issues, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run test suite and linters
5. Submit pull request

### Code Style

- Follow PEP 8
- Use Black for formatting
- Use isort for import sorting
- Type hints required
- Docstrings for all public APIs

### Testing Requirements

- Unit tests for all new code
- Integration tests for task workflows
- Minimum 80% code coverage
- All tests must pass

---

## License

MIT License - see LICENSE file for details.

---

## Support

- **Issues**: Report bugs via GitHub issues
- **Documentation**: See `docs/` directory
- **Examples**: See `PLUGIN_USAGE_GUIDE.md`

---

## Version History

- **v0.3.0** - Plugin system refactoring, evaluation task
- **v0.2.0** - Field-level training and prediction
- **v0.1.0** - Initial release with basic tasks

---

**LexiClass Worker** - Powering distributed document classification at scale.

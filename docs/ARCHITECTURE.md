# LexiClass Worker - Architecture

**Comprehensive architecture and design documentation**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Dependencies](#component-dependencies)
3. [LexiClass Library Integration](#lexiclass-library-integration)
4. [LexiClass-Core Integration](#lexiclass-core-integration)
5. [Task Architecture](#task-architecture)
6. [Database Architecture](#database-architecture)
7. [Storage Architecture](#storage-architecture)
8. [Queue Architecture](#queue-architecture)
9. [Plugin System](#plugin-system)
10. [Data Flow](#data-flow)
11. [Design Patterns](#design-patterns)
12. [Scalability Considerations](#scalability-considerations)

---

## System Overview

### Position in the LexiClass Ecosystem

LexiClass Worker is one of four core components in the LexiClass distributed document classification platform:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LexiClass System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │  LexiClass   │  Pure ML Library (Stateless)                  │
│  │  (Library)   │  - Tokenization, vectorization                │
│  └──────┬───────┘  - Training, prediction                       │
│         │          - 11 plugins (tokenizers, features, models)  │
│         │                                                        │
│         │ Uses                                                   │
│         ↓                                                        │
│  ┌──────────────────────────────────────────────┐              │
│  │        LexiClass Worker (This Repo)          │              │
│  │  - Celery task executor                      │              │
│  │  - Task schemas and validation               │              │
│  │  - Plugin orchestration                      │              │
│  │  - Result persistence                        │              │
│  └──────┬────────────────────────┬───────────────┘              │
│         │                        │                               │
│         │ Depends on            │ Communicates via              │
│         ↓                        ↓                               │
│  ┌──────────────┐         ┌─────────────┐                      │
│  │ LexiClass-   │         │ LexiClass-  │                      │
│  │ Core         │         │ API         │                      │
│  │              │         │             │                      │
│  │ - ORM models │         │ - REST API  │                      │
│  │ - Schemas    │         │ - Job queue │                      │
│  │ - DB session │         │ - Status    │                      │
│  └──────────────┘         └─────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

External Dependencies:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  PostgreSQL  │  │    Redis     │  │  Filesystem  │
│   (Metadata) │  │   (Queue)    │  │  (Models)    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Responsibilities

The Worker is responsible for:

1. **Task Execution**: Execute long-running ML operations asynchronously
2. **Plugin Orchestration**: Manage LexiClass plugin lifecycle
3. **Data Validation**: Validate inputs/outputs using Pydantic schemas
4. **Result Persistence**: Store results to database and filesystem
5. **Status Management**: Track and update task status
6. **Error Handling**: Handle failures with retry logic

### What the Worker Does NOT Do

- ❌ **ML Implementation**: No ML code; delegates to LexiClass library
- ❌ **Database Schema**: Schema defined in LexiClass-Core
- ❌ **API Exposure**: API handled by LexiClass-API
- ❌ **Queue Management**: Queue broker managed by Redis

---

## Component Dependencies

### Dependency Hierarchy

```
LexiClass Worker
├── Required Dependencies
│   ├── LexiClass (ML Library)
│   │   └── Provides: ML operations, plugins
│   ├── LexiClass-Core (Foundation)
│   │   └── Provides: Models, schemas, database session
│   ├── Celery
│   │   └── Provides: Task queue, worker process
│   ├── Pydantic
│   │   └── Provides: Data validation
│   └── SQLAlchemy + asyncpg
│       └── Provides: Database ORM, async operations
│
└── External Services
    ├── Redis
    │   └── Provides: Task broker, result backend
    ├── PostgreSQL
    │   └── Provides: Metadata storage
    └── Filesystem
        └── Provides: Model file storage
```

### Import Patterns

```python
# From LexiClass (ML Library)
from lexiclass.plugins import registry           # Plugin management
from lexiclass.io import DocumentLoader          # Document I/O
from lexiclass.classifier import SVMDocumentClassifier  # Legacy

# From LexiClass-Core (Foundation)
from lexiclass_core.models import Field, Model, Document  # ORM models
from lexiclass_core.db import get_db_session     # Database session
from lexiclass_core.queue_config import TASK_ROUTES  # Queue config

# Worker-specific
from lexiclass_worker.core.base import MLTaskBase  # Task base class
from lexiclass_worker.core.config import get_settings  # Config
```

---

## LexiClass Library Integration

### How the Worker Uses LexiClass

The Worker acts as a **thin orchestration layer** that calls LexiClass library:

```python
# Training Task Example
async def _train_field_model_async(...):
    # 1. Create plugins using LexiClass registry
    tokenizer_plugin = registry.create(tokenizer, locale=locale)
    feature_plugin = registry.create(feature_extractor)
    classifier_plugin = registry.create(classifier)

    # 2. Load documents (LexiClass DocumentLoader)
    all_docs = DocumentLoader.load_documents_from_directory(documents_dir)

    # 3. Execute ML pipeline
    tokenized_docs = [tokenizer_plugin.tokenize(text) for text in texts]
    feature_plugin.fit(tokenized_docs)
    X = feature_plugin.transform(tokenized_docs)
    classifier_plugin.train(X, labels)

    # 4. Save models (LexiClass plugin save methods)
    classifier_plugin.save(str(model_path))
    feature_plugin.save(str(vectorizer_path))
```

### Plugin Lifecycle

```
┌─────────────┐
│   Worker    │
│   Task      │
└──────┬──────┘
       │
       ├─ 1. registry.create(plugin_name, **params)
       │     ↓
       ├─ 2. Plugin instantiated with config
       │     ↓
       ├─ 3. Plugin methods called (fit, transform, train, predict)
       │     ↓
       ├─ 4. Plugin.save() stores to disk
       │     ↓
       └─ 5. Plugin lifecycle ends (GC)
```

### Supported Plugins

All 11 LexiClass plugins are supported:

**Tokenizers** (4):
- `icu` - Locale-aware tokenization
- `spacy` - Linguistic tokenization
- `sentencepiece` - Subword tokenization
- `huggingface` - Pre-trained tokenizers

**Feature Extractors** (4):
- `bow` - Bag-of-words
- `tfidf` - TF-IDF weighting
- `fasttext` - Word embeddings
- `sbert` - Sentence embeddings

**Classifiers** (3):
- `svm` - Linear SVM
- `xgboost` - Gradient boosting
- `transformer` - BERT/RoBERTa

---

## LexiClass-Core Integration

### Core Provides

1. **ORM Models**: Database table definitions
   ```python
   from lexiclass_core.models import (
       Field, FieldClass, DocumentLabel,
       Model, ModelStatus, Prediction
   )
   ```

2. **Database Session**: Async session management
   ```python
   from lexiclass_core.db import get_db_session, init_db

   async with get_db_session() as session:
       result = await session.execute(select(Field).where(...))
   ```

3. **Queue Configuration**: Shared between API and Worker
   ```python
   from lexiclass_core.queue_config import (
       TASK_QUEUES,  # Queue definitions
       TASK_ROUTES,  # Task routing rules
   )
   ```

### Database Session Architecture

```
┌─────────────────────────────────────────────────────┐
│              Worker Process Initialization           │
│  @worker_process_init signal                         │
│    └─> initialize_database()                         │
│          └─> init_db(DATABASE_URI)                   │
│                └─> Creates AsyncSessionFactory       │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Task Execution                       │
│  async with get_db_session() as session:            │
│      # Reuses AsyncSessionFactory                   │
│      result = await session.execute(...)            │
│      await session.commit()                         │
│  # Auto-commit on success, rollback on error        │
└─────────────────────────────────────────────────────┘
```

**Key Features**:
- **Worker-level init**: Database initialized once per worker process
- **Lazy fallback**: Auto-recovery if init fails
- **Thread-safe**: Double-check locking pattern
- **Context managers**: Automatic commit/rollback

See [DATABASE_SESSION_MANAGEMENT.md](DATABASE_SESSION_MANAGEMENT.md) for details.

---

## Task Architecture

### Task Base Class

All tasks inherit from `MLTaskBase`:

```python
class MLTaskBase(Task, ABC):
    """Abstract base for ML tasks."""

    # Properties
    @property
    @abstractmethod
    def input_schema(self) -> type[TaskInput]:
        """Pydantic schema for input validation."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[TaskOutput]:
        """Pydantic schema for output validation."""
        pass

    # Methods
    def validate_input(self, data: dict) -> TaskInput:
        """Validate and parse input."""
        return self.input_schema(**data)

    def validate_output(self, data: dict) -> TaskOutput:
        """Validate and structure output."""
        return self.output_schema(**data)

    # Lifecycle hooks
    def on_success(self, retval, task_id, args, kwargs):
        """Log success."""
        pass

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log failure and format error."""
        pass
```

### Task Implementation Pattern

```python
# 1. Define schemas
class TrainFieldModelInput(TaskInput):
    field_id: int
    project_id: int
    tokenizer: str = "icu"
    feature_extractor: str = "bow"
    classifier: str = "svm"
    locale: str = "en"

class TrainFieldModelOutput(TaskOutput):
    model_id: Optional[int] = None
    model_version: Optional[int] = None
    metrics: Optional[Dict] = None

# 2. Create task class
class TrainFieldModelTask(MLTaskBase):
    name = "lexiclass_worker.tasks.train_field_model_task"

    @property
    def input_schema(self):
        return TrainFieldModelInput

    @property
    def output_schema(self):
        return TrainFieldModelOutput

# 3. Implement async logic
async def _train_field_model_async(...):
    # Implementation using LexiClass library
    pass

# 4. Create Celery task wrapper
@app.task(base=TrainFieldModelTask, bind=True)
def train_field_model_task(self, **kwargs):
    input_data = self.validate_input(kwargs)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _train_field_model_async(...)
        )
        return self.validate_output(result).model_dump()
    finally:
        loop.close()
```

### Why This Pattern?

1. **Validation**: Pydantic catches errors early
2. **Type Safety**: Clear contracts for inputs/outputs
3. **Async Support**: Event loop for database operations
4. **Separation**: Business logic separated from Celery wrapper
5. **Testability**: Async functions can be tested independently

---

## Database Architecture

### Model Usage

Worker uses Core models but doesn't define schema:

```python
from lexiclass_core.models import (
    Field,          # Classification field definition
    FieldClass,     # Field class (e.g., "positive", "negative")
    DocumentLabel,  # Training/test labels
    Model,          # Trained model metadata
    ModelStatus,    # TRAINING, READY, FAILED
    Prediction,     # Prediction results
    Document,       # Document metadata
)
```

### Dynamic Path Generation

Model file paths are computed, not stored:

```python
class Model:
    def get_model_path(self, base_path, project_id):
        """Generate: {base}/{project}/models/{field}/v{version}/model.pkl"""
        return base_path / str(project_id) / "models" / str(self.field_id) / f"v{self.version}" / "model.pkl"

    def get_vectorizer_path(self, base_path, project_id):
        """Generate: {base}/{project}/models/{field}/v{version}/vectorizer.pkl"""
        return base_path / str(project_id) / "models" / str(self.field_id) / f"v{self.version}" / "vectorizer.pkl"
```

**Benefits**:
- No redundant path storage
- Consistent naming
- Easy version management
- Simplified migrations

### Training vs Test Data

Documents labeled with `is_training_data` flag:

```python
# Training: Get training documents
result = await session.execute(
    select(DocumentLabel)
    .where(DocumentLabel.field_id == field_id)
    .where(DocumentLabel.is_training_data == True)
)
training_labels = result.scalars().all()

# Evaluation: Get test documents
result = await session.execute(
    select(DocumentLabel)
    .where(DocumentLabel.field_id == field_id)
    .where(DocumentLabel.is_training_data == False)
)
test_labels = result.scalars().all()
```

---

## Storage Architecture

### Dual Storage Strategy

Results stored in **two places**:

#### 1. Database (PostgreSQL)
- **What**: Metadata and latest predictions
- **Models Table**: Model version, status, accuracy, metrics
- **Predictions Table**: Latest prediction per document per field
- **Why**: Fast queries, relational integrity

#### 2. Filesystem (Disk/S3)
- **What**: Serialized models and prediction histories
- **Model Files**: `model.pkl`, `vectorizer.pkl`
- **Prediction Files**: `predictions_v{N}.jsonl`
- **Why**: Large binary data, version history

### File Structure

```
STORAGE_PATH/
├── {project-id}/
│   ├── documents/
│   │   ├── {doc-id}.txt
│   │   └── ...
│   ├── models/
│   │   ├── {field-id}/
│   │   │   ├── v1/
│   │   │   │   ├── model.pkl
│   │   │   │   └── vectorizer.pkl
│   │   │   ├── v2/
│   │   │   │   ├── model.pkl
│   │   │   │   └── vectorizer.pkl
│   │   │   └── ...
│   │   └── ...
│   └── predictions/
│       ├── {field-id}/
│       │   ├── predictions_v1.jsonl
│       │   ├── predictions_v2.jsonl
│       │   └── ...
│       └── ...
└── ...
```

### Prediction Storage Example

**Database** (latest only):
```sql
SELECT document_id, class_id, confidence, model_version
FROM predictions
WHERE field_id = 123;
```

**Disk** (complete history):
```jsonl
{"document_id": 1, "class_id": 5, "confidence": 0.95, "model_version": 1}
{"document_id": 2, "class_id": 3, "confidence": 0.87, "model_version": 1}
...
```

---

## Queue Architecture

### Queue Configuration

Three dedicated queues with priority support:

```python
TASK_QUEUES = {
    'indexing': {
        'routing_key': 'indexing',
        'queue_arguments': {'x-max-priority': 10}
    },
    'training': {
        'routing_key': 'training',
        'queue_arguments': {'x-max-priority': 10}
    },
    'prediction': {
        'routing_key': 'prediction',
        'queue_arguments': {'x-max-priority': 10}
    }
}
```

### Task Routing

```python
TASK_ROUTES = {
    'lexiclass_worker.tasks.train_field_model_task': {'queue': 'training'},
    'lexiclass_worker.tasks.predict_field_documents_task': {'queue': 'prediction'},
    'lexiclass_worker.tasks.evaluate_field_model_task': {'queue': 'training'},
}
```

### Why Separate Queues?

1. **Isolation**: Long training doesn't block predictions
2. **Prioritization**: Urgent predictions processed first
3. **Scaling**: Scale prediction workers independently
4. **Monitoring**: Track queue depths separately

---

## Plugin System

### Plugin Registry Pattern

```python
from lexiclass.plugins import registry

# Create plugin
tokenizer = registry.create('spacy', model_name='en_core_web_sm')

# Check availability
registration = registry.get('sbert')
if not registration.is_available():
    missing = registration.get_missing_dependencies()
    print(f"Install: pip install {' '.join(missing)}")

# List plugins
tokenizers = registry.list_plugins(PluginType.TOKENIZER)
```

### Plugin Metadata Storage

Store plugin info in model metadata:

```python
model.metrics = {
    "tokenizer": "spacy",
    "feature_extractor": "tfidf",
    "classifier": "svm",
    "num_documents": 1000,
    "num_classes": 5,
    "num_features": 5000,
}
```

This enables:
- **Auto-detection**: Know which plugins were used
- **Consistency**: Use same plugins for prediction
- **Debugging**: Reproduce training configuration

---

## Data Flow

### Training Flow

```
1. API receives training request
   ↓
2. API submits to Redis queue
   task_id = train_field_model_task.delay(...)
   ↓
3. Worker picks task from queue
   ↓
4. Worker validates input (Pydantic)
   ↓
5. Worker loads training data from database
   SELECT * FROM document_labels WHERE is_training_data = true
   ↓
6. Worker creates plugins (LexiClass registry)
   tokenizer = registry.create('spacy')
   feature_extractor = registry.create('tfidf')
   classifier = registry.create('svm')
   ↓
7. Worker loads documents (LexiClass DocumentLoader)
   docs = DocumentLoader.load_documents_from_directory(...)
   ↓
8. Worker runs ML pipeline (LexiClass plugins)
   tokenized = tokenizer.tokenize(texts)
   feature_extractor.fit(tokenized)
   X = feature_extractor.transform(tokenized)
   classifier.train(X, labels)
   ↓
9. Worker saves models to disk (LexiClass plugin.save())
   classifier.save(model_path)
   feature_extractor.save(vectorizer_path)
   ↓
10. Worker updates database (Core models)
    model.status = READY
    model.metrics = {...}
    await session.commit()
    ↓
11. Worker returns result
    return {"status": "completed", "model_id": ...}
    ↓
12. API polls Redis for result
    result = AsyncResult(task_id).result
```

### Prediction Flow

```
1. API receives prediction request
   ↓
2. API submits to Redis queue
   ↓
3. Worker picks task
   ↓
4. Worker loads model metadata from database
   ↓
5. Worker creates plugins and loads models
   tokenizer = registry.create('spacy')
   feature_extractor = registry.create('tfidf')
   feature_extractor.load(vectorizer_path)
   classifier = registry.create('svm')
   classifier.load(model_path)
   ↓
6. Worker loads documents
   ↓
7. Worker runs prediction pipeline
   tokenized = tokenizer.tokenize(texts)
   X = feature_extractor.transform(tokenized)
   predictions, confidences = classifier.predict(X)
   ↓
8. Worker saves predictions
   - Database: Latest prediction per document
   - Disk: Complete prediction history (JSONL)
   ↓
9. Worker returns results
```

---

## Design Patterns

### 1. Protocol-Based Plugins
LexiClass uses Python protocols for plugin interfaces:
```python
class TokenizerProtocol(Protocol):
    def tokenize(self, text: str) -> list[str]: ...

class FeatureExtractorProtocol(Protocol):
    def fit(self, documents: list[list[str]]) -> None: ...
    def transform(self, documents: list[list[str]]) -> Any: ...
    def num_features(self) -> int: ...
```

### 2. Singleton Session Factory
One database session factory per worker process:
```python
_initialized = False
AsyncSessionFactory = None

def initialize_database():
    global _initialized, AsyncSessionFactory
    if not _initialized:
        init_db(DATABASE_URI)
        _initialized = True
```

### 3. Double-Check Locking
Thread-safe initialization:
```python
_init_lock = threading.Lock()

def initialize_database():
    if _initialized:  # Fast path
        return

    with _init_lock:  # Slow path
        if _initialized:  # Double-check
            return
        # Initialize
```

### 4. Async Context Managers
Automatic resource cleanup:
```python
async with get_db_session() as session:
    # Operations
    pass  # Auto-commit or rollback
```

### 5. Pydantic Validation
Input/output validation:
```python
class TrainFieldModelInput(TaskInput):
    field_id: int
    project_id: int
    tokenizer: str = "icu"

input_data = self.validate_input(kwargs)
```

---

## Scalability Considerations

### Horizontal Scaling

```bash
# Run multiple workers
celery -A lexiclass_worker.celery worker --concurrency=4
celery -A lexiclass_worker.celery worker --concurrency=4
celery -A lexiclass_worker.celery worker --concurrency=4

# Or with Docker
docker compose up --scale worker=10
```

### Queue-Based Scaling

Scale different task types independently:

```bash
# Prediction workers (more of these)
celery worker -Q prediction --concurrency=8

# Training workers (fewer, more resources)
celery worker -Q training --concurrency=2

# Indexing workers
celery worker -Q indexing --concurrency=4
```

### Performance Optimization

1. **Plugin Selection**:
   - Fast: `icu` + `bow` + `svm` (~3 min for 10K docs)
   - Balanced: `spacy` + `tfidf` + `svm` (~7 min)
   - Accurate: `spacy` + `tfidf` + `xgboost` (~15 min)

2. **Resource Limits**:
   ```yaml
   worker:
     resources:
       limits:
         memory: 4Gi
         cpu: "2"
   ```

3. **GPU Acceleration**:
   - Use for transformer models
   - 10-20x speedup for SBERT, transformers

---

## Security Considerations

### Input Validation
- All inputs validated with Pydantic
- SQL injection prevented by ORM
- File path validation prevents directory traversal

### Secrets Management
- Database credentials in environment variables
- Redis URLs not logged
- Model files access-controlled

### Resource Limits
- Task time limits (default: 1 hour)
- Memory limits in Docker
- File size limits for uploads

---

## Monitoring & Observability

### Metrics to Track
- Task processing times
- Queue depths
- Success/failure rates
- Model training accuracy
- Database connection pool usage

### Logging Strategy
- Structured JSON logging
- Correlation IDs for request tracing
- Log levels: DEBUG, INFO, WARNING, ERROR
- Separate logs per worker

### Health Checks
- Redis connectivity
- Database connectivity
- Filesystem access
- Plugin availability

---

## Summary

The LexiClass Worker is designed as a **thin orchestration layer** that:

1. ✅ **Delegates ML to LexiClass library** (no ML implementation in worker)
2. ✅ **Uses Core for persistence** (no schema definitions in worker)
3. ✅ **Validates data with Pydantic** (type-safe inputs/outputs)
4. ✅ **Manages async operations** (event loops for database)
5. ✅ **Supports all plugins** (flexible ML pipelines)
6. ✅ **Scales horizontally** (multiple workers, queue-based)

This architecture ensures clean separation of concerns, maintainability, and scalability.

---

**Last Updated**: 2025-11-02
**Version**: 0.3.0

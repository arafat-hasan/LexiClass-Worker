# LexiClass Worker

Celery worker service for executing LexiClass ML tasks. This service processes long-running machine learning jobs like model training, document indexing, and bulk prediction.

## Features

- Asynchronous task processing with Celery
- ML task execution using the LexiClass package
- Input/output validation using Pydantic schemas
- Structured logging and error handling
- Configurable storage backends
- Extensible task architecture
- Environment-based configuration
- Docker support for development and production

## Project Structure

```
src/lexiclass_worker/
├── core/                 # Core components
│   ├── base.py          # Base classes and protocols
│   ├── config.py        # Configuration management
│   ├── queue_config.py  # Shared queue configuration
│   └── storage.py       # Storage backend interface
├── tasks/               # Task implementations
│   ├── train.py         # Model training task
│   ├── index.py         # Document indexing task
│   └── predict.py       # Prediction task
├── celery.py           # Celery app configuration
└── __init__.py         # Package initialization
```

## Development Setup

### Local Development

1. Create and activate a virtual environment:

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish for Fish shell

# Upgrade pip and tools
python -m pip install --upgrade pip setuptools wheel
```

2. Install the package:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Using Docker (Recommended)

1. Prerequisites:
   - Docker
   - Docker Compose

2. Start the development environment:

```bash
# Build and start services
docker compose up --build

# View logs
docker compose logs -f worker

# Run in background
docker compose up -d
```

The development setup includes:
- Hot-reload for code changes
- Debug logging
- Redis instance
- Mounted volumes for persistence

## Production Deployment

1. Configure environment variables in a `.env` file:

```env
# Service
LEXICLASS_ENVIRONMENT=production
LEXICLASS_LOG_LEVEL=INFO
LEXICLASS_LOG_FORMAT=json

# Celery
LEXICLASS_CELERY__BROKER_URL=redis://redis:6379/0
LEXICLASS_CELERY__RESULT_BACKEND=redis://redis:6379/0
LEXICLASS_CELERY__TASK_TIME_LIMIT=3600

# Storage
LEXICLASS_STORAGE__BASE_PATH=/data
LEXICLASS_STORAGE__MODELS_DIR=models
LEXICLASS_STORAGE__INDEXES_DIR=indexes
```

2. Deploy using production configuration:

```bash
# Build and start production services
docker compose -f docker-compose.prod.yml up -d

# Scale workers if needed
docker compose -f docker-compose.prod.yml up -d --scale worker=3

# Monitor logs
docker compose -f docker-compose.prod.yml logs -f worker
```

The production setup includes:
- Resource limits and reservations
- Multiple worker replicas
- Log rotation
- Health checks
- Persistent volumes

## Container Management

### Health Checks

Monitor container health:
```bash
docker compose ps
```

### Resource Usage

View container resource usage:
```bash
docker stats
```

### Logs

View container logs:
```bash
# All logs
docker compose logs

# Specific service
docker compose logs worker

# Follow logs
docker compose logs -f worker

# Last N lines
docker compose logs --tail=100 worker
```

### Data Persistence

Data is stored in Docker volumes:
- `redis_data`: Redis data
- `ml_data`: Models and indexes

List volumes:
```bash
docker volume ls
```

Backup volumes:
```bash
docker run --rm -v ml_data:/data -v $(pwd):/backup alpine tar czf /backup/ml_data.tar.gz /data
```

## Queue Configuration

The worker uses dedicated queues for different types of tasks with priority support. The queue configuration is shared between the worker and API services through `lexiclass_worker.core.queue_config`:

```python
from lexiclass_worker.core.queue_config import TASK_QUEUES, TASK_ROUTES

# Available queues with priority support
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

# Task routing configuration
TASK_ROUTES = {
    'lexiclass_worker.tasks.index_documents_task': {'queue': 'indexing'},
    'lexiclass_worker.tasks.train_model_task': {'queue': 'training'},
    'lexiclass_worker.tasks.predict_documents_task': {'queue': 'prediction'}
}
```

This configuration ensures that:
- Each task type has a dedicated queue
- Tasks can be prioritized (0-10, higher is more important)
- Task routing is consistent across services

## Task Types

### 1. Model Training (`train_model_task`)

Trains an SVM classifier on labeled documents:

```python
from lexiclass_worker.tasks import train_model_task

result = train_model_task.delay(
    project_id="project123",
    labels_path="/path/to/labels.tsv",
    document_ids=["doc1", "doc2"]  # Optional
)
```

### 2. Document Indexing (`index_documents_task`)

Builds or updates document similarity index:

```python
from lexiclass_worker.tasks import index_documents_task

result = index_documents_task.delay(
    project_id="project123",
    documents_path="/path/to/docs",
    is_incremental=False
)
```

### 3. Prediction (`predict_documents_task`)

Runs classification on documents:

```python
from lexiclass_worker.tasks import predict_documents_task

result = predict_documents_task.delay(
    project_id="project123",
    document_ids=["doc1", "doc2"]  # Optional
)
```

## Development

### Code Quality

1. Format code:
```bash
# In Docker
docker compose exec worker black src/ tests/
docker compose exec worker isort src/ tests/

# Local
black src/ tests/
isort src/ tests/
```

2. Type checking:
```bash
# In Docker
docker compose exec worker mypy src/

# Local
mypy src/
```

### Testing

Run tests:
```bash
# In Docker
docker compose exec worker pytest

# Local
pytest
```

## Extending the Worker

### Adding New Tasks

1. Create a new module in `tasks/`:
```python
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..celery import app

class MyTaskInput(TaskInput):
    field1: str
    field2: int

class MyTaskOutput(TaskOutput):
    result: dict

class MyTask(MLTaskBase):
    name = "lexiclass_worker.tasks.my_task"
    
    @property
    def input_schema(self) -> type[TaskInput]:
        return MyTaskInput
    
    @property
    def output_schema(self) -> type[TaskOutput]:
        return MyTaskOutput

@app.task(base=MyTask, bind=True)
def my_task(self, **kwargs) -> dict:
    input_data = self.validate_input(kwargs)
    # Task implementation
    return self.validate_output(result)
```

### Custom Storage Backend

Implement the `StorageBackend` protocol:

```python
from lexiclass_worker.core.storage import StorageBackend

class MyStorage(StorageBackend):
    def save_model(self, model_path, index_path):
        # Implementation
        pass

    def load_model(self, model_path, index_path):
        # Implementation
        pass

    def delete_model(self, project_id):
        # Implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Worker not starting**:
   - Check Redis connection
   - Verify environment variables
   - Check container logs

2. **Task failures**:
   - Check worker logs
   - Verify input data
   - Check storage permissions

3. **Performance issues**:
   - Monitor resource usage
   - Check Redis memory
   - Consider scaling workers

### Monitoring

1. **Celery Flower** (optional):
Add to docker-compose:
```yaml
flower:
  image: mher/flower
  environment:
    - CELERY_BROKER_URL=redis://redis:6379/0
  ports:
    - "5555:5555"
  depends_on:
    - redis
    - worker
```

2. **Prometheus & Grafana** (recommended for production):
Consider adding monitoring stack for metrics collection.

## License

MIT
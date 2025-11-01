# LexiClass Worker - Configuration Reference

Complete reference for all configuration options.

---

## Configuration Sources

Configuration loaded from (in order of precedence):

1. Environment variables
2. `.env` file
3. Default values

---

## Environment Variables

### Database Configuration

```bash
# PostgreSQL connection URI (async driver required)
DATABASE_URI=postgresql+asyncpg://user:password@localhost:5432/lexiclass

# Connection pool settings (optional)
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_PRE_PING=true
```

### Celery Configuration

```bash
# Redis broker URL
CELERY__BROKER_URL=redis://localhost:6379/0

# Result backend URL
CELERY__RESULT_BACKEND=redis://localhost:6379/0

# Task time limits (seconds)
CELERY__TASK_TIME_LIMIT=3600              # Hard limit: 1 hour
CELERY__TASK_SOFT_TIME_LIMIT=3300         # Soft limit: 55 min

# Worker settings
CELERY__WORKER_PREFETCH_MULTIPLIER=1      # Tasks to prefetch
CELERY__WORKER_MAX_TASKS_PER_CHILD=100    # Tasks before restart
CELERY__WORKER_CONCURRENCY=4              # Number of worker processes
```

### Storage Configuration

```bash
# Base path for file storage
STORAGE__BASE_PATH=/data/lexiclass

# Subdirectories
STORAGE__MODELS_DIR=models
STORAGE__INDEXES_DIR=indexes
STORAGE__DOCUMENTS_DIR=documents
```

### Logging Configuration

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log format (text, json)
LOG_FORMAT=json

# Log file (optional)
LOG_FILE=/var/log/lexiclass-worker.log

# Library-specific log levels
LEXICLASS_LOG_LEVEL=INFO
LEXICLASS_GENSIM_LOG_LEVEL=WARNING
LEXICLASS_SKLEARN_LOG_LEVEL=WARNING
```

### LexiClass Library Settings

```bash
# Default locale for tokenization
LEXICLASS_LOCALE=en

# Random seed for reproducibility
LEXICLASS_RANDOM_SEED=42
```

---

## Configuration File (.env)

### Development Example

```bash
# .env.development
LEXICLASS_ENVIRONMENT=development

# Database
DATABASE_URI=postgresql+asyncpg://localhost/lexiclass_dev

# Redis
CELERY__BROKER_URL=redis://localhost:6379/0
CELERY__RESULT_BACKEND=redis://localhost:6379/0

# Storage
STORAGE__BASE_PATH=./data

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# LexiClass
LEXICLASS_LOG_LEVEL=DEBUG
LEXICLASS_LOCALE=en
```

### Production Example

```bash
# .env.production
LEXICLASS_ENVIRONMENT=production

# Database
DATABASE_URI=postgresql+asyncpg://user:password@db-host:5432/lexiclass
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
CELERY__BROKER_URL=redis://:password@redis-host:6379/0
CELERY__RESULT_BACKEND=redis://:password@redis-host:6379/0

# Celery
CELERY__TASK_TIME_LIMIT=7200
CELERY__WORKER_CONCURRENCY=8
CELERY__WORKER_MAX_TASKS_PER_CHILD=500

# Storage
STORAGE__BASE_PATH=/mnt/data/lexiclass

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/lexiclass-worker.log

# LexiClass
LEXICLASS_LOG_LEVEL=INFO
LEXICLASS_LOCALE=en
LEXICLASS_RANDOM_SEED=42
```

---

## Configuration Details

### Database URI Format

```
postgresql+asyncpg://[user[:password]@][host][:port][/database][?parameters]
```

**Examples**:
```bash
# Local development
postgresql+asyncpg://localhost/lexiclass_dev

# With credentials
postgresql+asyncpg://user:pass@localhost:5432/lexiclass

# Remote with SSL
postgresql+asyncpg://user:pass@db.example.com:5432/lexiclass?ssl=require

# Connection pooling
postgresql+asyncpg://user:pass@localhost/lexiclass?pool_size=20&max_overflow=40
```

### Redis URL Format

```
redis://[:password@]host[:port][/database]
```

**Examples**:
```bash
# Local without password
redis://localhost:6379/0

# With password
redis://:mypassword@localhost:6379/0

# Remote
redis://:password@redis.example.com:6379/0

# Redis Sentinel
sentinel://localhost:26379;sentinel://localhost:26380/mymaster/0
```

### Storage Paths

File structure:
```
STORAGE__BASE_PATH/
├── models/
│   └── {project_id}/
│       └── {field_id}/
│           └── v{version}/
│               ├── model.pkl
│               └── vectorizer.pkl
├── indexes/
│   └── {project_id}/
│       └── index.pkl
└── documents/
    └── {project_id}/
        └── {document_id}.txt
```

---

## Task-Specific Configuration

### Training Task

```bash
# Maximum training time (seconds)
CELERY__TASK_TIME_LIMIT=7200  # 2 hours for large datasets

# Plugin defaults
LEXICLASS_DEFAULT_TOKENIZER=icu
LEXICLASS_DEFAULT_FEATURE_EXTRACTOR=bow
LEXICLASS_DEFAULT_CLASSIFIER=svm
```

### Prediction Task

```bash
# Batch size for bulk predictions
PREDICTION_BATCH_SIZE=100

# Maximum prediction time
CELERY__TASK_TIME_LIMIT=1800  # 30 minutes
```

---

## Queue Configuration

Queues configured in `lexiclass_core.queue_config`:

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

### Custom Queue Configuration

Can override in worker start command:

```bash
celery -A lexiclass_worker.celery worker \
    --queues=training,prediction \
    --max-tasks-per-child=100 \
    --time-limit=7200
```

---

## Resource Limits

### Docker Compose

```yaml
worker:
  resources:
    limits:
      memory: 8Gi
      cpu: "4"
    reservations:
      memory: 4Gi
      cpu: "2"
```

### Kubernetes

```yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4000m"
  requests:
    memory: "4Gi"
    cpu: "2000m"
```

---

## Security Configuration

### Secrets Management

**DO NOT** commit secrets to version control.

**Use environment variables**:
```bash
# Load from secure vault
export DATABASE_URI=$(vault read -field=uri secret/database)
export CELERY__BROKER_URL=$(vault read -field=url secret/redis)
```

**Or Docker secrets**:
```yaml
services:
  worker:
    secrets:
      - db_uri
      - redis_url
    environment:
      DATABASE_URI_FILE: /run/secrets/db_uri
      REDIS_URL_FILE: /run/secrets/redis_url
```

---

## Monitoring Configuration

### Celery Flower

```bash
# Flower configuration
FLOWER_PORT=5555
FLOWER_BASIC_AUTH=user:password
FLOWER_URL_PREFIX=flower
```

Start Flower:
```bash
celery -A lexiclass_worker.celery flower \
    --port=5555 \
    --basic_auth=user:password
```

### Prometheus Metrics

```bash
# Enable Prometheus exporter
CELERY_ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
```

---

## Performance Tuning

### For High Throughput

```bash
# Increase concurrency
CELERY__WORKER_CONCURRENCY=16

# Reduce prefetch
CELERY__WORKER_PREFETCH_MULTIPLIER=1

# Increase pool
DATABASE_POOL_SIZE=40
DATABASE_MAX_OVERFLOW=80
```

### For Large Models

```bash
# Increase time limits
CELERY__TASK_TIME_LIMIT=14400  # 4 hours

# Increase memory (Docker)
# resources.limits.memory: 16Gi
```

### For GPU Acceleration

```bash
# Enable GPU for transformers
CUDA_VISIBLE_DEVICES=0,1

# PyTorch settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Troubleshooting Configuration

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG
LEXICLASS_LOG_LEVEL=DEBUG
CELERY_LOG_LEVEL=DEBUG
```

### Test Configuration

```python
# test_config.py
from lexiclass_worker.core.config import get_settings

settings = get_settings()
print(f"Database: {settings.DATABASE_URI}")
print(f"Redis: {settings.celery.broker_url}")
print(f"Storage: {settings.storage.base_path}")
```

```bash
python test_config.py
```

---

## Configuration Validation

### Startup Checks

Worker performs these checks on startup:
- ✅ Database connection
- ✅ Redis connection
- ✅ Storage path exists and writable
- ✅ Required plugins available

### Manual Validation

```bash
# Check database
psql $DATABASE_URI -c "SELECT 1"

# Check Redis
redis-cli -u $CELERY__BROKER_URL ping

# Check storage
test -w $STORAGE__BASE_PATH && echo "Writable" || echo "Not writable"
```

---

## Environment-Specific Configs

### Development

```bash
# Fast iteration
CELERY__WORKER_POOL=solo  # Single process
LOG_LEVEL=DEBUG
LOG_FORMAT=text
```

### Staging

```bash
# Production-like with debugging
CELERY__WORKER_CONCURRENCY=4
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Production

```bash
# Optimized for performance
CELERY__WORKER_CONCURRENCY=16
CELERY__WORKER_MAX_TASKS_PER_CHILD=1000
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE=/var/log/lexiclass-worker.log
```

---

## Default Values

If not specified, these defaults are used:

```python
# Database
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 20
DATABASE_POOL_PRE_PING = True

# Celery
CELERY__TASK_TIME_LIMIT = 3600
CELERY__TASK_SOFT_TIME_LIMIT = 3300
CELERY__WORKER_PREFETCH_MULTIPLIER = 1
CELERY__WORKER_MAX_TASKS_PER_CHILD = None  # Unlimited

# Storage
STORAGE__BASE_PATH = "./data"
STORAGE__MODELS_DIR = "models"
STORAGE__INDEXES_DIR = "indexes"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"

# LexiClass
LEXICLASS_LOCALE = "en"
LEXICLASS_RANDOM_SEED = 42
```

---

**Last Updated**: 2025-11-02

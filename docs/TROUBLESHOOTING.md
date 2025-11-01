# LexiClass Worker - Troubleshooting Guide

Common issues and solutions.

---

## Table of Contents

1. [Worker Issues](#worker-issues)
2. [Task Failures](#task-failures)
3. [Database Issues](#database-issues)
4. [Redis Issues](#redis-issues)
5. [Plugin Issues](#plugin-issues)
6. [Performance Issues](#performance-issues)
7. [Debugging Tools](#debugging-tools)

---

## Worker Issues

### Worker Won't Start

**Symptom**: Worker fails to start or exits immediately

**Causes**:
1. Database connection failed
2. Redis connection failed
3. Import errors
4. Configuration errors

**Solutions**:

```bash
# 1. Check logs
docker compose logs worker

# 2. Check database connection
psql $DATABASE_URI -c "SELECT 1"

# 3. Check Redis connection
redis-cli -u $CELERY__BROKER_URL ping

# 4. Verify imports
python -c "import lexiclass_worker; print('OK')"

# 5. Check configuration
python -c "from lexiclass_worker.core.config import get_settings; print(get_settings())"
```

### Worker Crashes Randomly

**Symptom**: Worker exits unexpectedly

**Causes**:
1. Out of memory
2. Task time limit exceeded
3. Unhandled exceptions

**Solutions**:

```bash
# 1. Increase memory limit (Docker)
# In docker-compose.yml:
resources:
  limits:
    memory: 16Gi

# 2. Increase task time limit
CELERY__TASK_TIME_LIMIT=7200

# 3. Enable auto-restart
CELERY__WORKER_MAX_TASKS_PER_CHILD=100

# 4. Check logs for errors
docker compose logs worker | grep ERROR
```

### Database Not Initialized Error

**Symptom**: `RuntimeError: Database not initialized`

**Cause**: Worker initialization failed

**Solution**:

```bash
# 1. Check worker startup logs
docker compose logs worker | grep "Database"

# 2. Verify DATABASE_URI is correct
echo $DATABASE_URI

# 3. Test connection
python -c "from lexiclass_core.db import init_db; init_db('$DATABASE_URI')"

# 4. Restart worker
docker compose restart worker
```

---

## Task Failures

### Training Task Fails

**Error**: `ValueError: No training labels found`

**Cause**: No documents marked with `is_training_data=True`

**Solution**:

```sql
-- Check labels
SELECT COUNT(*) FROM document_labels 
WHERE field_id = 123 AND is_training_data = true;

-- Mark documents as training data
UPDATE document_labels 
SET is_training_data = true 
WHERE field_id = 123 AND document_id IN (1, 2, 3);
```

**Error**: `ValueError: Need at least 2 labeled documents`

**Cause**: Insufficient training data

**Solution**:
- Add more labeled documents
- Check that documents exist in storage
- Verify document IDs match between database and filesystem

**Error**: `FileNotFoundError: Model file not found`

**Cause**: Storage path misconfigured

**Solution**:

```bash
# Check storage path exists
ls -la $STORAGE__BASE_PATH

# Create if missing
mkdir -p $STORAGE__BASE_PATH/models

# Fix permissions
chmod 755 $STORAGE__BASE_PATH
```

### Prediction Task Fails

**Error**: `ValueError: No ready model found`

**Cause**: No trained model available

**Solution**:

```sql
-- Check model status
SELECT id, version, status FROM models WHERE field_id = 123;

-- Train model if none exist
-- Submit training task
```

**Error**: `ValueError: Plugin mismatch`

**Cause**: Using different plugins than training

**Solution**:

```python
# Check model metadata
SELECT metrics FROM models WHERE id = 123;

# Use same plugins
result = predict_field_documents_task.delay(
    field_id=123,
    project_id=456,
    document_ids=[...],
    tokenizer=model.metrics['tokenizer'],
    feature_extractor=model.metrics['feature_extractor']
)
```

### Task Timeout

**Symptom**: `SoftTimeLimitExceeded` or `TimeLimitExceeded`

**Cause**: Task takes longer than configured limit

**Solution**:

```bash
# Increase time limit
CELERY__TASK_TIME_LIMIT=7200
CELERY__TASK_SOFT_TIME_LIMIT=6600

# Or use faster plugins
tokenizer="icu"  # instead of "spacy"
feature_extractor="bow"  # instead of "sbert"
```

---

## Database Issues

### Connection Pool Exhausted

**Error**: `QueuePool limit of size 10 overflow 20 reached`

**Cause**: Too many concurrent database connections

**Solution**:

```bash
# Increase pool size
DATABASE_POOL_SIZE=40
DATABASE_MAX_OVERFLOW=80

# Or reduce worker concurrency
CELERY__WORKER_CONCURRENCY=4

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='lexiclass';"
```

### Slow Queries

**Symptom**: Tasks take very long to complete

**Solution**:

```sql
-- Add indexes
CREATE INDEX idx_document_labels_field ON document_labels(field_id);
CREATE INDEX idx_document_labels_training ON document_labels(is_training_data);
CREATE INDEX idx_models_field_status ON models(field_id, status);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM document_labels WHERE field_id = 123;
```

### Database Deadlocks

**Error**: `deadlock detected`

**Cause**: Concurrent updates to same records

**Solution**:

```python
# Retry on deadlock
from sqlalchemy.exc import OperationalError

try:
    await session.commit()
except OperationalError as e:
    if 'deadlock detected' in str(e):
        await session.rollback()
        # Retry logic
```

---

## Redis Issues

### Cannot Connect to Redis

**Error**: `Error 111 connecting to localhost:6379. Connection refused`

**Solution**:

```bash
# 1. Check Redis is running
redis-cli ping

# 2. Start Redis
redis-server

# 3. Check connection
telnet localhost 6379

# 4. Verify URL
echo $CELERY__BROKER_URL
```

### Redis Memory Full

**Error**: `OOM command not allowed when used memory > 'maxmemory'`

**Solution**:

```bash
# Check Redis memory
redis-cli INFO memory

# Increase max memory
redis-cli CONFIG SET maxmemory 4gb

# Or clear old results
redis-cli KEYS "celery-task-meta-*" | xargs redis-cli DEL
```

### Tasks Stuck in Queue

**Symptom**: Tasks not being processed

**Solution**:

```bash
# Check queue depth
redis-cli LLEN celery

# Inspect pending tasks
redis-cli LRANGE celery 0 10

# Check workers are consuming
celery -A lexiclass_worker.celery inspect active

# Purge queue if needed
celery -A lexiclass_worker.celery purge
```

---

## Plugin Issues

### Plugin Not Available

**Error**: `PluginNotFoundError: Plugin 'sbert' dependencies not installed`

**Solution**:

```bash
# Check available plugins
python -c "from lexiclass.plugins import registry; print(registry.list_plugins())"

# Install missing dependencies
pip install sentence-transformers transformers torch

# Verify installation
python -c "from lexiclass.plugins import registry; reg = registry.get('sbert'); print(reg.is_available())"
```

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Cause**: GPU memory exhausted (transformers)

**Solution**:

```bash
# Reduce batch size
SBERT_BATCH_SIZE=16  # instead of 64

# Use smaller model
SBERT_MODEL=all-MiniLM-L6-v2  # instead of all-mpnet-base-v2

# Use CPU
CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### spaCy Model Not Found

**Error**: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution**:

```bash
# Download model
python -m spacy download en_core_web_sm

# Verify
python -c "import spacy; spacy.load('en_core_web_sm')"
```

---

## Performance Issues

### Slow Training

**Symptom**: Training takes very long

**Solutions**:

1. **Use faster plugins**:
   ```python
   tokenizer="icu"  # Faster than spacy
   feature_extractor="bow"  # Faster than tfidf
   classifier="svm"  # Faster than xgboost
   ```

2. **Reduce dataset size** (for testing):
   ```python
   # Use subset of data
   document_ids=sample(all_doc_ids, 1000)
   ```

3. **Enable GPU** (for transformers):
   ```bash
   docker run --gpus all lexiclass-worker
   ```

### High Memory Usage

**Symptom**: Worker uses too much RAM

**Solutions**:

```bash
# 1. Reduce concurrency
CELERY__WORKER_CONCURRENCY=2

# 2. Limit tasks per child
CELERY__WORKER_MAX_TASKS_PER_CHILD=50

# 3. Use lighter plugins
tokenizer="icu"
feature_extractor="bow"

# 4. Monitor memory
docker stats
```

### Tasks Piling Up

**Symptom**: Queue keeps growing

**Solutions**:

```bash
# 1. Scale workers
docker compose up --scale worker=8

# 2. Check worker health
celery -A lexiclass_worker.celery inspect active

# 3. Increase concurrency
CELERY__WORKER_CONCURRENCY=8

# 4. Use dedicated queue workers
celery worker -Q training --concurrency=2
celery worker -Q prediction --concurrency=8
```

---

## Debugging Tools

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG
LEXICLASS_LOG_LEVEL=DEBUG

# Restart worker
docker compose restart worker

# View logs
docker compose logs -f worker
```

### Inspect Task State

```bash
# Active tasks
celery -A lexiclass_worker.celery inspect active

# Registered tasks
celery -A lexiclass_worker.celery inspect registered

# Worker stats
celery -A lexiclass_worker.celery inspect stats

# Query specific task
celery -A lexiclass_worker.celery result <task-id>
```

### Python Debugger

Add to task code:

```python
import pdb; pdb.set_trace()

# Or
breakpoint()
```

Run worker with solo pool:

```bash
celery -A lexiclass_worker.celery worker --pool=solo
```

### Database Queries

```sql
-- Check recent tasks
SELECT * FROM celery_taskmeta ORDER BY date_done DESC LIMIT 10;

-- Check models
SELECT id, field_id, version, status, accuracy 
FROM models 
ORDER BY created_at DESC LIMIT 10;

-- Check predictions
SELECT field_id, COUNT(*) 
FROM predictions 
GROUP BY field_id;
```

### Health Check Script

```python
# health_check.py
import sys
from redis import Redis
from sqlalchemy import create_engine, text

def check_redis(url):
    try:
        r = Redis.from_url(url)
        r.ping()
        return True
    except Exception as e:
        print(f"Redis failed: {e}")
        return False

def check_database(uri):
    try:
        engine = create_engine(uri)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database failed: {e}")
        return False

if __name__ == "__main__":
    redis_ok = check_redis("redis://localhost:6379/0")
    db_ok = check_database("postgresql://localhost/lexiclass")

    if redis_ok and db_ok:
        print("✓ All systems operational")
        sys.exit(0)
    else:
        print("✗ System check failed")
        sys.exit(1)
```

---

## Getting Help

### Check Logs First

```bash
# Worker logs
docker compose logs -f worker

# Filter for errors
docker compose logs worker | grep -i error

# Save logs
docker compose logs worker > worker.log
```

### Collect Diagnostic Info

```bash
# System info
uname -a
python --version
docker --version

# Service status
docker compose ps

# Resource usage
docker stats

# Network connectivity
ping redis
ping postgres

# Configuration
env | grep -E "(DATABASE|CELERY|STORAGE|LOG)"
```

### Report Issue

Include:
1. Error message and full traceback
2. Relevant logs
3. Configuration (redact secrets)
4. Steps to reproduce
5. System information

---

## Common Error Messages

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| "Database not initialized" | Worker init failed | Restart worker |
| "Plugin not found" | Missing dependencies | Install plugin |
| "No training labels found" | No training data | Add labels |
| "Connection refused" | Service not running | Start service |
| "Out of memory" | Insufficient RAM | Increase memory |
| "Task timeout" | Task too slow | Increase limit |
| "Permission denied" | File permissions | Fix permissions |

---

**Last Updated**: 2025-11-02

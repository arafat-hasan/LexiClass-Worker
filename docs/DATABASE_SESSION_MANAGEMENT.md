# Database Session Management Architecture

## Overview

This document describes the database session management strategy used in the LexiClass Worker, including design patterns, initialization flow, and best practices.

---

## Problem Statement

**Challenge**: Celery workers are synchronous, but we need async database sessions.

**Complications**:
- Worker processes start synchronously
- Tasks create event loops for async operations
- Database session factory must be initialized before tasks run
- Must handle multi-process worker pools
- Need thread-safe initialization

**Previous Issue**: Database wasn't initialized, causing `RuntimeError: Database not initialized. Call init_db() first.`

---

## Solution Architecture

### Design Patterns Applied

#### 1. **Singleton Pattern**
- One `AsyncSessionFactory` per worker process
- Shared across all tasks in that process
- Initialized once during worker startup

#### 2. **Lazy Initialization with Double-Check Locking**
- Fallback if worker initialization fails
- Thread-safe using `threading.Lock`
- Fast path: check without lock
- Slow path: acquire lock and re-check

#### 3. **Worker Lifecycle Hooks**
- Use Celery signals for initialization
- `worker_process_init` - Initialize database when worker process starts
- `worker_ready` - Verify connections before accepting tasks

#### 4. **Dependency Injection**
- Explicit `ensure_db_initialized()` calls in database functions
- Safety net for robustness
- Clear initialization requirements

---

## Implementation Details

### 1. Worker Initialization (`celery.py`)

```python
@worker_process_init.connect
def init_worker_process(**kwargs):
    """Initialize worker process - runs once per worker process."""
    try:
        from .core.database import initialize_database
        initialize_database()
        logger.info("✓ Database session factory initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {str(e)}", exc_info=True)
        # Don't raise - let tasks handle database errors individually
```

**When it runs:**
- Once per worker process (for prefork pool)
- Before any tasks are executed
- In the main worker process context

**Benefits:**
- Database ready before first task
- Shared session factory across tasks
- Handles worker restart automatically

---

### 2. Idempotent Initialization (`database.py`)

```python
_init_lock = threading.Lock()
_initialized = False

def initialize_database() -> None:
    """Initialize database - idempotent and thread-safe."""
    global _initialized

    # Fast path: already initialized
    if _initialized and session_module.AsyncSessionFactory is not None:
        return

    # Slow path: need to initialize
    with _init_lock:
        # Double-check after acquiring lock
        if _initialized and session_module.AsyncSessionFactory is not None:
            return

        settings = get_settings()
        init_db(str(settings.DATABASE_URI))
        _initialized = True
```

**Pattern**: Double-Check Locking
- First check without lock (fast)
- Acquire lock if needed (slow)
- Re-check after lock (safety)

**Benefits:**
- Thread-safe initialization
- Minimal lock contention
- Safe to call multiple times
- Fast for subsequent calls

---

### 3. Lazy Initialization Fallback

```python
def ensure_db_initialized() -> None:
    """Ensure database is initialized (lazy initialization fallback)."""
    if session_module.AsyncSessionFactory is None:
        logger.warning("Database not initialized - attempting lazy initialization")
        try:
            initialize_database()
        except Exception as e:
            raise RuntimeError(f"Lazy database initialization failed: {str(e)}") from e
```

**When it's used:**
- Called at the start of every database function
- Provides safety net if worker init failed
- Logs warning to detect configuration issues

**Benefits:**
- Robustness - handles edge cases
- Debugging - warns about initialization problems
- Graceful degradation

---

### 4. Database Function Pattern

All async database functions follow this pattern:

```python
async def update_indexing_status(...) -> None:
    """Update status in database."""
    # 1. Ensure database is initialized
    ensure_db_initialized()

    # 2. Use database session
    async with get_db_session() as session:
        # ... database operations ...
        pass
    # 3. Auto-commit via context manager
```

---

## Initialization Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Worker Process Starts                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          worker_process_init signal fires                    │
│  - Calls initialize_database()                              │
│  - Creates AsyncSessionFactory (Singleton)                  │
│  - Sets _initialized = True                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            worker_ready signal fires                         │
│  - Verifies AsyncSessionFactory is not None                 │
│  - Logs connection status                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Worker Ready for Tasks                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Task Received & Started                        │
│  1. Task creates event loop (if needed)                     │
│  2. Calls database function                                 │
│  3. Database function calls ensure_db_initialized()         │
│  4. Check: Is AsyncSessionFactory initialized?              │
│     ├─ YES → Use existing session factory                   │
│     └─ NO  → Lazy initialization (with warning)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Thread Safety

### Multi-Process Workers (Prefork Pool)

- Each worker process has its own `AsyncSessionFactory`
- No shared memory between processes
- Each process initializes independently
- Process-level singleton pattern

### Multi-Threading in Tasks

- `_init_lock` protects initialization code
- Double-check locking prevents race conditions
- Once initialized, no lock needed (fast path)
- Safe for concurrent task execution

---

## Error Handling

### Initialization Failure

**During worker startup:**
```python
# Error is logged but doesn't crash worker
# Tasks will attempt lazy initialization
logger.error("✗ Failed to initialize database: {error}")
```

**During task execution:**
```python
# Error propagates to task
# Celery handles retry logic
raise RuntimeError("Lazy database initialization failed")
```

### Session Errors

All database functions use context managers:
```python
async with get_db_session() as session:
    # Operations
    pass  # Auto-commit on success
# Auto-rollback on exception
```

---

## Best Practices

### ✓ DO

1. **Call `ensure_db_initialized()` in database functions**
   - Provides safety net
   - Self-documenting code
   - Handles edge cases

2. **Use async context managers for sessions**
   ```python
   async with get_db_session() as session:
       # Your code here
   ```

3. **Let errors propagate to Celery**
   - Don't catch and hide database errors
   - Celery handles retries
   - Monitoring systems can alert

4. **Log initialization status**
   - Info: successful initialization
   - Warning: lazy initialization triggered
   - Error: initialization failures

### ✗ DON'T

1. **Don't initialize database in tasks**
   - Worker init does this
   - Creates unnecessary overhead
   - Can cause race conditions

2. **Don't store session references**
   ```python
   # Bad: Don't do this
   self.session = await get_db_session()

   # Good: Use context manager
   async with get_db_session() as session:
       # Use session
   ```

3. **Don't create new session factories**
   - Use the singleton
   - Don't call `init_db()` directly in tasks

4. **Don't ignore initialization warnings**
   - If you see lazy init warnings, fix worker config
   - Indicates worker startup problem

---

## Monitoring & Debugging

### Initialization Logs

**Successful startup:**
```json
{"level": "INFO", "message": "Initializing worker process..."}
{"level": "INFO", "message": "Database session factory initialized successfully"}
{"level": "INFO", "message": "✓ Database session factory initialized"}
```

**Lazy initialization (indicates problem):**
```json
{"level": "WARNING", "message": "Database not initialized - attempting lazy initialization"}
```

**Failure:**
```json
{"level": "ERROR", "message": "Failed to initialize database: <error>"}
```

### Health Checks

Check database status via logs:
```bash
# Look for initialization messages
grep "Database session factory" worker.log

# Check for lazy initialization warnings
grep "lazy initialization" worker.log
```

---

## Testing

### Unit Tests

Mock the session factory:
```python
@pytest.fixture
def mock_db_session():
    from lexiclass_core.db import session as session_module
    session_module.AsyncSessionFactory = MockFactory()
    yield
    session_module.AsyncSessionFactory = None
```

### Integration Tests

Use real database with test fixtures:
```python
@pytest.fixture
async def test_db():
    initialize_database()  # Real initialization
    yield
    # Cleanup
```

---

## Migration Guide

If upgrading from old code:

1. **Remove manual `init_db()` calls from tasks**
   - Worker handles this now

2. **Add `ensure_db_initialized()` to database functions**
   - Safety net for robustness

3. **Update worker startup scripts**
   - Verify signals are connected
   - Check logs for initialization

4. **Test with multiple workers**
   - Verify each process initializes correctly
   - No race conditions

---

## Performance Considerations

### Initialization Cost
- **Worker startup**: One-time cost (~10-100ms)
- **Task execution**: No overhead (fast path check)
- **Lazy init**: Only if worker init failed

### Session Pooling
- SQLAlchemy manages connection pool
- Pool size: 10 connections (configurable)
- Max overflow: 20 connections
- Pre-ping: Ensures connection health

### Fast Path Optimization
```python
# Fast path: no lock needed
if _initialized and session_module.AsyncSessionFactory is not None:
    return  # <1μs
```

---

## Troubleshooting

### Issue: "Database not initialized" error

**Symptoms:**
```
RuntimeError: Database not initialized. Call init_db() first.
```

**Causes:**
1. Worker initialization failed
2. Celery signals not connected
3. Import error in initialization

**Solutions:**
1. Check worker startup logs
2. Verify `DATABASE_URI` environment variable
3. Check database connectivity
4. Review `worker_process_init` signal

---

### Issue: Lazy initialization warnings

**Symptoms:**
```
WARNING: Database not initialized - attempting lazy initialization
```

**Cause:**
Worker init didn't run or failed silently

**Solution:**
1. Check worker startup logs for errors
2. Verify Celery signals are registered
3. Check `DATABASE_URI` configuration
4. Review worker startup script

---

### Issue: Connection pool exhausted

**Symptoms:**
```
QueuePool limit of size 10 overflow 20 reached
```

**Solutions:**
1. Increase pool size in `init_db()`
2. Verify sessions are properly closed (use context managers)
3. Check for long-running tasks holding connections
4. Add more worker processes

---

## References

- **Singleton Pattern**: Single instance per process
- **Double-Check Locking**: Thread-safe lazy initialization
- **Celery Signals**: Worker lifecycle hooks
- **Context Managers**: Proper resource cleanup
- **Connection Pooling**: SQLAlchemy pool management

---

## Changelog

### 2025-11-01: Initial Implementation
- Added worker-level database initialization
- Implemented lazy initialization fallback
- Added thread-safe singleton pattern
- Added comprehensive logging
- Fixed "Database not initialized" bug

---

**Maintained by**: LexiClass Development Team
**Last Updated**: 2025-11-01
**Version**: 1.0

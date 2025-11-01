# LexiClass Worker Tests

This directory contains integration tests for the LexiClass Worker, focusing on indexing status updates.

## Test Files

### `test_indexing_status.py`
Unit and integration tests for the indexing status update functions:
- `update_indexing_status()` - Main status update function
- `get_project_index_status()` - Status retrieval
- Tests for success, failure, and idempotency scenarios
- Backward compatibility tests

### `test_index_task_integration.py`
End-to-end integration tests for the indexing task:
- Full indexing workflow with status updates
- Success scenario tests
- Failure scenario tests (0 features, exceptions)
- Status verification at document and project levels

## Running Tests

### Prerequisites

Install test dependencies:

```bash
# Using pip
pip install -e ".[dev]"

# Or using your virtual environment
source .venv/bin/activate
pip install pytest pytest-asyncio aiosqlite
```

### Run All Tests

```bash
# From the Worker directory
pytest tests/ -v

# Or with coverage
pytest tests/ -v --cov=lexiclass_worker --cov-report=html
```

### Run Specific Test Files

```bash
# Run only status update tests
pytest tests/test_indexing_status.py -v

# Run only task integration tests
pytest tests/test_index_task_integration.py -v
```

### Run Specific Test Cases

```bash
# Run a specific test function
pytest tests/test_indexing_status.py::TestIndexingStatusUpdates::test_successful_indexing_updates_status -v

# Run all tests in a class
pytest tests/test_indexing_status.py::TestIndexingStatusUpdates -v
```

## Test Coverage

The tests cover:

1. **Success Scenarios**
   - Documents updated to INDEXED status
   - Project updated to INDEXED status
   - last_indexed_at timestamp set correctly

2. **Failure Scenarios**
   - Documents updated to FAILED status
   - Project updated to FAILED status
   - Error messages captured and logged

3. **Idempotency**
   - Re-running indexing on already-indexed project
   - Timestamp preservation on idempotent updates

4. **Edge Cases**
   - Projects with no documents
   - Partial document updates
   - Status transitions from IN_PROGRESS

5. **Atomicity**
   - Transaction rollback on errors
   - All-or-nothing updates

6. **Backward Compatibility**
   - Old status update functions still work
   - Enum/string value compatibility

## Test Database

Tests use an in-memory SQLite database for speed and isolation:
- Each test gets a fresh database
- No external dependencies required
- Automatic cleanup after tests

## Design Patterns Tested

The tests verify the implementation of:
- **State Pattern**: Proper state transitions (PENDING → INDEXED/FAILED)
- **Command Pattern**: Encapsulated status update operations
- **Transaction Script**: Atomic database updates

## Debugging Tests

Run tests with detailed output:

```bash
# Show print statements and logs
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Drop into debugger on failure
pytest tests/ -v --pdb
```

## Continuous Integration

These tests should be run in CI/CD pipelines before:
- Merging pull requests
- Deploying to staging/production
- Releasing new versions

Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ -v --cov=lexiclass_worker
```

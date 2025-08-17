# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev]"

# Copy application code
COPY src/ ./src/

# Create directories for models and indexes
RUN mkdir -p /data/models /data/indexes \
    && chmod -R 777 /data

# Set default environment variables
ENV LEXICLASS_STORAGE__BASE_PATH=/data \
    LEXICLASS_STORAGE__MODELS_DIR=models \
    LEXICLASS_STORAGE__INDEXES_DIR=indexes \
    LEXICLASS_LOG_FORMAT=json \
    LEXICLASS_LOG_LEVEL=INFO

# Create non-root user
RUN useradd -m -u 1000 worker \
    && chown -R worker:worker /app /data

USER worker

# Command to run the worker
CMD ["celery", "-A", "lexiclass_worker.celery", "worker", "-l", "INFO", "-Q", "ml_tasks"]

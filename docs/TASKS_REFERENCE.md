# LexiClass Worker - Tasks Reference

Complete API reference for all worker tasks.

---

## Available Tasks

1. [train_field_model_task](#train_field_model_task)
2. [predict_field_documents_task](#predict_field_documents_task)
3. [evaluate_field_model_task](#evaluate_field_model_task)

---

## train_field_model_task

Train a classification model for a specific field using labeled documents.

### Import

```python
from lexiclass_worker.tasks import train_field_model_task
```

### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `field_id` | int | ✅ | - | Field ID to train |
| `project_id` | int | ✅ | - | Project ID |
| `tokenizer` | str | ❌ | `"icu"` | Tokenizer plugin (icu, spacy, sentencepiece, huggingface) |
| `feature_extractor` | str | ❌ | `"bow"` | Feature extractor plugin (bow, tfidf, fasttext, sbert) |
| `classifier` | str | ❌ | `"svm"` | Classifier plugin (svm, xgboost, transformer) |
| `locale` | str | ❌ | `"en"` | Locale for tokenization |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | Task status (`completed`, `failed`) |
| `project_id` | int | Project ID |
| `model_id` | int | Created model ID |
| `model_version` | int | Model version number |
| `metrics` | dict | Training metrics |

### Example Usage

```python
# Basic usage (default plugins)
result = train_field_model_task.delay(
    field_id=123,
    project_id=456
)

# With custom plugins
result = train_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",
    feature_extractor="tfidf",
    classifier="svm",
    locale="en"
)

# Check status
print(result.state)  # PENDING, STARTED, SUCCESS, FAILURE

# Get result (blocks)
result_data = result.get(timeout=600)
print(result_data)
# {
#   "status": "completed",
#   "project_id": 456,
#   "model_id": 789,
#   "model_version": 1,
#   "metrics": {
#     "num_documents": 1000,
#     "num_classes": 5,
#     "num_features": 5000,
#     "tokenizer": "spacy",
#     "feature_extractor": "tfidf",
#     "classifier": "svm"
#   }
# }
```

### Behavior

1. Loads documents with `is_training_data=True`
2. Creates specified plugins
3. Tokenizes documents
4. Fits feature extractor
5. Trains classifier
6. Saves model files to disk
7. Updates database with model metadata
8. Returns model ID and metrics

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Field {id} not found` | Field doesn't exist | Check field_id |
| `ValueError: No training labels found` | No is_training_data=True labels | Add training labels |
| `ValueError: Need at least 2 labeled documents` | Insufficient training data | Add more labels |
| `PluginNotFoundError` | Plugin dependencies not installed | Install plugin dependencies |

### Notes

- Training data must be marked with `is_training_data=True`
- Minimum 2 labeled documents required
- Model files saved to `{base_path}/{project_id}/models/{field_id}/v{version}/`
- Automatic version increment

---

## predict_field_documents_task

Run predictions on documents using a trained model.

### Import

```python
from lexiclass_worker.tasks import predict_field_documents_task
```

### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `field_id` | int | ✅ | - | Field ID for prediction |
| `project_id` | int | ✅ | - | Project ID |
| `document_ids` | List[int] | ✅ | - | Document IDs to predict |
| `tokenizer` | str | ❌ | `"icu"` | Tokenizer (must match training) |
| `feature_extractor` | str | ❌ | `"bow"` | Feature extractor (must match training) |
| `locale` | str | ❌ | `"en"` | Locale (must match training) |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | Task status |
| `project_id` | int | Project ID |
| `field_id` | int | Field ID |
| `model_version` | int | Model version used |
| `predictions_created` | int | Number of predictions created |
| `total_documents` | int | Total documents processed |

### Example Usage

```python
# Predict specific documents
result = predict_field_documents_task.delay(
    field_id=123,
    project_id=456,
    document_ids=[789, 790, 791],
    tokenizer="spacy",
    feature_extractor="tfidf",
    locale="en"
)

# Get result
result_data = result.get(timeout=300)
print(result_data)
# {
#   "status": "completed",
#   "project_id": 456,
#   "field_id": 123,
#   "model_version": 2,
#   "predictions_created": 3,
#   "total_documents": 3
# }
```

### Behavior

1. Loads latest READY model for field
2. Creates plugins matching training
3. Loads model files
4. Loads requested documents
5. Runs prediction pipeline
6. Saves predictions to database and disk
7. Returns prediction statistics

### Storage

- **Database**: Latest prediction per document (fast access)
- **Disk**: Complete prediction history in JSONL

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Field {id} not found` | Field doesn't exist | Check field_id |
| `ValueError: No ready model found` | No trained model | Train model first |
| `ValueError: Model file not found` | Model files missing | Check storage |
| `ValueError: No valid documents found` | Documents don't exist | Check document_ids |

### Notes

- Uses latest READY model
- Plugins must match training configuration
- Predictions overwrite existing ones in database
- Complete history saved to disk

---

## evaluate_field_model_task

Evaluate a trained model on test data.

### Import

```python
from lexiclass_worker.tasks import evaluate_field_model_task
```

### Input Schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `field_id` | int | ✅ | - | Field ID to evaluate |
| `project_id` | int | ✅ | - | Project ID |
| `tokenizer` | str | ❌ | `"icu"` | Tokenizer (must match training) |
| `feature_extractor` | str | ❌ | `"bow"` | Feature extractor (must match training) |
| `locale` | str | ❌ | `"en"` | Locale (must match training) |

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | Task status |
| `project_id` | int | Project ID |
| `field_id` | int | Field ID |
| `model_version` | int | Model version evaluated |
| `metrics` | dict | Evaluation metrics |
| `num_test_documents` | int | Number of test documents |

### Example Usage

```python
# Evaluate model
result = evaluate_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",
    feature_extractor="tfidf",
    locale="en"
)

# Get result
result_data = result.get(timeout=300)
print(result_data)
# {
#   "status": "completed",
#   "project_id": 456,
#   "field_id": 123,
#   "model_version": 2,
#   "num_test_documents": 200,
#   "metrics": {
#     "accuracy": 0.9245,
#     "precision": 0.9183,
#     "recall": 0.9267,
#     "f1_score": 0.9225,
#     "per_class_metrics": {
#       "class_a": {"precision": 0.95, "recall": 0.93, "f1-score": 0.94},
#       "class_b": {"precision": 0.89, "recall": 0.92, "f1-score": 0.90}
#     }
#   }
# }
```

### Behavior

1. Loads documents with `is_training_data=False`
2. Loads latest READY model
3. Creates plugins matching training
4. Runs predictions on test set
5. Calculates metrics
6. Updates model with evaluation results
7. Returns comprehensive metrics

### Metrics Calculated

- **Accuracy**: Overall prediction accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1 Score**: Weighted F1 across classes
- **Per-class Metrics**: Precision, recall, F1 per class

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: No test labels found` | No is_training_data=False labels | Add test labels |
| `ValueError: Need at least 2 test documents` | Insufficient test data | Add more test labels |

### Notes

- Test data must be marked with `is_training_data=False`
- Separate from training data (proper ML practice)
- Results stored in `model.metrics["evaluation"]`
- Accuracy stored in `model.accuracy`

---

## Common Patterns

### Async Task Execution

```python
# Submit task
task = train_field_model_task.delay(...)

# Don't block, return task ID to client
return {"task_id": task.id}

# Client polls for status
from celery.result import AsyncResult
result = AsyncResult(task_id)
status = result.state
if status == "SUCCESS":
    data = result.result
```

### Error Handling

```python
try:
    result = task.get(timeout=600)
except TimeoutError:
    print("Task timeout")
except Exception as e:
    print(f"Task failed: {e}")
```

### Celery Result Backend

```python
# Get task state
result.state  # PENDING, STARTED, SUCCESS, FAILURE

# Get result (blocks)
result.get(timeout=600)

# Get result (non-blocking)
if result.ready():
    data = result.result

# Check if successful
if result.successful():
    data = result.result

# Check if failed
if result.failed():
    print(result.traceback)
```

---

## Plugin Compatibility

| Tokenizer | Feature Extractor | Classifier | Speed | Accuracy |
|-----------|------------------|------------|-------|----------|
| icu | bow | svm | ⚡⚡⚡ | ~88% |
| spacy | tfidf | svm | ⚡⚡⚡ | ~92% |
| spacy | tfidf | xgboost | ⚡⚡ | ~94% |
| huggingface | sbert | transformer | ⚡ | ~96% |

**Important**: Prediction and evaluation must use same plugins as training.

---

## Task Queues

| Task | Queue | Priority Support |
|------|-------|-----------------|
| train_field_model_task | training | ✅ (0-10) |
| predict_field_documents_task | prediction | ✅ (0-10) |
| evaluate_field_model_task | training | ✅ (0-10) |

### Priority Example

```python
# High priority task
result = train_field_model_task.apply_async(
    kwargs={...},
    priority=9
)

# Low priority task
result = train_field_model_task.apply_async(
    kwargs={...},
    priority=1
)
```

---

**Last Updated**: 2025-11-02

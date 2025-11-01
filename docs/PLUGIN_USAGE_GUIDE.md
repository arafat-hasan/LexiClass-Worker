# LexiClass Worker - Plugin Usage Guide

Quick reference for using the plugin system in LexiClass Worker tasks.

---

## Task Overview

| Task | Purpose | Uses Training Data | Uses Test Data |
|------|---------|-------------------|----------------|
| `train_field_model_task` | Train classification model | ✅ (is_training_data=True) | ❌ |
| `predict_field_documents_task` | Predict document classes | ❌ | ❌ |
| `evaluate_field_model_task` | Evaluate model performance | ❌ | ✅ (is_training_data=False) |

---

## Plugin Parameters

### Common Parameters (All Tasks)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | string | `"icu"` | Tokenizer plugin name |
| `feature_extractor` | string | `"bow"` | Feature extractor plugin name |
| `locale` | string | `"en"` | Locale for tokenization |

### Training-Only Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier` | string | `"svm"` | Classifier plugin name |

---

## Example Task Submissions

### 1. Training with Default Plugins (Fast Baseline)

```python
# Using defaults: icu tokenizer, bow features, svm classifier
task_result = train_field_model_task.delay(
    field_id=123,
    project_id=456
)
```

**Expected Performance:**
- Speed: ⚡⚡⚡ Very Fast
- Accuracy: ~88-90%
- Use Case: Quick prototyping, large datasets

---

### 2. Training with Better Quality (Production)

```python
# Using spacy tokenizer and tfidf features
task_result = train_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",
    feature_extractor="tfidf",
    classifier="svm"
)
```

**Expected Performance:**
- Speed: ⚡⚡⚡ Fast
- Accuracy: ~92-94%
- Use Case: Production systems, balanced performance

---

### 3. Training with XGBoost (High Accuracy)

```python
# Using spacy tokenizer, tfidf features, and xgboost
task_result = train_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",
    feature_extractor="tfidf",
    classifier="xgboost"
)
```

**Expected Performance:**
- Speed: ⚡⚡ Medium
- Accuracy: ~93-95%
- Use Case: High accuracy requirements

---

### 4. Training with Transformers (State-of-the-Art)

```python
# Using transformer embeddings (requires GPU for reasonable speed)
task_result = train_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="huggingface",
    feature_extractor="sbert",
    classifier="transformer"
)
```

**Expected Performance:**
- Speed: ⚡ Slow (20+ min without GPU)
- Accuracy: ~95-97%
- Use Case: Research, maximum accuracy needed
- **Requires**: GPU acceleration

---

### 5. Prediction Task

```python
# Must use SAME plugins as training!
task_result = predict_field_documents_task.delay(
    field_id=123,
    project_id=456,
    document_ids=[789, 790, 791],
    tokenizer="spacy",           # Match training
    feature_extractor="tfidf",   # Match training
    locale="en"                  # Match training
)
```

**Important**: Plugin parameters must match those used during training.

---

### 6. Evaluation Task

```python
# Evaluate on test set (is_training_data=False)
task_result = evaluate_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",           # Match training
    feature_extractor="tfidf",   # Match training
    locale="en"                  # Match training
)
```

**Returns**:
```json
{
  "status": "completed",
  "field_id": 123,
  "model_version": 3,
  "metrics": {
    "accuracy": 0.9245,
    "precision": 0.9183,
    "recall": 0.9267,
    "f1_score": 0.9225,
    "num_test_documents": 500,
    "per_class_metrics": {
      "class_a": {"precision": 0.95, "recall": 0.93, "f1-score": 0.94},
      "class_b": {"precision": 0.89, "recall": 0.92, "f1-score": 0.90}
    }
  }
}
```

---

## Plugin Combinations Cheat Sheet

### By Speed (Fastest to Slowest)

1. **Fastest** (2-5 min for 10K docs)
   - `tokenizer="icu"`, `feature_extractor="bow"`, `classifier="svm"`

2. **Fast** (5-10 min for 10K docs)
   - `tokenizer="spacy"`, `feature_extractor="tfidf"`, `classifier="svm"`

3. **Medium** (10-20 min for 10K docs)
   - `tokenizer="spacy"`, `feature_extractor="tfidf"`, `classifier="xgboost"`

4. **Slow** (30+ min for 10K docs, GPU recommended)
   - `tokenizer="huggingface"`, `feature_extractor="sbert"`, `classifier="transformer"`

---

### By Accuracy (Good to Best)

1. **Good** (~88-90%)
   - `tokenizer="icu"`, `feature_extractor="bow"`, `classifier="svm"`

2. **Better** (~92-94%)
   - `tokenizer="spacy"`, `feature_extractor="tfidf"`, `classifier="svm"`

3. **Best** (~93-95%)
   - `tokenizer="spacy"`, `feature_extractor="tfidf"`, `classifier="xgboost"`

4. **State-of-the-Art** (~95-97%)
   - `tokenizer="huggingface"`, `feature_extractor="sbert"`, `classifier="transformer"`

---

### By Use Case

#### Quick Prototyping
```python
tokenizer="icu"
feature_extractor="bow"
classifier="svm"
```

#### Production Systems
```python
tokenizer="spacy"
feature_extractor="tfidf"
classifier="svm"  # or "xgboost" for higher accuracy
```

#### High-Accuracy Requirements
```python
tokenizer="spacy"
feature_extractor="fasttext"
classifier="xgboost"
```

#### Research / Maximum Accuracy
```python
tokenizer="huggingface"
feature_extractor="sbert"
classifier="transformer"
```

---

## Multi-Language Support

### English (Default)
```python
tokenizer="icu"  # or "spacy"
locale="en"
```

### Spanish
```python
tokenizer="icu"
locale="es"
```

### French
```python
tokenizer="icu"
locale="fr"
```

### Any Language (Universal)
```python
tokenizer="sentencepiece"  # Train on your corpus
feature_extractor="sbert"  # Multilingual models available
```

---

## Common Issues & Solutions

### Issue: "Plugin not available"

**Error:**
```
PluginNotFoundError: Plugin 'sbert' dependencies not installed
```

**Solution:**
Install required dependencies:
```bash
pip install sentence-transformers transformers torch
```

---

### Issue: "Plugin mismatch during prediction"

**Error:**
```
ValueError: Feature dimensions don't match
```

**Cause:** Using different plugins for prediction than training

**Solution:**
- Always use the same `tokenizer` and `feature_extractor` for prediction/evaluation as used in training
- Check model metadata in database to see which plugins were used
- The API should auto-populate these parameters from model metadata

---

### Issue: "Out of memory"

**Cause:** Using memory-intensive plugins (sbert, transformer) on large datasets

**Solutions:**
1. Use lighter plugins (icu + bow + svm)
2. Reduce batch size (if using transformers)
3. Use GPU for transformer models
4. Process data in smaller batches

---

## Best Practices

### 1. Start Simple, Then Optimize
```python
# Step 1: Baseline (icu + bow + svm)
# Step 2: Better quality (spacy + tfidf + svm)
# Step 3: Higher accuracy (spacy + tfidf + xgboost)
# Step 4: Maximum accuracy (huggingface + sbert + transformer)
```

### 2. Match Plugins Between Training and Prediction
```python
# Training
train_field_model_task.delay(
    ...,
    tokenizer="spacy",
    feature_extractor="tfidf"
)

# Prediction (MUST match!)
predict_field_documents_task.delay(
    ...,
    tokenizer="spacy",
    feature_extractor="tfidf"
)
```

### 3. Always Evaluate on Test Data
```python
# After training, run evaluation
evaluate_field_model_task.delay(
    field_id=123,
    project_id=456,
    tokenizer="spacy",
    feature_extractor="tfidf"
)
```

### 4. Store Plugin Metadata with Models
The worker automatically stores plugin info in `model.metrics`:
```json
{
  "tokenizer": "spacy",
  "feature_extractor": "tfidf",
  "classifier": "svm",
  ...
}
```

Use this metadata to auto-populate prediction parameters in the API.

---

## Performance Tips

### GPU Acceleration
For transformer models, ensure GPU is available:
```python
# The worker will automatically use GPU if available
# Check logs for: "Using device: cuda"
```

### Memory Optimization
For large datasets:
```python
# Use simpler plugins
tokenizer="icu"        # Instead of spacy
feature_extractor="bow" # Instead of sbert
```

### Speed vs Accuracy Trade-off
| Priority | Recommended Combo | Time (10K docs) | Accuracy |
|----------|------------------|-----------------|----------|
| Speed | icu + bow + svm | ~3 min | 88% |
| Balanced | spacy + tfidf + svm | ~7 min | 92% |
| Accuracy | spacy + tfidf + xgboost | ~15 min | 94% |
| Maximum | huggingface + sbert + transformer | ~30 min (GPU) | 96% |

---

## Quick Reference: Plugin Names

### Tokenizers
- `icu` - Fast, locale-aware (default)
- `spacy` - Linguistic features
- `sentencepiece` - Subword tokenization
- `huggingface` - Pre-trained tokenizers

### Feature Extractors
- `bow` - Bag-of-words (default)
- `tfidf` - TF-IDF weighting
- `fasttext` - Word embeddings
- `sbert` - Sentence embeddings

### Classifiers
- `svm` - Linear SVM (default)
- `xgboost` - Gradient boosting
- `transformer` - BERT/RoBERTa

---

**Need more help?** See `REFACTORING_SUMMARY.md` for detailed implementation notes.

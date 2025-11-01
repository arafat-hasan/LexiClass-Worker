# LexiClass Worker Refactoring Summary

**Date**: 2025-11-02
**Purpose**: Refactor worker tasks to properly use LexiClass library with plugin support

---

## Overview

This refactoring updates the LexiClass Worker to properly utilize the LexiClass ML library's plugin architecture instead of implementing ML logic locally. The changes ensure proper separation of concerns and enable flexibility in choosing tokenizers, feature extractors, and classifiers.

---

## Key Changes

### 1. **Field Training Task** (`src/lexiclass_worker/tasks/field_train.py`)

#### Schema Updates
- **Added plugin parameters** to `TrainFieldModelInput`:
  - `tokenizer` (default: "icu") - Tokenizer plugin name
  - `feature_extractor` (default: "bow") - Feature extractor plugin name
  - `classifier` (default: "svm") - Classifier plugin name
  - `locale` (default: "en") - Locale for tokenization

#### Implementation Changes
- **Replaced** `TextClassificationPipeline` with direct plugin usage
- **Now uses** `lexiclass.plugins.registry` to create plugins:
  ```python
  tokenizer_plugin = registry.create(tokenizer, locale=locale)
  feature_plugin = registry.create(feature_extractor)
  classifier_plugin = registry.create(classifier)
  ```
- **Explicit workflow**:
  1. Tokenize documents using tokenizer plugin
  2. Fit feature extractor on tokenized documents
  3. Transform documents to feature vectors
  4. Train classifier on feature vectors
  5. Save both classifier and feature extractor separately

#### Data Filtering
- **Training data**: Uses documents with `is_training_data=True`
- This ensures clean separation between training and test sets

---

### 2. **Field Prediction Task** (`src/lexiclass_worker/tasks/field_predict.py`)

#### Schema Updates
- **Added plugin parameters** to `PredictFieldDocumentsInput`:
  - `tokenizer` (default: "icu") - Must match training
  - `feature_extractor` (default: "bow") - Must match training
  - `locale` (default: "en") - Must match training

#### Implementation Changes
- **Removed** direct pickle loading of classifier/vectorizer
- **Now uses** LexiClass plugin system:
  ```python
  tokenizer_plugin = registry.create(tokenizer, locale=locale)
  feature_plugin = registry.create(feature_extractor)
  feature_plugin.load(str(vectorizer_path))

  classifier_type = model.metrics.get("classifier", "svm")
  classifier_plugin = registry.create(classifier_type)
  classifier_plugin.load(str(model_path))
  ```
- **Prediction workflow**:
  1. Load tokenizer and feature extractor plugins
  2. Load classifier from model metadata
  3. Tokenize input documents
  4. Transform to feature vectors
  5. Run classifier prediction
  6. Store predictions to database and disk

---

### 3. **Field Evaluation Task** (`src/lexiclass_worker/tasks/field_evaluate.py`) - NEW

#### Purpose
Evaluate trained models on test data (documents with `is_training_data=False`)

#### Schema
- `EvaluateFieldModelInput`:
  - `field_id` - Field to evaluate
  - `project_id` - Project ID
  - `tokenizer` (default: "icu") - Must match training
  - `feature_extractor` (default: "bow") - Must match training
  - `locale` (default: "en") - Must match training

- `EvaluateFieldModelOutput`:
  - `field_id` - Field evaluated
  - `model_version` - Model version evaluated
  - `metrics` - Evaluation metrics (accuracy, precision, recall, F1)
  - `num_test_documents` - Number of test documents

#### Implementation
- **Filters** documents with `is_training_data=False`
- **Loads** latest ready model for the field
- **Runs** predictions on test data using same plugins as training
- **Calculates** comprehensive metrics:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 score (weighted)
  - Per-class metrics (precision, recall, F1 per class)
- **Updates** model record with evaluation metrics
- **Stores** accuracy in model.accuracy field

---

## Plugin Support Matrix

### Default Plugins
- **Tokenizer**: `icu` (locale-aware, fast)
- **Feature Extractor**: `bow` (bag-of-words, simple)
- **Classifier**: `svm` (linear SVM, fast and accurate)

### Supported Plugins

#### Tokenizers
- `icu` - Fast, locale-aware tokenization (default)
- `spacy` - Linguistic features, stop word filtering
- `sentencepiece` - Trainable subword tokenization
- `huggingface` - Access to 1000+ pre-trained tokenizers

#### Feature Extractors
- `bow` - Simple word counts (default)
- `tfidf` - TF-IDF weighting
- `fasttext` - Subword embeddings
- `sbert` - Sentence-BERT transformer embeddings

#### Classifiers
- `svm` - Linear SVM (default)
- `xgboost` - Gradient boosting
- `transformer` - Fine-tuned BERT/RoBERTa

---

## Data Flow

### Training Flow
1. **API** submits training task with plugin parameters
2. **Worker** receives task from Redis queue
3. **Task** loads training data (is_training_data=True)
4. **Plugins** created using registry:
   - Tokenizer tokenizes documents
   - Feature extractor fits and transforms
   - Classifier trains on feature vectors
5. **Models saved** to disk (classifier.pkl, vectorizer.pkl)
6. **Database updated** with model metadata including plugin info
7. **Status** updated to READY

### Prediction Flow
1. **API** submits prediction task with plugin parameters
2. **Worker** loads latest ready model
3. **Plugins** created and loaded:
   - Same tokenizer/feature extractor as training
   - Classifier loaded from disk
4. **Documents** tokenized and transformed
5. **Predictions** generated
6. **Results** saved to database and disk (JSONL)

### Evaluation Flow
1. **API** submits evaluation task
2. **Worker** loads test data (is_training_data=False)
3. **Same plugins** as training loaded
4. **Predictions** run on test set
5. **Metrics** calculated (accuracy, precision, recall, F1)
6. **Model updated** with evaluation results

---

## Breaking Changes

### API Integration Required

The API service must be updated to:

1. **Accept plugin parameters** in training endpoints:
   ```json
   {
     "field_id": 123,
     "project_id": 456,
     "tokenizer": "icu",
     "feature_extractor": "tfidf",
     "classifier": "svm",
     "locale": "en"
   }
   ```

2. **Pass same parameters** to prediction endpoints:
   ```json
   {
     "field_id": 123,
     "project_id": 456,
     "document_ids": [1, 2, 3],
     "tokenizer": "icu",
     "feature_extractor": "tfidf",
     "locale": "en"
   }
   ```

3. **Add evaluation endpoint** to trigger evaluation tasks:
   ```json
   {
     "field_id": 123,
     "project_id": 456,
     "tokenizer": "icu",
     "feature_extractor": "tfidf",
     "locale": "en"
   }
   ```

4. **Store plugin metadata** with models to auto-populate parameters

---

## Database Schema Considerations

### Model.metrics Field
The `metrics` JSONB field now stores:
```json
{
  "num_documents": 1000,
  "num_classes": 5,
  "num_features": 5000,
  "tokenizer": "icu",
  "feature_extractor": "tfidf",
  "classifier": "svm",
  "evaluation": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92,
    "num_test_documents": 200,
    "per_class_metrics": {...}
  }
}
```

This enables:
- Auto-detection of which plugins were used during training
- Proper parameter passing for prediction/evaluation
- Tracking of both training and evaluation metrics

---

## Migration Guide

### For Existing Models
1. Models trained with old code will have `metrics` without plugin info
2. Prediction/evaluation tasks will default to "icu", "bow", "svm"
3. **Recommendation**: Retrain models to capture plugin metadata

### For New Deployments
1. All new training tasks should specify plugins explicitly
2. API should validate plugin compatibility
3. Consider adding plugin validation to API schemas

---

## Testing Recommendations

### Unit Tests
- Test each plugin combination independently
- Verify plugin loading/saving
- Test error handling for missing plugins

### Integration Tests
- Full workflow: train → predict → evaluate
- Multiple plugin combinations
- Edge cases (empty test sets, missing documents)

### Performance Tests
- Compare plugin combinations (speed vs accuracy)
- Memory usage with different feature extractors
- GPU acceleration for transformer models

---

## Benefits

1. **Flexibility**: Easy to switch between ML algorithms
2. **Extensibility**: Add new plugins without changing worker code
3. **Maintainability**: Clear separation between worker and ML logic
4. **Performance**: Choose speed vs accuracy based on needs
5. **Consistency**: All tasks use same plugin architecture
6. **Testing**: Separate training and test data (is_training_data flag)

---

## Files Modified

1. `src/lexiclass_worker/tasks/field_train.py` - Updated to use plugins
2. `src/lexiclass_worker/tasks/field_predict.py` - Updated to use plugins
3. `src/lexiclass_worker/tasks/field_evaluate.py` - **NEW** evaluation task
4. `src/lexiclass_worker/tasks/__init__.py` - Export new evaluate task

---

## Next Steps

1. **Update API service** to support new plugin parameters
2. **Add plugin validation** to API schemas
3. **Create API endpoints** for evaluation tasks
4. **Update documentation** with plugin usage examples
5. **Add integration tests** for all plugin combinations
6. **Consider** adding plugin selection UI in dashboard

---

## Notes

- Plugin parameters must match between training and prediction/evaluation
- Default values ensure backward compatibility
- Model metadata stores which plugins were used
- Evaluation is now separate from training (proper ML practice)
- Test data (is_training_data=False) used only for evaluation

---

**Reviewed by**: Claude Code
**Status**: ✅ Completed
**All syntax checks**: ✅ Passed

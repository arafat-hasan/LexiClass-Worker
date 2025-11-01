"""Field-level evaluation task implementation."""

import asyncio
import logging
from typing import Dict, Optional

from pydantic import Field as PydanticField

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import get_db_session

logger = logging.getLogger(__name__)


class EvaluateFieldModelInput(TaskInput):
    """Input schema for field model evaluation task."""

    field_id: int = PydanticField(..., description="Field ID to evaluate")
    project_id: int = PydanticField(..., description="Project ID")
    tokenizer: str = PydanticField(
        default="icu",
        description="Tokenizer plugin name (must match training)"
    )
    feature_extractor: str = PydanticField(
        default="bow",
        description="Feature extractor plugin name (must match training)"
    )
    locale: str = PydanticField(
        default="en",
        description="Locale for tokenization (must match training)"
    )


class EvaluateFieldModelOutput(TaskOutput):
    """Output schema for field model evaluation task."""

    field_id: int
    model_version: Optional[int] = None
    metrics: Optional[Dict] = None
    num_test_documents: Optional[int] = None


class EvaluateFieldModelTask(MLTaskBase):
    """Task for evaluating field-specific models on test data."""

    name = "lexiclass_worker.tasks.evaluate_field_model_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return EvaluateFieldModelInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return EvaluateFieldModelOutput


async def _evaluate_field_model_async(
    field_id: int,
    project_id: int,
    tokenizer: str = "icu",
    feature_extractor: str = "bow",
    locale: str = "en"
) -> dict:
    """Evaluate a model for a specific field using test data (async implementation)."""
    from lexiclass.io import DocumentLoader
    from lexiclass.plugins import registry
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

    settings = get_settings()

    # Import API models dynamically to access database
    from ..models import Field, FieldClass, DocumentLabel, Model, ModelStatus

    async with get_db_session() as session:
        # Get field
        from sqlalchemy import select
        result = await session.execute(select(Field).where(Field.id == field_id))
        field = result.scalar_one_or_none()

        if not field:
            raise ValueError(f"Field {field_id} not found")

        logger.info(f"Starting evaluation for field: {field.name} ({field_id})")

        # Get test labels - ONLY those marked as test data (is_training_data=False)
        result = await session.execute(
            select(DocumentLabel)
            .where(DocumentLabel.field_id == field_id)
            .where(DocumentLabel.is_training_data == False)
        )
        test_labels = result.scalars().all()

        if not test_labels:
            raise ValueError(
                f"No test labels found for field {field_id}. "
                f"Please ensure documents are labeled with is_training_data=False."
            )

        logger.info(f"Found {len(test_labels)} test labels (is_training_data=False)")

        # Get field classes
        result = await session.execute(
            select(FieldClass).where(FieldClass.field_id == field_id)
        )
        classes = result.scalars().all()
        class_map = {cls.id: cls.name for cls in classes}

        # Get latest ready model
        result = await session.execute(
            select(Model)
            .where(Model.field_id == field_id)
            .where(Model.status == ModelStatus.READY)
            .order_by(Model.version.desc())
            .limit(1)
        )
        model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"No ready model found for field {field_id}")

        logger.info(f"Evaluating model version {model.version} (ID: {model.id})")

        # Load model files
        model_path = model.get_model_path(settings.storage.base_path, project_id)
        vectorizer_path = model.get_vectorizer_path(settings.storage.base_path, project_id)

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise ValueError(f"Vectorizer file not found: {vectorizer_path}")

        logger.info(
            f"Loading plugins: tokenizer={tokenizer}, "
            f"feature_extractor={feature_extractor}"
        )

        # Create tokenizer plugin
        tokenizer_plugin = registry.create(tokenizer, locale=locale)

        # Create and load feature extractor plugin
        feature_plugin = registry.create(feature_extractor)
        feature_plugin.load(str(vectorizer_path))

        # Determine which classifier was used from model metadata
        classifier_type = model.metrics.get("classifier", "svm") if model.metrics else "svm"
        classifier_plugin = registry.create(classifier_type)
        classifier_plugin.load(str(model_path))

        # Load document contents
        documents_dir = settings.storage.base_path / str(project_id) / "documents"
        all_docs = DocumentLoader.load_documents_from_directory(str(documents_dir))

        logger.info(
            f"Found {len(all_docs)} documents in storage at {documents_dir}, "
            f"{len(test_labels)} test labels in database"
        )

        # Prepare test data
        texts = []
        true_labels = []
        missing_docs = []
        missing_classes = []
        matched_docs = []

        for label in test_labels:
            class_name = class_map.get(label.class_id)

            if not class_name:
                missing_classes.append(label.class_id)
                continue

            # DocumentLoader returns dict with STRING keys, but document_id is INTEGER
            # Convert document_id to string for lookup
            doc_key = str(label.document_id)

            if doc_key not in all_docs:
                missing_docs.append(label.document_id)
                logger.debug(
                    f"Document ID {label.document_id} (looking for key '{doc_key}') "
                    f"not found in storage"
                )
                continue

            texts.append(all_docs[doc_key])
            true_labels.append(class_name)
            matched_docs.append(label.document_id)

        # Log matching results
        logger.info(
            f"Document matching results: "
            f"{len(matched_docs)} successfully matched, "
            f"{len(missing_docs)} missing from storage, "
            f"{len(missing_classes)} with missing classes"
        )

        if missing_docs:
            logger.warning(
                f"Missing {len(missing_docs)} document files for test documents. "
                f"Document IDs: {missing_docs[:10]}{'...' if len(missing_docs) > 10 else ''}"
            )

        if missing_classes:
            logger.warning(
                f"Missing {len(missing_classes)} class definitions. "
                f"Class IDs: {missing_classes[:10]}{'...' if len(missing_classes) > 10 else ''}"
            )

        if len(texts) < 2:
            raise ValueError(
                f"Need at least 2 test documents for evaluation. "
                f"Found {len(test_labels)} test labels but only {len(texts)} have valid files."
            )

        # Make predictions on test data
        logger.info(f"Evaluating on {len(texts)} test documents...")

        # Tokenize documents
        tokenized_docs = [tokenizer_plugin.tokenize(text) for text in texts]

        # Transform to feature vectors
        X = feature_plugin.transform(tokenized_docs)

        # Predict
        predicted_labels, confidences = classifier_plugin.predict(X)

        # Calculate evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', zero_division=0
        )

        # Get per-class metrics
        report_dict = classification_report(
            true_labels, predicted_labels, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "num_test_documents": len(texts),
            "per_class_metrics": report_dict,
            "confusion_matrix": None,  # Could add confusion matrix if needed
        }

        logger.info(
            f"Evaluation completed: Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        # Update model with evaluation metrics
        if model.metrics is None:
            model.metrics = {}

        model.metrics["evaluation"] = metrics
        model.accuracy = float(accuracy)
        await session.commit()

        logger.info(f"Updated model {model.id} with evaluation metrics")

        return {
            "status": "completed",
            "project_id": project_id,
            "field_id": field_id,
            "model_version": model.version,
            "metrics": metrics,
            "num_test_documents": len(texts),
        }


@app.task(base=EvaluateFieldModelTask, bind=True)
def evaluate_field_model_task(self, **kwargs) -> dict:
    """Evaluate a model for a specific field on test data (Celery task wrapper).

    Args:
        field_id: Field ID to evaluate
        project_id: Project ID
        tokenizer: Tokenizer plugin name (must match training)
        feature_extractor: Feature extractor plugin name (must match training)
        locale: Locale for tokenization (must match training)

    Returns:
        Task result with evaluation metrics
    """
    from ..core.database import ensure_db_initialized

    # Validate input
    input_data = self.validate_input(kwargs)

    logger.info(
        f"Starting evaluation task for field_id={input_data.field_id}, "
        f"project_id={input_data.project_id}"
    )

    # Ensure database is initialized
    ensure_db_initialized()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _evaluate_field_model_async(
                field_id=input_data.field_id,
                project_id=input_data.project_id,
                tokenizer=input_data.tokenizer,
                feature_extractor=input_data.feature_extractor,
                locale=input_data.locale,
            )
        )
        return self.validate_output(result).model_dump()
    finally:
        loop.close()

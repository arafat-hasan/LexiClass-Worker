"""Field-level prediction task implementation."""

import asyncio
import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field as PydanticField

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import get_db_session

logger = logging.getLogger(__name__)


class PredictFieldDocumentsInput(TaskInput):
    """Input schema for field document prediction task."""

    field_id: int = PydanticField(..., description="Field ID to use for prediction")
    project_id: int = PydanticField(..., description="Project ID")
    document_ids: List[int] = PydanticField(..., description="Document IDs to predict")
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


class PredictFieldDocumentsOutput(TaskOutput):
    """Output schema for field document prediction task."""

    field_id: int
    predictions_created: Optional[int] = None
    total_documents: int
    model_version: Optional[int] = None


class PredictFieldDocumentsTask(MLTaskBase):
    """Task for running predictions on field documents."""

    name = "lexiclass_worker.tasks.predict_field_documents_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return PredictFieldDocumentsInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return PredictFieldDocumentsOutput


async def _predict_field_documents_async(
    field_id: int,
    project_id: int,
    document_ids: List[int],
    tokenizer: str = "icu",
    feature_extractor: str = "bow",
    locale: str = "en"
) -> dict:
    """Predict documents using a field's model (async implementation)."""
    from lexiclass.io import DocumentLoader
    from lexiclass.plugins import registry

    settings = get_settings()

    from ..models import Field, FieldClass, Model, Prediction, ModelStatus
    from sqlalchemy import select, delete

    async with get_db_session() as session:
        # Get field
        result = await session.execute(select(Field).where(Field.id == field_id))
        field = result.scalar_one_or_none()

        if not field:
            raise ValueError(f"Field {field_id} not found")

        logger.info(f"Starting prediction for field: {field.name} ({field_id})")

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

        logger.info(f"Using model version {model.version} (ID: {model.id})")

        # Load model and vectorizer using LexiClass plugins
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
        # For now, we'll infer from the stored metadata or default to svm
        classifier_type = model.metrics.get("classifier", "svm") if model.metrics else "svm"
        classifier_plugin = registry.create(classifier_type)
        classifier_plugin.load(str(model_path))

        # Get field classes
        result = await session.execute(
            select(FieldClass).where(FieldClass.field_id == field_id)
        )
        classes = result.scalars().all()
        class_name_to_id = {cls.name: cls.id for cls in classes}

        # Load documents
        documents_dir = settings.storage.base_path / str(project_id) / "documents"
        all_docs = DocumentLoader.load_documents_from_directory(str(documents_dir))

        # Filter requested documents - convert doc_ids to strings for lookup
        docs_to_predict = {}
        for doc_id in document_ids:
            doc_key = str(doc_id)
            if doc_key in all_docs:
                docs_to_predict[doc_id] = all_docs[doc_key]

        if not docs_to_predict:
            raise ValueError("No valid documents found for prediction")

        logger.info(f"Predicting {len(docs_to_predict)} documents")

        # Make predictions using plugins
        texts = list(docs_to_predict.values())
        doc_ids_list = list(docs_to_predict.keys())

        # Tokenize documents
        logger.info("Tokenizing documents...")
        tokenized_docs = [tokenizer_plugin.tokenize(text) for text in texts]

        # Transform to feature vectors
        logger.info("Transforming to feature vectors...")
        X = feature_plugin.transform(tokenized_docs)

        # Predict
        logger.info("Running classifier prediction...")
        predicted_classes, confidences = classifier_plugin.predict(X)

        # Store prediction scores to disk
        from ..models import Prediction as PredictionModel

        predictions_path = PredictionModel.get_prediction_scores_path(
            settings.storage.base_path, project_id, field_id, model.version
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        # Write all prediction scores to JSONL file
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for doc_id, predicted_class, confidence in zip(doc_ids_list, predicted_classes, confidences):
                class_id = class_name_to_id.get(predicted_class)
                if class_id:
                    score_entry = {
                        "document_id": doc_id,
                        "predicted_class": predicted_class,
                        "class_id": class_id,
                        "confidence": float(confidence),
                        "model_version": model.version,
                    }
                    f.write(json.dumps(score_entry) + '\n')

        logger.info(f"Saved prediction scores to {predictions_path}")

        # Delete existing predictions for these documents and field
        await session.execute(
            delete(Prediction)
            .where(Prediction.field_id == field_id)
            .where(Prediction.document_id.in_(document_ids))
        )

        # Create new predictions (only latest per document per field)
        predictions_created = 0
        for doc_id, predicted_class, confidence in zip(doc_ids_list, predicted_classes, confidences):
            class_id = class_name_to_id.get(predicted_class)
            if class_id:
                prediction = Prediction(
                    document_id=doc_id,
                    field_id=field_id,
                    class_id=class_id,
                    model_version=model.version,
                    confidence=float(confidence),
                    pred_metadata={
                        "predicted_class": predicted_class,
                    },
                )
                session.add(prediction)
                predictions_created += 1

        await session.commit()

        logger.info(f"Created {predictions_created} predictions for field {field_id}")

        return {
            "status": "completed",
            "project_id": project_id,
            "field_id": field_id,
            "model_version": model.version,
            "predictions_created": predictions_created,
            "total_documents": len(document_ids),
        }


@app.task(base=PredictFieldDocumentsTask, bind=True)
def predict_field_documents_task(self, **kwargs) -> dict:
    """Predict documents using a field's model (Celery task wrapper).

    Args:
        field_id: Field ID to use for prediction
        project_id: Project ID
        document_ids: List of document IDs to predict

    Returns:
        Task result with prediction statistics
    """
    from lexiclass_core.db.session import AsyncSessionFactory
    from ..core.database import initialize_database

    # Validate input
    input_data = self.validate_input(kwargs)

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Initialize database in this event loop context
        # This ensures database connections are bound to the same loop we're using
        if AsyncSessionFactory is None:
            logger.info("Initializing database in task event loop...")
            initialize_database()

        result = loop.run_until_complete(
            _predict_field_documents_async(
                field_id=input_data.field_id,
                project_id=input_data.project_id,
                document_ids=input_data.document_ids,
                tokenizer=input_data.tokenizer,
                feature_extractor=input_data.feature_extractor,
                locale=input_data.locale,
            )
        )
        return self.validate_output(result).model_dump()
    finally:
        loop.close()

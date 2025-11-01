"""Field-level training task implementation."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field as PydanticField

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import get_db_session

logger = logging.getLogger(__name__)


class TrainFieldModelInput(TaskInput):
    """Input schema for field model training task."""

    field_id: int = PydanticField(..., description="Field ID to train")
    project_id: int = PydanticField(..., description="Project ID")
    tokenizer: str = PydanticField(
        default="icu",
        description="Tokenizer plugin name (icu, spacy, sentencepiece, huggingface)"
    )
    feature_extractor: str = PydanticField(
        default="bow",
        description="Feature extractor plugin name (bow, tfidf, fasttext, sbert)"
    )
    classifier: str = PydanticField(
        default="svm",
        description="Classifier plugin name (svm, xgboost, transformer)"
    )
    locale: str = PydanticField(
        default="en",
        description="Locale for tokenization (e.g., 'en', 'es', 'fr')"
    )


class TrainFieldModelOutput(TaskOutput):
    """Output schema for field model training task."""

    model_id: Optional[int] = None
    model_version: Optional[int] = None
    metrics: Optional[Dict] = None


class TrainFieldModelTask(MLTaskBase):
    """Task for training field-specific classification models."""

    name = "lexiclass_worker.tasks.train_field_model_task"

    @property
    def input_schema(self) -> type[TaskInput]:
        return TrainFieldModelInput

    @property
    def output_schema(self) -> type[TaskOutput]:
        return TrainFieldModelOutput


async def _train_field_model_async(
    field_id: int,
    project_id: int,
    tokenizer: str = "icu",
    feature_extractor: str = "bow",
    classifier: str = "svm",
    locale: str = "en"
) -> dict:
    """Train a model for a specific field (async implementation)."""
    from lexiclass.io import DocumentLoader
    from lexiclass.plugins import registry

    settings = get_settings()

    # Import API models dynamically to access database
    from ..models import Field, FieldClass, DocumentLabel, Model, Document

    async with get_db_session() as session:
        # Get field
        from sqlalchemy import select
        result = await session.execute(select(Field).where(Field.id == field_id))
        field = result.scalar_one_or_none()

        if not field:
            raise ValueError(f"Field {field_id} not found")

        logger.info(f"Starting training for field: {field.name} ({field_id})")

        # Get training labels - ONLY those marked as training data
        result = await session.execute(
            select(DocumentLabel)
            .where(DocumentLabel.field_id == field_id)
            .where(DocumentLabel.is_training_data == True)
        )
        labels = result.scalars().all()

        if not labels:
            raise ValueError(
                f"No training labels found for field {field_id}. "
                f"Please ensure documents are labeled with is_training_data=True."
            )

        logger.info(f"Found {len(labels)} training labels (is_training_data=True)")

        # Get field classes
        result = await session.execute(
            select(FieldClass).where(FieldClass.field_id == field_id)
        )
        classes = result.scalars().all()
        class_map = {cls.id: cls.name for cls in classes}

        # Get latest model version
        result = await session.execute(
            select(Model)
            .where(Model.field_id == field_id)
            .order_by(Model.version.desc())
            .limit(1)
        )
        latest_model = result.scalar_one_or_none()
        new_version = (latest_model.version + 1) if latest_model else 1

        # Create model record with TRAINING status
        from ..models import ModelStatus

        new_model = Model(
            field_id=field_id,
            version=new_version,
            status=ModelStatus.TRAINING,
        )
        session.add(new_model)
        await session.commit()
        await session.refresh(new_model)

        model_id = new_model.id

        try:
            # Load document contents
            documents_dir = settings.storage.base_path / str(project_id) / "documents"
            all_docs = DocumentLoader.load_documents_from_directory(str(documents_dir))

            logger.info(
                f"Found {len(all_docs)} documents in storage at {documents_dir}, "
                f"{len(labels)} labels in database"
            )

            # Prepare training data
            texts = []
            labels_list = []
            missing_docs = []
            missing_classes = []
            matched_docs = []

            for label in labels:
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
                labels_list.append(class_name)
                matched_docs.append(label.document_id)

            # Log matching results
            logger.info(
                f"Document matching results: "
                f"{len(matched_docs)} successfully matched, "
                f"{len(missing_docs)} missing from storage, "
                f"{len(missing_classes)} with missing classes"
            )

            # Log diagnostics if documents are missing
            if missing_docs:
                logger.warning(
                    f"Missing {len(missing_docs)} document files for labeled documents. "
                    f"Document IDs: {missing_docs[:10]}{'...' if len(missing_docs) > 10 else ''}"
                )

            if missing_classes:
                logger.warning(
                    f"Missing {len(missing_classes)} class definitions. "
                    f"Class IDs: {missing_classes[:10]}{'...' if len(missing_classes) > 10 else ''}"
                )

            if len(texts) < 2:
                error_details = []
                error_details.append(f"Found {len(labels)} labels in database")
                error_details.append(f"Found {len(all_docs)} documents in storage")
                error_details.append(f"Successfully matched {len(texts)} documents with labels")

                if missing_docs:
                    error_details.append(
                        f"{len(missing_docs)} labels have no corresponding document files in storage"
                    )
                if missing_classes:
                    error_details.append(
                        f"{len(missing_classes)} labels reference missing classes"
                    )

                error_msg = (
                    f"Need at least 2 labeled documents with valid files for training. "
                    + " | ".join(error_details) + ". "
                    f"Please ensure documents are indexed before training."
                )
                raise ValueError(error_msg)

            # Create plugins using LexiClass registry
            logger.info(
                f"Initializing plugins: tokenizer={tokenizer}, "
                f"feature_extractor={feature_extractor}, classifier={classifier}"
            )

            # Create tokenizer plugin
            tokenizer_plugin = registry.create(tokenizer, locale=locale)

            # Create feature extractor plugin
            feature_plugin = registry.create(feature_extractor)

            # Create classifier plugin
            classifier_plugin = registry.create(classifier)

            logger.info("Plugins created successfully")

            # Tokenize documents
            logger.info("Tokenizing documents...")
            tokenized_docs = [tokenizer_plugin.tokenize(text) for text in texts]

            # Fit feature extractor
            logger.info("Fitting feature extractor...")
            feature_plugin.fit(tokenized_docs)

            # Transform to feature vectors
            logger.info("Transforming documents to feature vectors...")
            X = feature_plugin.transform(tokenized_docs)

            # Train classifier
            logger.info(f"Training {classifier} classifier on {len(texts)} documents with {len(set(labels_list))} classes...")
            classifier_plugin.train(X, labels_list)

            # Calculate training statistics
            training_stats = {
                "num_documents": len(texts),
                "num_classes": len(set(labels_list)),
                "num_features": feature_plugin.num_features(),
                "tokenizer": tokenizer,
                "feature_extractor": feature_extractor,
                "classifier": classifier,
            }

            # Save model files using dynamic path generation
            model_path = new_model.get_model_path(settings.storage.base_path, project_id)
            vectorizer_path = new_model.get_vectorizer_path(settings.storage.base_path, project_id)

            logger.info(f"Saving model files to {model_path.parent}")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Save classifier
            classifier_plugin.save(str(model_path))

            # Save feature extractor (vectorizer)
            feature_plugin.save(str(vectorizer_path))

            # Update model status
            new_model.status = ModelStatus.READY
            new_model.accuracy = None  # No accuracy metrics during training (evaluation is separate)
            new_model.metrics = training_stats  # Store training statistics
            new_model.trained_at = datetime.utcnow()

            await session.commit()

            logger.info(f"Training completed successfully for field {field_id}")

            return {
                "status": "completed",
                "project_id": project_id,
                "model_id": model_id,
                "model_version": new_version,
                "training_stats": training_stats,
            }

        except Exception as e:
            # Update model status to FAILED
            new_model.status = ModelStatus.FAILED
            await session.commit()
            logger.error(f"Training failed for field {field_id}: {e}")
            raise


@app.task(base=TrainFieldModelTask, bind=True)
def train_field_model_task(self, **kwargs) -> dict:
    """Train a model for a specific field (Celery task wrapper).

    Args:
        field_id: Field ID to train
        project_id: Project ID

    Returns:
        Task result with training statistics
    """
    from ..core.database import ensure_db_initialized

    # Validate input
    input_data = self.validate_input(kwargs)

    logger.info(
        f"Starting training task for field_id={input_data.field_id}, "
        f"project_id={input_data.project_id}"
    )

    # Ensure database is initialized (uses our new safety check)
    ensure_db_initialized()

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _train_field_model_async(
                field_id=input_data.field_id,
                project_id=input_data.project_id,
                tokenizer=input_data.tokenizer,
                feature_extractor=input_data.feature_extractor,
                classifier=input_data.classifier,
                locale=input_data.locale,
            )
        )
        return self.validate_output(result).model_dump()
    finally:
        loop.close()

"""Field-level training task implementation."""

import asyncio
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from pydantic import Field as PydanticField
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ..celery import app
from ..core.base import MLTaskBase, TaskInput, TaskOutput
from ..core.config import get_settings
from ..core.database import get_db_session

logger = logging.getLogger(__name__)


class TrainFieldModelInput(TaskInput):
    """Input schema for field model training task."""

    field_id: str = PydanticField(..., description="Field ID to train")
    project_id: str = PydanticField(..., description="Project ID")


class TrainFieldModelOutput(TaskOutput):
    """Output schema for field model training task."""

    model_id: Optional[str] = None
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


async def _train_field_model_async(field_id: str, project_id: str) -> dict:
    """Train a model for a specific field (async implementation)."""
    from lexiclass.io import DocumentLoader

    settings = get_settings()

    # Import API models dynamically to access database
    # We'll need to add these to the worker
    from ..models import Field, FieldClass, DocumentLabel, Model, Document

    async with get_db_session() as session:
        # Get field
        from sqlalchemy import select
        result = await session.execute(select(Field).where(Field.id == field_id))
        field = result.scalar_one_or_none()

        if not field:
            raise ValueError(f"Field {field_id} not found")

        logger.info(f"Starting training for field: {field.name} ({field_id})")

        # Get training labels
        result = await session.execute(
            select(DocumentLabel)
            .where(DocumentLabel.field_id == field_id)
            .where(DocumentLabel.is_training_data == True)
        )
        labels = result.scalars().all()

        if not labels:
            raise ValueError(f"No training labels found for field {field_id}")

        logger.info(f"Found {len(labels)} training labels")

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

        model_path_rel = f"{project_id}/models/{field_id}/v{new_version}/model.pkl"
        vectorizer_path_rel = f"{project_id}/models/{field_id}/v{new_version}/vectorizer.pkl"

        new_model = Model(
            id=str(uuid4()),
            field_id=field_id,
            version=new_version,
            model_path=model_path_rel,
            vectorizer_path=vectorizer_path_rel,
            status=ModelStatus.TRAINING,
        )
        session.add(new_model)
        await session.commit()
        await session.refresh(new_model)

        model_id = new_model.id

        try:
            # Load document contents
            documents_dir = settings.storage.base_path / project_id / "documents"
            all_docs = DocumentLoader.load_documents_from_directory(str(documents_dir))

            # Prepare training data
            texts = []
            labels_list = []

            for label in labels:
                class_name = class_map.get(label.class_id)
                if class_name and label.document_id in all_docs:
                    texts.append(all_docs[label.document_id])
                    labels_list.append(class_name)

            if len(texts) < 2:
                raise ValueError("Need at least 2 labeled documents for training")

            logger.info(f"Training with {len(texts)} documents, {len(set(labels_list))} classes")

            # Train vectorizer and classifier
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)

            # Split data for evaluation
            if len(texts) >= 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels_list, test_size=0.2, random_state=42, stratify=labels_list
                )
            else:
                X_train, X_test, y_train, y_test = X, X, labels_list, labels_list

            # Train classifier
            classifier = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                max_iter=1000,
                random_state=42
            )
            classifier.fit(X_train, y_train)

            # Evaluate
            y_pred = classifier.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "training_samples": len(texts),
                "num_classes": len(set(labels_list)),
            }

            # Save model files
            model_path = settings.storage.base_path / model_path_rel
            vectorizer_path = settings.storage.base_path / vectorizer_path_rel

            model_path.parent.mkdir(parents=True, exist_ok=True)
            vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

            with open(model_path, 'wb') as f:
                pickle.dump(classifier, f)

            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

            logger.info(f"Saved model to {model_path}")

            # Update model status
            new_model.status = ModelStatus.READY
            new_model.accuracy = metrics["accuracy"]
            new_model.metrics = metrics
            new_model.trained_at = datetime.utcnow()

            await session.commit()

            logger.info(f"Training completed successfully for field {field_id}")

            return {
                "status": "completed",
                "project_id": project_id,
                "model_id": model_id,
                "model_version": new_version,
                "metrics": metrics,
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
    # Validate input
    input_data = self.validate_input(kwargs)

    # Run async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _train_field_model_async(input_data.field_id, input_data.project_id)
        )
        return self.validate_output(result).model_dump()
    finally:
        loop.close()

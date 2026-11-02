"""Cloud Run Job entry point for model training."""

import logging
import sys
import uuid
from datetime import UTC, datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from challenge.config import get_settings
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import TARGET_COL, DelayModel

logger = logging.getLogger(__name__)

TRAINING_FILENAME = "data.csv"
# Same holdout split as the notebook and the unit tests.
VALIDATION_SIZE = 0.33
RANDOM_STATE = 42


### Define here all the step for the pipeline implementation ###


def new_run_id() -> str:
    """Return a sortable, unique model version, e.g. ``20261102T010203Z-1a2b3c``."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def step_load_data(model: DelayModel, project: str, dataset: str) -> pd.DataFrame:
    """Load raw flight records through the model's injected reader."""
    data = model.load_data(project=project, bq_dataset=dataset)
    logger.info("Loaded %d raw flight records", len(data))
    return data


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix and the delay target."""
    features, target = model.preprocess(data, target_column=TARGET_COL)
    logger.info("Preprocessed %d rows into %d features", *features.shape)
    return features, target


def step_train(model: DelayModel, features: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Evaluate on a holdout split, then refit on all rows for the shipped model.

    The returned metrics describe the holdout evaluation and are stored in
    ``model.metadata`` so they are persisted next to the artifact.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    labels = y_val[TARGET_COL]
    metrics = {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision_delay": round(precision_score(labels, predictions), 4),
        "recall_delay": round(recall_score(labels, predictions), 4),
        "f1_delay": round(f1_score(labels, predictions), 4),
        "recall_on_time": round(recall_score(labels, predictions, pos_label=0), 4),
    }
    logger.info("Holdout metrics: %s", metrics)

    model.fit(features, target)
    model.metadata.update(
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        training_rows=len(features),
        validation_rows=len(x_val),
        metrics=metrics,
    )
    return metrics


def step_save(model: DelayModel, bucket: str, run_id: str) -> str:
    """Persist the model and its metadata; return the artifact URI."""
    model.metadata["version"] = run_id
    uri = model.save(bucket=bucket, run_id=run_id)
    logger.info("Saved model %s to %s", run_id, uri)
    return uri


def main() -> str:
    """Orchestrate the full training pipeline."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        # Imported lazily so local runs never need the storage client.
        from challenge.connectors.gcs import GCSArtifactStore  # noqa: PLC0415

        model = DelayModel(reader=BigQueryClient(), store=GCSArtifactStore())
    else:
        model = DelayModel(reader=LocalCSVClient(TRAINING_FILENAME))

    run_id = new_run_id()
    logger.info("Training run %s (%s mode)", run_id, settings.deployment_mode)
    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features, target = step_preprocess(model, data)
    step_train(model, features, target)
    return step_save(model, settings.gcp_bucket_artifacts, run_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)

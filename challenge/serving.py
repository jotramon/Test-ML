"""Cloud Run Job entry point for model serving."""

import logging
import sys

import pandas as pd

from challenge.config import get_settings
from challenge.connectors.local import LocalCSVClient
from challenge.connectors.protocols import PREDICTION_ID_COLS
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

SERVING_FILENAME = "serving_input.csv"

### Define here all the step for the pipeline implementation ###


def step_load_model(model: DelayModel, bucket: str, run_id: str | None = None) -> None:
    """Load the model to serve: ``run_id`` if given, else the latest version.

    Raises if no artifact exists, so the job fails loudly instead of serving
    the untrained-model fallback.
    """
    model.load(bucket=bucket, run_id=run_id)
    logger.info("Loaded model %s", run_id or "latest")


def step_load_data(model: DelayModel, project: str, dataset: str) -> pd.DataFrame:
    """Load the flights to score through the model's injected reader."""
    data = model.load_data(project=project, bq_dataset=dataset)
    logger.info("Loaded %d flights to score", len(data))
    return data


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix (no target at serving time)."""
    return model.preprocess(data)


def step_predict(model: DelayModel, features: pd.DataFrame) -> list[int]:
    """Return a 0/1 delay prediction per flight."""
    predictions = model.predict(features)
    logger.info("Predicted %d delays out of %d flights", sum(predictions), len(predictions))
    return predictions


def step_write(model: DelayModel, predictions: list[int], data: pd.DataFrame) -> None:
    """Persist predictions next to the identifiers they belong to."""
    model.write_predictions(predictions, data[PREDICTION_ID_COLS])


def main() -> str:
    """Orchestrate the full serving pipeline."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        # Imported lazily so local runs never need the storage client.
        from challenge.connectors.gcs import (  # noqa: PLC0415
            GCSArtifactStore,
            GCSCSVClient,
        )

        model = DelayModel(
            reader=GCSCSVClient(blob_name=SERVING_FILENAME), store=GCSArtifactStore()
        )
        destination = f"{settings.bq_dataset}.{settings.bq_predictions_table}"
    else:
        model = DelayModel(reader=LocalCSVClient(SERVING_FILENAME))
        destination = f"{settings.artifacts_dir}/predictions.csv"

    step_load_model(model, settings.gcp_bucket_artifacts)
    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features = step_preprocess(model, data)
    predictions = step_predict(model, features)
    step_write(model, predictions, data)
    logger.info("Wrote %d predictions to %s", len(predictions), destination)
    return destination


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Serving job failed")
        sys.exit(1)

"""Artifact-store protocol and local/GCS filesystem implementations."""

from __future__ import annotations

import io
import json
from typing import Any

import joblib
import pandas as pd
from google.cloud import storage

from challenge.config import get_settings


def _storage_client() -> storage.Client:
    return storage.Client(project=get_settings().gcp_project_id or None)


class GCSArtifactStore:
    """Save and load model artifacts on Google Cloud Storage.

    The bucket name is read from ``Settings.gcp_bucket_artifacts`` at call time.

    Layout::

        gs://<bucket>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _bucket_name(self) -> str:
        return get_settings().gcp_bucket_artifacts

    def save(self, model: Any, run_id: str) -> str:
        """Serialize and upload model to GCS; update latest.txt."""
        if model._model is None:
            raise ValueError("Cannot save an untrained model; call fit() first")
        bucket_name = self._bucket_name()
        bucket = _storage_client().bucket(bucket_name)

        buffer = io.BytesIO()
        joblib.dump(model._model, buffer)
        blob_name = f"{run_id}/model.joblib"
        bucket.blob(blob_name).upload_from_string(
            buffer.getvalue(), content_type="application/octet-stream"
        )
        metadata = getattr(model, "metadata", None)
        if metadata:
            bucket.blob(f"{run_id}/metadata.json").upload_from_string(
                json.dumps(metadata, indent=2), content_type="application/json"
            )
        # Written after the model so latest.txt never points at a missing artifact.
        bucket.blob("latest.txt").upload_from_string(run_id)
        return f"gs://{bucket_name}/{blob_name}"

    def load(self, model: Any, run_id: str | None) -> None:
        """Download and deserialize model from GCS into model._model."""
        bucket_name = self._bucket_name()
        bucket = _storage_client().bucket(bucket_name)

        if run_id is None:
            latest = bucket.blob("latest.txt")
            if not latest.exists():
                raise FileNotFoundError(f"No latest.txt in gs://{bucket_name}; train a model first")
            run_id = latest.download_as_text().strip()

        payload = bucket.blob(f"{run_id}/model.joblib").download_as_bytes()
        model._model = joblib.load(io.BytesIO(payload))


class GCSCSVClient:
    """Read flight data as a CSV object from the GCS input bucket.

    Predictions are written to BigQuery (not GCS) by delegating to an
    internal ``BigQueryClient`` — only the read side bypasses BigQuery.
    """

    def __init__(self, blob_name: str, _bq_writer: Any | None = None) -> None:
        """Initialise with the input-bucket object name and an optional injected writer."""
        self._blob_name = blob_name
        self._bq_writer = _bq_writer

    def read(self) -> pd.DataFrame:
        """Download and parse the CSV object from the configured input bucket."""
        bucket = _storage_client().bucket(get_settings().gcp_bucket_input)
        payload = bucket.blob(self._blob_name).download_as_bytes()
        return pd.read_csv(io.BytesIO(payload))

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to the configured BigQuery predictions table."""
        if self._bq_writer is None:
            # Imported lazily so the BigQuery client is only built when writing.
            from challenge.connectors.bigquery import BigQueryClient  # noqa: PLC0415

            self._bq_writer = BigQueryClient()
        self._bq_writer.write(predictions, identifiers)

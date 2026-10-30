"""Artifact-store protocol and local/GCS filesystem implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd


class GCSArtifactStore:
    """Save and load model artifacts on Google Cloud Storage.

    The bucket name is read from ``Settings.gcp_bucket_artifacts`` at call time.

    Layout::

        gs://<bucket>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _bucket_name(self) -> str:
        ...

    def save(self, model: Any, run_id: str) -> str:
        """Serialize and upload model to GCS; update latest.txt."""
        ...

    def load(self, model: Any, run_id: str | None) -> None:
        """Download and deserialize model from GCS into model._model."""
        ...


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
        ...

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to the configured BigQuery predictions table."""
        ...

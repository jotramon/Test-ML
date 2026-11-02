"""Data-reader implementation for BigQuery"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from challenge.config import get_settings
from challenge.connectors.protocols import build_predictions_frame

_PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("OPERA", "STRING"),
    bigquery.SchemaField("TIPOVUELO", "STRING"),
    bigquery.SchemaField("MES", "INTEGER"),
    bigquery.SchemaField("prediction", "INTEGER"),
]


class BigQueryClient:
    """Read raw flight data from BigQuery and write predictions to a BQ table.

    Reads the query from ``assets/load_data.sql`` (repo root) and substitutes
    project, dataset, and table name from ``Settings``.  Both ``read()`` and
    ``write()`` share the same ``google.cloud.bigquery.Client`` instance.

    Parameters
    ----------
    _client:
        Optional pre-built BQ client for testing. When ``None`` (default) a
        client is created using Application Default Credentials bound to
        ``Settings.gcp_project_id``.
    """

    _SQL_PATH = Path(__file__).resolve().parents[2] / "assets" / "load_data.sql"

    def __init__(self, _client: bigquery.Client | None = None) -> None:
        """Initialise with optional injected BQ client."""
        if _client is None:
            _client = bigquery.Client(project=get_settings().gcp_project_id or None)
        self._client = _client

    def read(self) -> pd.DataFrame:
        """Execute ``assets/load_data.sql`` against BigQuery and return results."""
        settings = get_settings()
        query = self._SQL_PATH.read_text().format(
            project=settings.gcp_project_id,
            dataset=settings.bq_dataset,
            raw_table=settings.bq_raw_table,
        )
        return self._client.query(query).to_dataframe()

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Insert predictions into the configured BigQuery predictions table.

        Each row contains ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) and ``prediction`` (0 or 1).  Uses ``WRITE_TRUNCATE`` to
        replace the table on each serving run.
        """
        settings = get_settings()
        frame = build_predictions_frame(predictions, identifiers)
        table_id = (
            f"{settings.gcp_project_id}.{settings.bq_dataset}.{settings.bq_predictions_table}"
        )
        job_config = bigquery.LoadJobConfig(
            schema=_PREDICTIONS_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        job = self._client.load_table_from_dataframe(frame, table_id, job_config=job_config)
        job.result()  # block until done; raises if the load job failed

"""Data-reader implementation for BigQuery"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from challenge.config import get_settings


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


    def __init__(self, _client: bigquery.Client | None = None) -> None:
        """Initialise with optional injected BQ client."""
        ...

    def read(self) -> pd.DataFrame:
        """Execute ``assets/load_data.sql`` against BigQuery and return results."""
        ...

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Insert predictions into the configured BigQuery predictions table.

        Each row contains ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) and ``prediction`` (0 or 1).  Uses ``WRITE_TRUNCATE`` to
        replace the table on each serving run.
        """
        ...

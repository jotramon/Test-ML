"""Helper functions for the pipeline control API.

Every helper takes the active ``Settings`` and dispatches on
``deployment_mode``: ``gcp`` talks to Cloud Run, GCS and BigQuery, ``local``
runs the pipelines in-process against ``DATA_DIR`` / ``ARTIFACTS_DIR`` so the
API (and the stress test) can run without GCP credentials.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from challenge.config import Settings
from challenge.connectors.protocols import PREDICTION_COL, PREDICTION_ID_COLS

logger = logging.getLogger(__name__)

# Upper bound for a synchronous pipeline run; matches the jobs' --task-timeout.
JOB_TIMEOUT_SECONDS = 15 * 60
# How long an instance reuses its snapshot of the predictions table.
PREDICTIONS_CACHE_TTL_SECONDS = 30
RESULT_DELAY_KEY = "predicted_delay"
_MONTHS = range(1, 13)
_MONTHS_ERROR = "mes must be a comma-separated list of integers between 1 and 12"

_PREDICTION_COLUMNS = [*PREDICTION_ID_COLS, PREDICTION_COL]


# --- Clients (created once per process; both are thread-safe) ---


@lru_cache(maxsize=1)
def _bigquery_client(project: str):
    from google.cloud import bigquery  # noqa: PLC0415

    return bigquery.Client(project=project)


@lru_cache(maxsize=1)
def _storage_bucket(project: str, bucket: str):
    from google.cloud import storage  # noqa: PLC0415

    return storage.Client(project=project).bucket(bucket)


@lru_cache(maxsize=1)
def _jobs_client():
    from google.cloud import run_v2  # noqa: PLC0415

    return run_v2.JobsClient()


# --- Pre-flight checks ---


def count_raw_flights(settings: Settings) -> int:
    """Return the number of rows available for training."""
    if settings.deployment_mode == "gcp":
        from google.api_core.exceptions import NotFound  # noqa: PLC0415

        table_id = settings.bq_table_id(settings.bq_raw_table)
        try:
            # Table metadata, not a query: free and instant.
            return (
                _bigquery_client(settings.gcp_project_id).get_table(table_id).num_rows
            )
        except NotFound:
            return 0
    path = Path(settings.data_dir) / "data.csv"
    if not path.exists():
        return 0
    with path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def model_exists(settings: Settings) -> bool:
    """Return True if a trained model (a ``latest.txt`` pointer) exists."""
    if settings.deployment_mode == "gcp":
        bucket = _storage_bucket(settings.gcp_project_id, settings.gcp_bucket_artifacts)
        return bucket.blob("latest.txt").exists()
    return (Path(settings.artifacts_dir) / "latest.txt").exists()


# --- Pipeline execution ---


class JobAlreadyRunningError(RuntimeError):
    """Raised when a run of the same pipeline is still in progress."""


# One lock per pipeline: rejects a second run on the same instance at once,
# without waiting for the first to finish.
_job_locks: dict[str, threading.Lock] = {}
_job_locks_guard = threading.Lock()


def _job_lock(job_name: str) -> threading.Lock:
    with _job_locks_guard:
        return _job_locks.setdefault(job_name, threading.Lock())


@lru_cache(maxsize=1)
def _executions_client():
    from google.cloud import run_v2  # noqa: PLC0415

    return run_v2.ExecutionsClient()


def _has_active_execution(job_path: str) -> bool:
    """True if a recent execution of the job has no completion time yet.

    Covers runs started by other instances (or by hand); executions are
    listed newest first, so a few are enough.
    """
    recent = islice(_executions_client().list_executions(parent=job_path), 5)
    return any("completion_time" not in execution for execution in recent)


def run_job(settings: Settings, job_name: str) -> Any:
    """Run a pipeline to completion and return its execution (``.name`` is its id).

    In gcp mode this executes the Cloud Run Job and blocks until it finishes;
    in local mode it calls the pipeline's ``main()`` in-process. Raises
    ``JobAlreadyRunningError`` if a run of the same pipeline is in progress,
    and ``RuntimeError`` if the run fails.
    """
    lock = _job_lock(job_name)
    if not lock.acquire(blocking=False):
        raise JobAlreadyRunningError(f"{job_name} is already running")
    try:
        if settings.deployment_mode == "gcp":
            return _run_cloud_run_job(settings, job_name)
        return _run_local_pipeline(settings, job_name)
    finally:
        lock.release()


def _run_cloud_run_job(settings: Settings, job_name: str) -> Any:
    client = _jobs_client()
    name = client.job_path(settings.gcp_project_id, settings.gcp_region, job_name)
    if _has_active_execution(name):
        raise JobAlreadyRunningError(f"{job_name} is already running")
    try:
        execution = client.run_job(name=name).result(timeout=JOB_TIMEOUT_SECONDS)
    except Exception as exc:
        raise RuntimeError(f"Cloud Run Job {job_name} failed: {exc}") from exc
    if execution.failed_count or not execution.succeeded_count:
        raise RuntimeError(
            f"Cloud Run Job {job_name} failed: execution {execution.name}"
        )
    return execution


def _run_local_pipeline(settings: Settings, job_name: str) -> Any:
    from challenge import serving, training  # noqa: PLC0415

    pipelines: dict[str, Callable[[], str]] = {
        settings.training_job: training.main,
        settings.serving_job: serving.main,
    }
    try:
        output = pipelines[job_name]()
    except Exception as exc:
        raise RuntimeError(f"Local pipeline {job_name} failed: {exc}") from exc
    return SimpleNamespace(name=f"local/{job_name}/{output}")


def latest_metadata(settings: Settings) -> dict[str, Any]:
    """Return ``metadata.json`` of the latest model; FileNotFoundError if none."""
    if settings.deployment_mode == "gcp":
        bucket = _storage_bucket(settings.gcp_project_id, settings.gcp_bucket_artifacts)
        pointer = bucket.blob("latest.txt")
        if not pointer.exists():
            raise FileNotFoundError(
                f"No latest.txt in gs://{settings.gcp_bucket_artifacts}"
            )
        run_id = pointer.download_as_text().strip()
        return json.loads(bucket.blob(f"{run_id}/metadata.json").download_as_text())
    root = Path(settings.artifacts_dir)
    run_id = (root / "latest.txt").read_text().strip()
    return json.loads((root / run_id / "metadata.json").read_text())


# --- Prediction results ---


class _PredictionsCache:
    """Per-instance snapshot of the predictions table, refreshed after a TTL.

    The table only changes when the serving pipeline runs, so reads are served
    from memory instead of one BigQuery query per request. The lock also
    collapses concurrent refreshes into a single load.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._frame: pd.DataFrame | None = None
        self._loaded_at = 0.0

    def get(self, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        with self._lock:
            if self._frame is None or time.monotonic() - self._loaded_at > self._ttl:
                self._frame = loader()
                self._loaded_at = time.monotonic()
            return self._frame

    def invalidate(self) -> None:
        with self._lock:
            self._frame = None


predictions_cache = _PredictionsCache(PREDICTIONS_CACHE_TTL_SECONDS)


def _load_predictions(settings: Settings) -> pd.DataFrame:
    if settings.deployment_mode == "gcp":
        from google.api_core.exceptions import NotFound  # noqa: PLC0415

        table_id = settings.bq_table_id(settings.bq_predictions_table)
        query = f"SELECT {', '.join(_PREDICTION_COLUMNS)} FROM `{table_id}`"
        try:
            frame = (
                _bigquery_client(settings.gcp_project_id)
                .query(query)
                .to_dataframe(create_bqstorage_client=False)
            )
        except NotFound:
            frame = pd.DataFrame(columns=_PREDICTION_COLUMNS)
    else:
        path = Path(settings.artifacts_dir) / "predictions.csv"
        frame = (
            pd.read_csv(path)
            if path.exists()
            else pd.DataFrame(columns=_PREDICTION_COLUMNS)
        )
    # Stable order so pages don't shift between refreshes (the table has no row id).
    return frame.sort_values(PREDICTION_ID_COLS, kind="mergesort").reset_index(
        drop=True
    )


def query_predictions(  # noqa: PLR0913 — mirrors the endpoint query params
    settings: Settings,
    page: int,
    page_size: int,
    *,
    opera: list[str] | None = None,
    tipovuelo: list[str] | None = None,
    mes: list[int] | None = None,
) -> dict[str, Any]:
    """Return ``{"total_predictions", "predictions"}`` for one filtered page."""
    frame = predictions_cache.get(lambda: _load_predictions(settings))
    mask = pd.Series(True, index=frame.index)
    for column, values in (("OPERA", opera), ("TIPOVUELO", tipovuelo), ("MES", mes)):
        if values:
            mask &= frame[column].isin(values)
    filtered = frame[mask]
    start = (page - 1) * page_size
    rows = filtered.iloc[start : start + page_size]
    return {
        "total_predictions": len(filtered),
        "predictions": [
            {
                "OPERA": str(opera),
                "TIPOVUELO": str(tipovuelo),
                "MES": int(mes),
                RESULT_DELAY_KEY: int(prediction),
            }
            for opera, tipovuelo, mes, prediction in rows[
                _PREDICTION_COLUMNS
            ].itertuples(index=False, name=None)
        ],
    }


def parse_csv_param(value: str | None) -> list[str] | None:
    """Split a comma-separated query param; None if absent or empty."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_months(value: str | None) -> list[int] | None:
    """Parse ``mes`` into ints in 1..12; ValueError with a readable message otherwise."""
    items = parse_csv_param(value)
    if items is None:
        return None
    try:
        months = [int(item) for item in items]
    except ValueError:
        raise ValueError(_MONTHS_ERROR) from None
    if any(month not in _MONTHS for month in months):
        raise ValueError(_MONTHS_ERROR)
    return months

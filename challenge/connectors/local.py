"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


class LocalCSVClient:
    """Read a CSV from the local data directory and write predictions to disk.

    The data directory is taken from the ``DATA_DIR`` environment variable,
    defaulting to ``data/`` relative to the current working directory.

    When ``filename`` is ``None`` (default): globs for the single ``.csv`` in
    the directory, raising ``ValueError`` on multiples and ``FileNotFoundError``
    on none — preserving original training behaviour.

    When ``filename`` is provided: reads that specific file directly, raising
    ``FileNotFoundError`` if it does not exist.
    """

    def __init__(self, filename: str | None = None) -> None:
        """Initialise the reader with an optional specific filename."""
        self._filename = filename

    def read(self) -> pd.DataFrame:
        """Return a DataFrame from the target CSV file."""
        ...

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Creates the artifacts directory if it does not exist.
        Output columns: ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1).
        """
        ...


class LocalArtifactStore:
    """Save and load model artifacts on the local filesystem.

    The root directory is taken from the ``ARTIFACTS_DIR`` environment variable,
    defaulting to ``artifacts/`` relative to the current working directory.

    Layout::

        <ARTIFACTS_DIR>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _root(self) -> Path:
        return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def save(self, model: Any, run_id: str) -> str:
        """Save model to <ARTIFACTS_DIR>/<run_id>/model.joblib and update latest.txt."""
        ...

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        ...
"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from challenge.connectors.protocols import build_predictions_frame


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

    def _data_dir(self) -> Path:
        return Path(os.environ.get("DATA_DIR", "data"))

    def _artifacts_dir(self) -> Path:
        return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def _resolve_path(self) -> Path:
        data_dir = self._data_dir()
        if self._filename is not None:
            path = data_dir / self._filename
            if not path.is_file():
                raise FileNotFoundError(f"CSV file not found: {path}")
            return path

        candidates = sorted(data_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise ValueError(
                f"Expected a single CSV in {data_dir}, found {len(candidates)}: {names}. "
                "Pass `filename` to choose one."
            )
        return candidates[0]

    def read(self) -> pd.DataFrame:
        """Return a DataFrame from the target CSV file."""
        return pd.read_csv(self._resolve_path(), low_memory=False)

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Creates the artifacts directory if it does not exist.
        Output columns: ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1).
        """
        out_dir = self._artifacts_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = build_predictions_frame(predictions, identifiers)
        frame.to_csv(out_dir / "predictions.csv", index=False)


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
        if model._model is None:
            raise ValueError("Cannot save an untrained model; call fit() first")
        root = self._root()
        path = root / run_id / "model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model._model, path)
        # Written after the model so latest.txt never points at a missing artifact.
        (root / "latest.txt").write_text(run_id)
        return str(path)

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        root = self._root()
        if run_id is None:
            latest = root / "latest.txt"
            if not latest.is_file():
                raise FileNotFoundError(f"No latest.txt in {root}; train a model first")
            run_id = latest.read_text().strip()

        path = root / run_id / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        model._model = joblib.load(path)

"""Protocols for data connectors"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

PREDICTION_ID_COLS = ["OPERA", "TIPOVUELO", "MES"]
PREDICTION_COL = "prediction"


@runtime_checkable
class DataReader(Protocol):
    """Interface for loading raw flight data and writing predictions."""

    def read(self) -> pd.DataFrame:
        """Return a DataFrame containing raw flight records."""
        ...

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Persist predictions, paired with their OPERA/TIPOVUELO/MES identifiers."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Interface for persisting and retrieving trained model artifacts."""

    def save(self, model: Any, run_id: str) -> str:
        """Persist model to storage and return the artifact path."""
        ...

    def load(self, model: Any, run_id: str | None) -> None:
        """Deserialise artifact into model._model; resolve latest when run_id is None."""
        ...


def build_predictions_frame(
    predictions: list[int], identifiers: pd.DataFrame
) -> pd.DataFrame:
    """Pair predictions with their identifiers in the shared output layout.

    Shared by every ``DataReader.write`` implementation so all sinks persist
    the same columns: ``OPERA``, ``TIPOVUELO``, ``MES``, ``prediction``.
    """
    if len(predictions) != len(identifiers):
        raise ValueError(
            f"Got {len(predictions)} predictions for {len(identifiers)} identifier rows"
        )
    frame = identifiers[PREDICTION_ID_COLS].reset_index(drop=True).copy()
    frame[PREDICTION_COL] = [int(p) for p in predictions]
    return frame

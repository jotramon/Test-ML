"""Protocols for data connectors"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


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
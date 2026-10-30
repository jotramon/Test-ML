from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xgboost as xgb

from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    ArtifactStore,
    DataReader,
)


class BaseDelayModel(ABC):
    """Abstract interface — defines the obligatory ML contract.

    Subclasses must implement preprocess, fit, and predict.
    Attempting to instantiate DelayModel without all three raises TypeError.
    """

    def __init__(self) -> None:
        self._model = None  # fitted XGBClassifier, populated by fit()

    @abstractmethod
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference.

        Parameters
        ----------
        data : pd.DataFrame
            Raw flight records with columns needed for training and/or inference.
        target_column : str, optional
            Name of the target column to return (e.g. "delay").

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features and target DataFrames when target_column is provided.
        pd.DataFrame
            Features DataFrame only when target_column is None.
        """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the selected model.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).
        target : pd.DataFrame
            Binary delay labels (column = "delay").
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions for each row in features.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).

        Returns
        -------
        list[int]
            0 (no delay) or 1 (delayed) for each input row.
        """


class GCPModelMixin:
    """Optional GCP I/O operations with injectable client support.

    Accepts ``reader`` and ``store`` keyword arguments so local and GCP
    implementations can be swapped without changing the step-function signatures.
    Defaults to ``LocalCSVReader`` and ``LocalArtifactStore`` for zero-credential
    local operation.
    """

    def __init__(
        self,
        reader: DataReader | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialise with optional injectable clients.

        Parameters
        ----------
        reader : DataReader, optional
            Data-loading client; defaults to ``LocalCSVReader()``.
        store : ArtifactStore, optional
            Artifact-persistence client; defaults to ``LocalArtifactStore()``.
        """
        ...

    def load_data(self, project: str = "", bq_dataset: str = "") -> pd.DataFrame:
        """Load raw flight data via the injected DataReader.

        Parameters
        ----------
        project : str
            GCP project ID (unused in local mode; accepted for API compatibility).
        bq_dataset : str
            BigQuery dataset name (unused in local mode; accepted for API compatibility).

        Returns
        -------
        pd.DataFrame
            Raw flight records ready to pass to preprocess().
        """
        ...

    def save(self, bucket: str = "", run_id: str = "") -> str:
        """Persist the fitted model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str
            Unique identifier for this training run.

        Returns
        -------
        str
            Path or URI of the saved artifact.
        """
        ...

    def load(self, bucket: str = "", run_id: str | None = None) -> None:
        """Restore a model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str, optional
            Specific run to load; latest artifact when omitted.
        """
        ...


class DelayModel(GCPModelMixin, BaseDelayModel):
    """Concrete class to implement for the challenge.

    Required ML methods are declared abstract in BaseDelayModel — omitting
    any of them raises TypeError at instantiation time.

    GCP I/O methods from GCPModelMixin are optional: implement them to
    connect preprocess/fit/predict to BigQuery and Cloud Storage.
    """

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference."""
        ...

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train model with class-imbalance weighting."""
        ...

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions."""
        ...

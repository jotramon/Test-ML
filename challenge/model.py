from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, overload

import pandas as pd
import xgboost as xgb

from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    ArtifactStore,
    DataReader,
)

logger = logging.getLogger(__name__)

# Top-10 features by XGBoost importance (exploration.ipynb, section 5).
FEATURES_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]
# Raw columns the one-hot encoding is built from (the only columns serving_input.csv has).
RAW_FEATURE_COLS = ["OPERA", "TIPOVUELO", "MES"]
TARGET_COL = "delay"
DELAY_THRESHOLD_MINUTES = 15
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class BaseDelayModel(ABC):
    """Abstract interface — defines the obligatory ML contract.

    Subclasses must implement preprocess, fit, and predict.
    Attempting to instantiate DelayModel without all three raises TypeError.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None  # populated by fit() or load()

    @overload
    def preprocess(
        self, data: pd.DataFrame, target_column: None = None
    ) -> pd.DataFrame: ...

    @overload
    def preprocess(
        self, data: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

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
    Defaults to ``LocalCSVClient`` and ``LocalArtifactStore`` for zero-credential
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
            Data-loading client; defaults to ``LocalCSVClient()``.
        store : ArtifactStore, optional
            Artifact-persistence client; defaults to ``LocalArtifactStore()``.
        """
        super().__init__()
        self._reader: DataReader = reader if reader is not None else LocalCSVClient()
        self._store: ArtifactStore = (
            store if store is not None else LocalArtifactStore()
        )
        # Run metadata (version, metrics, ...) persisted next to the artifact by the store.
        self.metadata: dict[str, Any] = {}

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
        # Project/dataset are resolved by the reader from Settings; kept for API compatibility.
        return self._reader.read()

    def write_predictions(
        self, predictions: list[int], identifiers: pd.DataFrame
    ) -> None:
        """Persist predictions through the injected DataReader.

        Parameters
        ----------
        predictions : list[int]
            0/1 delay predictions, row-aligned with ``identifiers``.
        identifiers : pd.DataFrame
            ``OPERA``, ``TIPOVUELO`` and ``MES`` of each scored flight.
        """
        self._reader.write(predictions, identifiers)

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
        if not run_id:
            raise ValueError("run_id is required to save a model artifact")
        return self._store.save(self, run_id)

    def load(self, bucket: str = "", run_id: str | None = None) -> None:
        """Restore a model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str, optional
            Specific run to load; latest artifact when omitted.
        """
        self._store.load(self, run_id)


class DelayModel(GCPModelMixin, BaseDelayModel):
    """Concrete class to implement for the challenge.

    Required ML methods are declared abstract in BaseDelayModel — omitting
    any of them raises TypeError at instantiation time.

    GCP I/O methods from GCPModelMixin are optional: implement them to
    connect preprocess/fit/predict to BigQuery and Cloud Storage.
    """

    @overload
    def preprocess(
        self, data: pd.DataFrame, target_column: None = None
    ) -> pd.DataFrame: ...

    @overload
    def preprocess(
        self, data: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference."""
        features = self._build_features(data)
        if target_column is None:
            return features
        return features, self._build_target(data, target_column)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train model with class-imbalance weighting.

        XGBoost on the top-10 features with ``scale_pos_weight = n_neg / n_pos``.
        Balanced XGBoost and LogisticRegression tie on quality (class-1 F1
        0.366 vs 0.364); XGBoost wins on class-1 recall stability across
        5-fold CV (0.689 ± 0.006 vs 0.647 ± 0.038). See docs/challenge.md.
        """
        labels = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target
        n_negative = int((labels == 0).sum())
        n_positive = int((labels == 1).sum())
        if n_negative == 0 or n_positive == 0:
            raise ValueError(
                "Training data must contain both delayed and on-time flights"
            )

        # Delays are ~18% of flights; without re-weighting the model predicts
        # almost only "no delay" (class-1 recall ~0.01 in the notebook).
        model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=n_negative / n_positive,
        )
        model.fit(features[FEATURES_COLS], labels)
        self._model = model

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions."""
        if self._model is None:
            # Interface contract: predict() must answer even before fit()/load().
            # Pipelines load an artifact explicitly and fail loudly if none exists,
            # so this fallback is only reached when used directly.
            logger.warning(
                "predict() called on an untrained model; returning 0 for every row"
            )
            return [0] * len(features)
        return [int(p) for p in self._model.predict(features[FEATURES_COLS])]

    @staticmethod
    def _build_features(data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode OPERA/TIPOVUELO/MES and keep exactly FEATURES_COLS."""
        missing = [col for col in RAW_FEATURE_COLS if col not in data.columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")

        dummies = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                # Cast so MES read as float/str still yields "MES_7", not "MES_7.0".
                pd.get_dummies(data["MES"].astype(int), prefix="MES"),
            ],
            axis=1,
        )
        # Categories absent from a batch (e.g. a small serving file) become 0-columns.
        return dummies.reindex(columns=FEATURES_COLS, fill_value=0).astype(int)

    @staticmethod
    def _build_target(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Return the target column, deriving ``delay`` from the schedule when absent."""
        if target_column in data.columns:
            return data[[target_column]].astype(int)
        if target_column != TARGET_COL:
            raise ValueError(f"Unknown target column '{target_column}'")

        scheduled = pd.to_datetime(data["Fecha-I"], format=_DATETIME_FORMAT)
        actual = pd.to_datetime(data["Fecha-O"], format=_DATETIME_FORMAT)
        min_diff = (actual - scheduled).dt.total_seconds() / 60
        delay = (min_diff > DELAY_THRESHOLD_MINUTES).astype(int)
        return pd.DataFrame({TARGET_COL: delay}, index=data.index)

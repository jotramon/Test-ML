"""Unit tests for GCPModelMixin and the untrained-predict guard in DelayModel."""

from unittest.mock import MagicMock

import pandas as pd

from challenge.connectors.protocols import ArtifactStore, DataReader
from challenge.model import DelayModel


def test_load_data_delegates_to_reader_read():
    """load_data() calls reader.read() once and returns its result."""
    mock_reader = MagicMock(spec=DataReader)
    expected = pd.DataFrame({"col": [1, 2]})
    mock_reader.read.return_value = expected
    model = DelayModel(reader=mock_reader)
    result = model.load_data()
    mock_reader.read.assert_called_once()
    assert result.equals(expected)


def test_load_data_accepts_project_and_bq_dataset_kwargs():
    """load_data() accepts project and bq_dataset kwargs without raising."""
    mock_reader = MagicMock(spec=DataReader)
    mock_reader.read.return_value = pd.DataFrame({"col": [1]})
    model = DelayModel(reader=mock_reader)
    result = model.load_data(project="proj", bq_dataset="ds")
    assert isinstance(result, pd.DataFrame)


def test_save_delegates_to_store_save():
    """save() calls store.save(model, run_id) and returns the artifact path."""
    mock_store = MagicMock(spec=ArtifactStore)
    mock_store.save.return_value = "path/to/model.joblib"
    model = DelayModel(store=mock_store)
    result = model.save(bucket="bucket", run_id="run-abc")
    mock_store.save.assert_called_once_with(model, "run-abc")
    assert result == "path/to/model.joblib"


def test_load_delegates_to_store_load_explicit_run_id():
    """load() with an explicit run_id delegates to store.load(model, run_id)."""
    mock_store = MagicMock(spec=ArtifactStore)
    model = DelayModel(store=mock_store)
    model.load(run_id="run-abc")
    mock_store.load.assert_called_once_with(model, "run-abc")


def test_load_delegates_to_store_load_none_run_id():
    """load() with no run_id delegates to store.load(model, None) for latest resolution."""
    mock_store = MagicMock(spec=ArtifactStore)
    model = DelayModel(store=mock_store)
    model.load()
    mock_store.load.assert_called_once_with(model, None)

"""Unit tests for challenge.connectors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from challenge.config import clear_settings_cache
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.connectors.protocols import DataReader

_SINGLE_FILE_ROWS = 2
_CUSTOM_VALUE = 42


def test_local_csv_reader_single_file(tmp_path, monkeypatch):
    """Single CSV in DATA_DIR returns a DataFrame."""
    csv = tmp_path / "flights.csv"
    csv.write_text("col\n1\n2\n")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    df = LocalCSVClient().read()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col"]
    assert len(df) == _SINGLE_FILE_ROWS


def test_local_csv_reader_no_csv_raises(tmp_path, monkeypatch):
    """Empty directory raises FileNotFoundError."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="No CSV files"):
        LocalCSVClient().read()


def test_local_csv_reader_ignores_non_csv(tmp_path, monkeypatch):
    """Non-CSV files alongside a CSV are ignored."""
    csv = tmp_path / "flights.csv"
    csv.write_text("col\n1\n")
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "data.parquet").write_bytes(b"")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    df = LocalCSVClient().read()
    assert isinstance(df, pd.DataFrame)


def test_local_csv_reader_custom_data_dir(tmp_path, monkeypatch):
    """DATA_DIR env var controls which directory is scanned."""
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "custom.csv").write_text("x\n42\n")
    monkeypatch.setenv("DATA_DIR", str(custom))
    df = LocalCSVClient().read()
    assert df["x"].iloc[0] == _CUSTOM_VALUE


def test_local_csv_reader_satisfies_protocol():
    """isinstance check confirms structural compatibility."""
    assert isinstance(LocalCSVClient(), DataReader)


# --- LocalCSVReader.write ---


def test_local_csv_reader_write_creates_csv(tmp_path, monkeypatch):
    """write() produces predictions.csv with correct columns and values."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    identifiers = pd.DataFrame(
        {
            "OPERA": ["Grupo LATAM", "Sky Airline", "Copa Air"],
            "TIPOVUELO": ["I", "N", "I"],
            "MES": [1, 2, 3],
        }
    )
    LocalCSVClient().write([0, 1, 0], identifiers)
    out = tmp_path / "predictions.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["OPERA", "TIPOVUELO", "MES", "prediction"]
    assert list(df["prediction"]) == [0, 1, 0]
    assert list(df["OPERA"]) == ["Grupo LATAM", "Sky Airline", "Copa Air"]
    assert list(df["MES"]) == [1, 2, 3]


def test_local_csv_reader_write_creates_dir(tmp_path, monkeypatch):
    """write() creates the artifacts directory if it does not exist."""
    new_dir = tmp_path / "new_artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(new_dir))
    identifiers = pd.DataFrame({"OPERA": ["Grupo LATAM"], "TIPOVUELO": ["I"], "MES": [1]})
    LocalCSVClient().write([1], identifiers)
    assert (new_dir / "predictions.csv").exists()


# --- BigQueryClient ---


def test_bigquery_reader_satisfies_protocol():
    """BigQueryClient satisfies the DataReader protocol at runtime."""
    mock_client = MagicMock()
    reader = BigQueryClient(_client=mock_client)
    assert isinstance(reader, DataReader)


def test_bigquery_reader_read_formats_sql_and_queries(tmp_path, monkeypatch):
    """read() formats the SQL file with settings values and calls client.query()."""
    sql_file = tmp_path / "load_data.sql"
    sql_file.write_text("SELECT * FROM `{project}.{dataset}.{raw_table}`\n")

    mock_client = MagicMock()
    expected_df = pd.DataFrame({"col": [1, 2]})
    mock_client.query.return_value.to_dataframe.return_value = expected_df

    monkeypatch.setenv("GCP_PROJECT_ID", "my-proj")
    monkeypatch.setenv("BQ_DATASET", "my_ds")
    monkeypatch.setenv("BQ_RAW_TABLE", "raw_flights")
    clear_settings_cache()

    with patch("challenge.connectors.bigquery.BigQueryClient._SQL_PATH", sql_file):
        reader = BigQueryClient(_client=mock_client)
        result = reader.read()

    mock_client.query.assert_called_once_with(
        "SELECT * FROM `my-proj.my_ds.raw_flights`\n"
    )
    assert result.equals(expected_df)


def test_bigquery_reader_read_raises_when_sql_missing():
    """read() raises FileNotFoundError when load_data.sql is absent."""
    mock_client = MagicMock()
    with patch(
        "challenge.connectors.bigquery.BigQueryClient._SQL_PATH",
        Path("/nonexistent/path/load_data.sql"),
    ):
        reader = BigQueryClient(_client=mock_client)
        with pytest.raises(FileNotFoundError):
            reader.read()


def test_bigquery_reader_write_calls_load_table(monkeypatch):
    """write() builds the correct DataFrame and calls load_table_from_dataframe."""
    mock_client = MagicMock()

    monkeypatch.setenv("GCP_PROJECT_ID", "my-proj")
    monkeypatch.setenv("BQ_DATASET", "my_ds")
    monkeypatch.setenv("BQ_PREDICTIONS_TABLE", "preds")
    clear_settings_cache()

    reader = BigQueryClient(_client=mock_client)
    identifiers = pd.DataFrame(
        {"OPERA": ["Grupo LATAM", "Sky Airline"], "TIPOVUELO": ["I", "N"], "MES": [1, 2]}
    )
    reader.write([0, 1], identifiers)

    call_args = mock_client.load_table_from_dataframe.call_args
    df_arg = call_args[0][0]
    table_arg = call_args[0][1]
    assert table_arg == "my-proj.my_ds.preds"
    assert list(df_arg.columns) == ["OPERA", "TIPOVUELO", "MES", "prediction"]
    assert list(df_arg["prediction"]) == [0, 1]


def test_bigquery_writer_load_job_failure_propagates(monkeypatch):
    """write() propagates RuntimeError when the BigQuery load job fails."""
    mock_client = MagicMock()
    mock_client.load_table_from_dataframe.return_value.result.side_effect = RuntimeError(
        "BQ load failed"
    )

    monkeypatch.setenv("GCP_PROJECT_ID", "p")
    monkeypatch.setenv("BQ_DATASET", "d")
    monkeypatch.setenv("BQ_PREDICTIONS_TABLE", "t")
    clear_settings_cache()

    reader = BigQueryClient(_client=mock_client)
    identifiers = pd.DataFrame({"OPERA": ["Grupo LATAM"], "TIPOVUELO": ["I"], "MES": [1]})

    with pytest.raises(RuntimeError, match="BQ load failed"):
        reader.write([0], identifiers)

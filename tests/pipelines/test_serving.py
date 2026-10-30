"""Unit tests for challenge/serving.py entry point."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from challenge.config import Settings
from challenge.connectors.gcs import GCSCSVClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import FEATURES_COLS
from challenge.serving import main


def _make_features() -> pd.DataFrame:
    return pd.DataFrame({col: [0] for col in FEATURES_COLS})


def _fake_settings(deployment_mode: str = "gcp") -> Settings:
    return Settings(
        deployment_mode=deployment_mode,
        data_dir="data",
        artifacts_dir="artifacts",
        gcp_project_id="test-proj",
        gcp_region="us-central1",
        gcp_bucket_artifacts="test-bucket",
        gcp_bucket_input="test-input-bucket",
        bq_dataset="test_ds",
        bq_raw_table="raw_flights",
        bq_predictions_table="predictions",
    )


def _fake_flight_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fecha-I": ["2017-06-01 10:00:00"],
            "Fecha-O": ["2017-06-01 10:30:00"],
            "OPERA": ["Grupo LATAM"],
            "TIPOVUELO": ["I"],
            "MES": [6],
        }
    )


# --- GCP branch ---


def test_main_gcp_branch_uses_gcs_csv_reader():
    """main() in GCP mode instantiates GCSCSVClient and calls reader.write()."""
    mock_reader = MagicMock(spec=GCSCSVClient)
    mock_reader.read.return_value = _fake_flight_data()

    with (
        patch("challenge.serving.get_settings", return_value=_fake_settings("gcp")),
        patch("challenge.connectors.gcs.GCSCSVClient", return_value=mock_reader),
        patch("challenge.serving.step_load_model"),
    ):
        main()

    mock_reader.write.assert_called_once()
    predictions_arg = mock_reader.write.call_args[0][0]
    assert isinstance(predictions_arg, list)
    assert all(p in (0, 1) for p in predictions_arg)
    identifiers_arg = mock_reader.write.call_args[0][1]
    assert list(identifiers_arg.columns) == ["OPERA", "TIPOVUELO", "MES"]


def test_main_local_branch_uses_local_csv_reader():
    """main() in local mode instantiates LocalCSVReader and calls reader.write()."""
    mock_reader = MagicMock(spec=LocalCSVClient)
    mock_reader.read.return_value = _fake_flight_data()

    with (
        patch("challenge.serving.get_settings", return_value=_fake_settings("local")),
        patch("challenge.serving.LocalCSVClient", return_value=mock_reader),
        patch("challenge.serving.step_load_model"),
    ):
        main()

    mock_reader.write.assert_called_once()
    identifiers_arg = mock_reader.write.call_args[0][1]
    assert list(identifiers_arg.columns) == ["OPERA", "TIPOVUELO", "MES"]


def test_settings_gcp_mode_raises_when_gcp_vars_missing():
    """Settings raises ValueError when DEPLOYMENT_MODE=gcp but GCP vars are absent."""
    with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
        Settings(
            deployment_mode="gcp",
            data_dir="data",
            artifacts_dir="artifacts",
            gcp_project_id="",
            gcp_region="us-central1",
            gcp_bucket_artifacts="",
            gcp_bucket_input="",
            bq_dataset="mle_challenge",
            bq_raw_table="raw_flights",
            bq_predictions_table="predictions",
        )


# --- error propagation ---


def test_main_step_load_data_error_propagates():
    """main() propagates FileNotFoundError raised by step_load_data."""
    with (
        patch("challenge.serving.get_settings", return_value=_fake_settings("local")),
        patch("challenge.serving.LocalCSVClient"),
        patch("challenge.serving.step_load_model"),
        patch(
            "challenge.serving.step_load_data",
            side_effect=FileNotFoundError("no data"),
        ),
        pytest.raises(FileNotFoundError, match="no data"),
    ):
        main()


def test_main_step_predict_error_propagates():
    """main() propagates RuntimeError raised by step_predict."""
    with (
        patch("challenge.serving.get_settings", return_value=_fake_settings("local")),
        patch("challenge.serving.LocalCSVClient"),
        patch("challenge.serving.step_load_model"),
        patch("challenge.serving.step_load_data", return_value=_fake_flight_data()),
        patch("challenge.serving.step_preprocess", return_value=_make_features()),
        patch(
            "challenge.serving.step_predict",
            side_effect=RuntimeError("predict failed"),
        ),
        pytest.raises(RuntimeError, match="predict failed"),
    ):
        main()

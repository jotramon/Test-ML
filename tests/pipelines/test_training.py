"""Unit tests for challenge/training.py entry point."""

from unittest.mock import patch

import pandas as pd

from challenge.config import Settings
from challenge.training import main


def _fake_settings(deployment_mode: str = "local") -> Settings:
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


def _make_features() -> pd.DataFrame:
    return pd.DataFrame({col: [0] for col in FEATURES_COLS})


def _make_target() -> pd.DataFrame:
    return pd.DataFrame({"delay": [0]})


def test_main_local_branch_orchestrates_all_steps():
    """main() local mode chains all four training steps and returns the artifact URI."""
    features = _make_features()
    target = _make_target()

    with (
        patch("challenge.training.get_settings", return_value=_fake_settings("local")),
        patch("challenge.training.LocalCSVClient"),
        patch(
            "challenge.training.step_load_data", return_value=_fake_flight_data()
        ) as mock_load,
        patch(
            "challenge.training.step_preprocess", return_value=(features, target)
        ) as mock_pre,
        patch("challenge.training.step_train") as mock_train,
        patch(
            "challenge.training.step_save", return_value="artifacts/run/model.joblib"
        ) as mock_save,
    ):
        result = main()

    mock_load.assert_called_once()
    mock_pre.assert_called_once()
    mock_train.assert_called_once()
    mock_save.assert_called_once()
    assert result == "artifacts/run/model.joblib"


def test_main_local_branch_passes_settings_values_to_steps():
    """main() passes gcp_project_id and bq_dataset to step_load_data; bucket to step_save."""
    features = _make_features()
    target = _make_target()
    settings = _fake_settings("local")

    with (
        patch("challenge.training.get_settings", return_value=settings),
        patch("challenge.training.LocalCSVClient"),
        patch(
            "challenge.training.step_load_data", return_value=_fake_flight_data()
        ) as mock_load,
        patch("challenge.training.step_preprocess", return_value=(features, target)),
        patch("challenge.training.step_train"),
        patch(
            "challenge.training.step_save", return_value="artifacts/run/model.joblib"
        ) as mock_save,
    ):
        main()

    assert mock_load.call_args[0][1] == settings.gcp_project_id
    assert mock_load.call_args[0][2] == settings.bq_dataset
    assert mock_save.call_args[0][1] == settings.gcp_bucket_artifacts


def test_main_gcp_branch_uses_bigquery_and_gcs():
    """main() gcp mode instantiates BigQueryClient and GCSArtifactStore."""
    features = _make_features()
    target = _make_target()

    with (
        patch("challenge.training.get_settings", return_value=_fake_settings("gcp")),
        patch("challenge.training.BigQueryClient") as mock_bq,
        patch("challenge.connectors.gcs.GCSArtifactStore") as mock_gcs,
        patch("challenge.training.step_load_data", return_value=_fake_flight_data()),
        patch("challenge.training.step_preprocess", return_value=(features, target)),
        patch("challenge.training.step_train"),
        patch("challenge.training.step_save", return_value="gs://bucket/run/model.joblib"),
    ):
        result = main()

    mock_bq.assert_called_once()
    mock_gcs.assert_called_once()
    assert result == "gs://bucket/run/model.joblib"

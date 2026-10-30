"""Unit tests for challenge/config.py (Settings, get_settings, clear_settings_cache)."""

import pytest

from challenge.config import Settings, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Clear the lru_cache before and after each test so env patches take effect."""
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_get_settings_returns_settings_object():
    """get_settings() returns a Settings instance."""
    result = get_settings()
    assert isinstance(result, Settings)


def test_get_settings_reads_env_vars(monkeypatch):
    """get_settings() picks up env var overrides applied before the first call."""
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("DATA_DIR", "custom_data_dir")
    result = get_settings()
    assert result.deployment_mode == "local"
    assert result.data_dir == "custom_data_dir"


def test_settings_gcp_mode_raises_for_missing_bucket_artifacts():
    """Settings raises ValueError when gcp_bucket_artifacts is empty in gcp mode."""
    with pytest.raises(ValueError, match="GCP_BUCKET_ARTIFACTS"):
        Settings(
            deployment_mode="gcp",
            data_dir="data",
            artifacts_dir="artifacts",
            gcp_project_id="my-project",
            gcp_region="us-central1",
            gcp_bucket_artifacts="",
            gcp_bucket_input="my-input",
            bq_dataset="mle_challenge",
            bq_raw_table="raw_flights",
            bq_predictions_table="predictions",
        )


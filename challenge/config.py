"""Centralised, lazily-loaded configuration for the challenge package."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """All environment-variable configuration in one place."""

    deployment_mode: str
    data_dir: str
    artifacts_dir: str
    gcp_project_id: str
    gcp_region: str
    gcp_bucket_artifacts: str
    gcp_bucket_input: str
    bq_dataset: str
    bq_raw_table: str
    bq_predictions_table: str

    def __post_init__(self) -> None:
        """Validate required GCP vars when deployment_mode is 'gcp'."""
        ...


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings read from environment variables."""
    ...


def clear_settings_cache() -> None:
    """Invalidate the cached settings — call in test teardown after patching env vars."""
    get_settings.cache_clear()

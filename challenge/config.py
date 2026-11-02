"""Centralised, lazily-loaded configuration for the challenge package."""

import os
from dataclasses import dataclass
from functools import lru_cache

_VALID_MODES = ("local", "gcp")


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
    training_job: str = "mle-training"
    serving_job: str = "mle-serving"

    def __post_init__(self) -> None:
        """Validate required GCP vars when deployment_mode is 'gcp'."""
        if self.deployment_mode not in _VALID_MODES:
            raise ValueError(
                f"DEPLOYMENT_MODE must be one of {_VALID_MODES}, got '{self.deployment_mode}'"
            )
        if self.deployment_mode != "gcp":
            return

        required = {
            "GCP_PROJECT_ID": self.gcp_project_id,
            "GCP_REGION": self.gcp_region,
            "GCP_BUCKET_ARTIFACTS": self.gcp_bucket_artifacts,
            "GCP_BUCKET_INPUT": self.gcp_bucket_input,
            "BQ_DATASET": self.bq_dataset,
            "BQ_RAW_TABLE": self.bq_raw_table,
            "BQ_PREDICTIONS_TABLE": self.bq_predictions_table,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables for gcp mode: {', '.join(missing)}"
            )

    def bq_table_id(self, table: str) -> str:
        """Return the fully qualified ``project.dataset.table`` id of a dataset table."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{table}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings read from environment variables."""
    return Settings(
        deployment_mode=os.environ.get("DEPLOYMENT_MODE", "local"),
        data_dir=os.environ.get("DATA_DIR", "data"),
        artifacts_dir=os.environ.get("ARTIFACTS_DIR", "artifacts"),
        gcp_project_id=os.environ.get("GCP_PROJECT_ID", ""),
        gcp_region=os.environ.get("GCP_REGION", "us-central1"),
        gcp_bucket_artifacts=os.environ.get("GCP_BUCKET_ARTIFACTS", ""),
        gcp_bucket_input=os.environ.get("GCP_BUCKET_INPUT", ""),
        bq_dataset=os.environ.get("BQ_DATASET", "mle_challenge"),
        bq_raw_table=os.environ.get("BQ_RAW_TABLE", "raw_flights"),
        bq_predictions_table=os.environ.get("BQ_PREDICTIONS_TABLE", "predictions"),
        training_job=os.environ.get("TRAINING_JOB", "mle-training"),
        serving_job=os.environ.get("SERVING_JOB", "mle-serving"),
    )


def clear_settings_cache() -> None:
    """Invalidate the cached settings — call in test teardown after patching env vars."""
    get_settings.cache_clear()

"""Cloud Run Job entry point for model training."""

import logging
import sys
import uuid

import pandas as pd

from challenge.config import get_settings
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)


### Define here all the step for the pipeline implementation ###

def main() -> str:
    """Orchestrate the full training pipeline."""
    ...
    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)

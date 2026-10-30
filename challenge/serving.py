"""Cloud Run Job entry point for model serving."""

import logging
import sys

import pandas as pd

from challenge.config import get_settings
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

### Define here all the step for the pipeline implementation ###


def main() -> str:
    """Orchestrate the full serving pipeline."""
    ...


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Serving job failed")
        sys.exit(1)

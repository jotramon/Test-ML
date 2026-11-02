"""Pipeline control FastAPI — triggers Cloud Run Job training/serving pipelines."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from challenge.api import utils
from challenge.config import get_settings

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAME = "delay-classifier"
PREDICT_PREVIEW_SIZE = 10
MAX_PAGE_SIZE = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — validate configuration before serving traffic."""
    settings = get_settings()
    logger.info("Pipeline control API starting in %s mode", settings.deployment_mode)
    yield


app = FastAPI(lifespan=lifespan)


# Thin wrappers over utils bound to the active settings; the endpoints call
# these so each external dependency can be patched independently.


def _check_raw_flights() -> int:
    return utils.count_raw_flights(get_settings())


def _check_model_exists() -> bool:
    return utils.model_exists(get_settings())


def _submit_and_wait(job_name: str) -> Any:
    return utils.run_job(get_settings(), job_name)


def _query_predictions(
    page: int,
    page_size: int,
    opera: list[str] | None,
    tipovuelo: list[str] | None,
    mes: list[int] | None,
) -> dict:
    return utils.query_predictions(
        get_settings(), page, page_size, opera=opera, tipovuelo=tipovuelo, mes=mes
    )


def _pipeline_error(name: str, exc: Exception) -> HTTPException:
    if isinstance(exc, utils.JobAlreadyRunningError):
        return HTTPException(
            status_code=409,
            detail=f"{name} pipeline is already running; retry when it finishes",
        )
    logger.exception("%s pipeline failed", name)
    return HTTPException(
        status_code=500,
        detail={"message": f"{name} pipeline failed", "details": str(exc)},
    )


# Endpoints are sync so FastAPI runs them in its threadpool: a training run
# blocks for minutes and must not stall the event loop serving other requests.


@app.get("/health", status_code=200)
def get_health() -> dict:
    """Return liveness status."""
    return {"status": "ok"}


@app.post("/pipeline/train", status_code=200)
def post_pipeline_train() -> dict:
    """Trigger the training Cloud Run Job and return model metadata."""
    if _check_raw_flights() == 0:
        raise HTTPException(status_code=422, detail="raw_flights is empty; load data first")
    try:
        execution = _submit_and_wait(get_settings().training_job)
    except Exception as exc:
        raise _pipeline_error("Training", exc) from exc

    try:
        metadata = utils.latest_metadata(get_settings())
    except FileNotFoundError:
        logger.warning("Training succeeded but no model metadata was found")
        metadata = {}
    return {
        "model_display_name": MODEL_DISPLAY_NAME,
        "pipeline_job_id": execution.name,
        "version": metadata.get("version"),
        "trained_at": metadata.get("trained_at"),
        "training_rows": metadata.get("training_rows"),
        "validation_rows": metadata.get("validation_rows"),
        "metrics": metadata.get("metrics", {}),
    }


@app.post("/pipeline/predict", status_code=200)
def post_pipeline_predict() -> dict:
    """Trigger the serving Cloud Run Job and return first 10 predictions."""
    if not _check_model_exists():
        raise HTTPException(status_code=422, detail="no trained model; call /pipeline/train first")
    try:
        execution = _submit_and_wait(get_settings().serving_job)
    except Exception as exc:
        raise _pipeline_error("Serving", exc) from exc

    utils.predictions_cache.invalidate()
    result = _query_predictions(1, PREDICT_PREVIEW_SIZE, None, None, None)
    return {
        "pipeline_job_id": execution.name,
        "total_predictions": result["total_predictions"],
        "predictions": result["predictions"],
    }


@app.get("/pipeline/predict/results", status_code=200)
def get_prediction_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=MAX_PAGE_SIZE),
    opera: str | None = None,
    tipovuelo: str | None = None,
    mes: str | None = None,
) -> dict:
    """Return a paginated page of predictions, optionally filtered by OPERA/TIPOVUELO/MES."""
    try:
        months = utils.parse_months(mes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _query_predictions(
        page,
        page_size,
        utils.parse_csv_param(opera),
        utils.parse_csv_param(tipovuelo),
        months,
    )
    return {"page": page, "page_size": page_size, **result}

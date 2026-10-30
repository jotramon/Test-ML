"""Pipeline control FastAPI — triggers Cloud Run Job training/serving pipelines."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — placeholder for startup/shutdown logic."""
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Return liveness status."""
    return {"status": "ok"}


@app.post("/pipeline/train", status_code=200)
async def post_pipeline_train() -> dict:
    """Trigger the training Cloud Run Job and return model metadata."""
    ...


@app.post("/pipeline/predict", status_code=200)
async def post_pipeline_predict() -> dict:
    """Trigger the serving Cloud Run Job and return first 10 predictions."""
    ...


@app.get("/pipeline/predict/results", status_code=200)
async def get_prediction_results(
    page: int = 1,
    page_size: int = 10,
    opera: str | None = None,
    tipovuelo: str | None = None,
    mes: str | None = None,
) -> dict:
    """Return a paginated page of predictions, optionally filtered by OPERA/TIPOVUELO/MES."""
    ...

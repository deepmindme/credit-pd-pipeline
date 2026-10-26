"""REST API для PD-модели.

Локально: uvicorn src.api.app:app --reload
Документация Swagger: http://127.0.0.1:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    ClientFeatures,
    HealthResponse,
    PredictionResponse,
)
from src.models.predict import DEFAULT_MODEL_PATH, LoadedModel, load_model

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        app.state.model = load_model(model_path)
        logger.info("Модель загружена из %s", model_path)
    except FileNotFoundError:
        # Сервис всё равно поднимается: /health честно покажет, что модели нет,
        # а /predict вернёт 503 вместо падения контейнера.
        app.state.model = None
        logger.error("Файл модели не найден: %s (выполните dvc pull)", model_path)
    yield


app = FastAPI(
    title="Credit Default PD API",
    description="Оценка вероятности дефолта клиента кредитной карты в следующем месяце.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model(request: Request) -> LoadedModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return model


def model_version(model: LoadedModel) -> str:
    meta = model.meta
    if not meta:
        return "unknown"
    return f"{meta.get('experiment')}-v{meta.get('registered_version')}"


def score(model: LoadedModel, clients: list[ClientFeatures]) -> list[PredictionResponse]:
    df = pd.DataFrame([c.model_dump() for c in clients])
    result = model.predict(df)
    return [
        PredictionResponse(
            default_prediction=int(row.default_prediction),
            default_probability=round(float(row.default_probability), 6),
            threshold=round(model.threshold, 6),
        )
        for row in result.itertuples()
    ]


@app.get("/")
def read_root() -> dict:
    return {"message": "Credit Default Prediction API is alive!", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    model = request.app.state.model
    if model is None:
        # 503, чтобы HEALTHCHECK Docker и скрипт запуска не считали сервис готовым.
        response.status_code = 503
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model_version(model),
        threshold=model.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures, request: Request) -> PredictionResponse:
    return score(get_model(request), [client])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    model = get_model(request)
    return BatchResponse(
        model_version=model_version(model), predictions=score(model, batch.clients)
    )

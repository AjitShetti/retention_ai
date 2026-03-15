from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import Settings, get_settings
from src.logger import logger
from src.utils import INVERSE_TARGET_MAP, prepare_inference_frame


def load_model_bundle(settings: Settings) -> tuple[dict, dict]:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {settings.model_path}")
    if not settings.metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found at {settings.metadata_path}")

    model_bundle = joblib.load(settings.model_path)
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    return model_bundle, metadata


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = active_settings
        try:
            model_bundle, metadata = load_model_bundle(active_settings)
            app.state.model_bundle = model_bundle
            app.state.metadata = metadata
            app.state.model_loaded = True
            app.state.model_error = None
            logger.info("Loaded model artifact from %s", active_settings.model_path)
        except Exception as error:
            app.state.model_bundle = None
            app.state.metadata = None
            app.state.model_loaded = False
            app.state.model_error = str(error)
            logger.exception("Failed to load model artifacts")
        yield

    app = FastAPI(title="Retention AI", version=active_settings.model_version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        metadata = app.state.metadata or {}
        status = "ok" if app.state.model_loaded else "degraded"
        return HealthResponse(
            status=status,
            model_loaded=app.state.model_loaded,
            model_name=metadata.get("model_name", active_settings.model_name),
            model_version=metadata.get("model_version", active_settings.model_version),
            detail=app.state.model_error,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if not app.state.model_loaded or app.state.model_bundle is None:
            raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

        features = prepare_inference_frame(pd.DataFrame([request.to_feature_dict()]))
        pipeline = app.state.model_bundle["pipeline"]
        probability = float(pipeline.predict_proba(features)[0][1])
        prediction = INVERSE_TARGET_MAP[int(probability >= 0.5)]
        metadata = app.state.metadata or {}

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_name=metadata.get("model_name", active_settings.model_name),
            model_version=metadata.get("model_version", active_settings.model_version),
        )

    return app


app = create_app()

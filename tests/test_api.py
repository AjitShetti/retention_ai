from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.config import BASE_DIR, Settings
from src.pipeline.train import run_training
from src.utils import FEATURE_COLUMNS, prepare_inference_frame, prepare_training_frame, validate_required_columns


@pytest.fixture()
def trained_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        model_filename="churn_model.joblib",
        metadata_filename="model_metadata.json",
        data_path=BASE_DIR / "notebooks" / "data" / "data.csv",
        log_level="INFO",
        model_name="retention-baseline",
        model_version="test-version",
    )
    run_training(settings)
    return settings


@pytest.fixture()
def sample_payload() -> dict[str, object]:
    return {
        "gender": "female",
        "SeniorCitizen": 0,
        "Partner": " yes ",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "dsl",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "electronic check",
        "MonthlyCharges": 55.2,
        "TotalCharges": "650.5",
    }


def test_validate_required_columns_raises_for_missing_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(__import__("pandas").DataFrame([{"gender": "Female"}]))


def test_prepare_training_frame_coerces_total_charges_and_normalizes_categories():
    import pandas as pd

    dataset = pd.DataFrame(
        [
            {
                "customerID": "0001-A",
                "gender": " female ",
                "SeniorCitizen": "0",
                "Partner": " yes ",
                "Dependents": "No",
                "tenure": "5",
                "PhoneService": "Yes",
                "MultipleLines": "No phone service",
                "InternetService": "dsl",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "electronic check",
                "MonthlyCharges": "29.85",
                "TotalCharges": "  ",
                "Churn": "No",
            }
        ]
    )

    features, target = prepare_training_frame(dataset)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "gender"] == "Female"
    assert features.loc[0, "Partner"] == "Yes"
    assert features.loc[0, "InternetService"] == "DSL"
    assert __import__("pandas").isna(features.loc[0, "TotalCharges"])
    assert target.iloc[0] == 0


def test_training_creates_artifacts_and_metadata(trained_settings: Settings):
    assert trained_settings.model_path.exists()
    assert trained_settings.metadata_path.exists()

    metadata = json.loads(trained_settings.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "test-version"
    assert "validation_accuracy" in metadata["metrics"]


def test_health_endpoint_reports_loaded_model(trained_settings: Settings):
    with TestClient(create_app(trained_settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction(trained_settings: Settings, sample_payload: dict[str, object]):
    with TestClient(create_app(trained_settings)) as client:
        response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"Yes", "No"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-version"


def test_predict_returns_validation_error_for_invalid_payload(trained_settings: Settings, sample_payload: dict[str, object]):
    invalid_payload = dict(sample_payload)
    invalid_payload.pop("Contract")

    with TestClient(create_app(trained_settings)) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_returns_controlled_error_when_model_missing(tmp_path: Path, sample_payload: dict[str, object]):
    missing_settings = Settings(
        artifact_dir=tmp_path / "missing-artifacts",
        model_filename="churn_model.joblib",
        metadata_filename="model_metadata.json",
        data_path=BASE_DIR / "notebooks" / "data" / "data.csv",
        log_level="INFO",
        model_name="retention-baseline",
        model_version="test-version",
    )

    with TestClient(create_app(missing_settings)) as client:
        health_response = client.get("/health")
        predict_response = client.post("/predict", json=sample_payload)

    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] is False
    assert predict_response.status_code == 503


def test_prepare_inference_frame_preserves_feature_order(sample_payload: dict[str, object]):
    import pandas as pd

    frame = prepare_inference_frame(pd.DataFrame([sample_payload]))
    assert list(frame.columns) == FEATURE_COLUMNS

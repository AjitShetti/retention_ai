from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Settings, get_settings
from src.exception import CustomException
from src.logger import logger
from src.utils import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_training_frame


def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_training_pipeline(estimator: Any | None = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = estimator or LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(data_path: Path, estimator: Any | None = None) -> dict[str, Any]:
    dataset = load_dataset(data_path)
    features, target = prepare_training_frame(dataset)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_training_pipeline(estimator=estimator)
    pipeline.fit(X_train, y_train)

    valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
    valid_predictions = (valid_probabilities >= 0.5).astype(int)

    metrics = {
        "validation_accuracy": round(float(accuracy_score(y_valid, valid_predictions)), 4),
        "validation_roc_auc": round(float(roc_auc_score(y_valid, valid_probabilities)), 4),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
    }

    logger.info("Model trained successfully with metrics: %s", metrics)
    return {"pipeline": pipeline, "metrics": metrics, "feature_columns": FEATURE_COLUMNS}


def persist_artifacts(model_bundle: dict[str, Any], settings: Settings) -> dict[str, Any]:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, settings.model_path)

    metadata = {
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "artifact_path": str(settings.model_path),
        "feature_columns": model_bundle["feature_columns"],
        "metrics": model_bundle["metrics"],
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model artifact to %s", settings.model_path)
    return metadata


def run_training(settings: Settings | None = None) -> dict[str, Any]:
    active_settings = settings or get_settings()
    try:
        model_bundle = train_model(active_settings.data_path)
        return persist_artifacts(model_bundle, active_settings)
    except Exception as error:
        logger.exception("Training pipeline failed")
        raise CustomException(error, error_detail=__import__("sys")) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the retention AI churn model.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional CSV path override.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Optional artifact directory override.")
    args = parser.parse_args()

    settings = get_settings()
    if args.data_path is not None or args.artifact_dir is not None:
        settings = Settings(
            artifact_dir=args.artifact_dir or settings.artifact_dir,
            model_filename=settings.model_filename,
            metadata_filename=settings.metadata_filename,
            data_path=args.data_path or settings.data_path,
            log_level=settings.log_level,
            model_name=settings.model_name,
            model_version=settings.model_version,
        )

    metadata = run_training(settings)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

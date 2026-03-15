from __future__ import annotations

from typing import Any

import pandas as pd


TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

CANONICAL_CATEGORY_MAPS = {
    "gender": {"male": "Male", "female": "Female"},
    "Partner": {"yes": "Yes", "no": "No"},
    "Dependents": {"yes": "Yes", "no": "No"},
    "PhoneService": {"yes": "Yes", "no": "No"},
    "MultipleLines": {"yes": "Yes", "no": "No", "no phone service": "No phone service"},
    "InternetService": {"dsl": "DSL", "fiber optic": "Fiber optic", "no": "No"},
    "OnlineSecurity": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "OnlineBackup": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "DeviceProtection": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "TechSupport": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "StreamingTV": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "StreamingMovies": {"yes": "Yes", "no": "No", "no internet service": "No internet service"},
    "Contract": {
        "month-to-month": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
    },
    "PaperlessBilling": {"yes": "Yes", "no": "No"},
    "PaymentMethod": {
        "electronic check": "Electronic check",
        "mailed check": "Mailed check",
        "bank transfer (automatic)": "Bank transfer (automatic)",
        "credit card (automatic)": "Credit card (automatic)",
    },
}

TARGET_MAP = {"yes": 1, "no": 0}
INVERSE_TARGET_MAP = {1: "Yes", 0: "No"}


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    required = required_columns or REQUIRED_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def canonicalize_categorical_value(column: str, value: Any) -> Any:
    if pd.isna(value):
        return value
    normalized = str(value).strip()
    if not normalized:
        return pd.NA
    return CANONICAL_CATEGORY_MAPS.get(column, {}).get(normalized.lower(), normalized)


def normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in CATEGORICAL_FEATURES:
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(
                lambda value, feature=column: canonicalize_categorical_value(feature, value)
            )

    for column in NUMERIC_FEATURES:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_required_columns(df)
    normalized = normalize_feature_frame(df)
    features = normalized[FEATURE_COLUMNS].copy()
    target = normalized[TARGET_COLUMN].astype(str).str.strip().str.lower().map(TARGET_MAP)
    if target.isna().any():
        raise ValueError("Target column contains unsupported values.")
    return features, target.astype(int)


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, FEATURE_COLUMNS)
    normalized = normalize_feature_frame(df)
    return normalized[FEATURE_COLUMNS].copy()

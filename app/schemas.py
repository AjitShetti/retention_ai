from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float | str

    @field_validator("gender", "InternetService", "Contract", "PaymentMethod", mode="before")
    @classmethod
    def validate_domain_fields(cls, value: object, info):
        if value is None:
            raise ValueError(f"{info.field_name} is required.")

        raw = str(value).strip()
        normalized = raw.lower()

        allowed_maps = {
            "gender": {"male": "Male", "female": "Female"},
            "InternetService": {"dsl": "DSL", "fiber optic": "Fiber optic", "no": "No"},
            "Contract": {
                "month-to-month": "Month-to-month",
                "one year": "One year",
                "two year": "Two year",
            },
            "PaymentMethod": {
                "electronic check": "Electronic check",
                "mailed check": "Mailed check",
                "bank transfer (automatic)": "Bank transfer (automatic)",
                "credit card (automatic)": "Credit card (automatic)",
            },
        }

        field_map = allowed_maps.get(info.field_name, {})
        if normalized not in field_map:
            allowed_values = list(field_map.values())
            raise ValueError(f"Unrecognized {info.field_name}. Allowed values: {allowed_values}")

        return field_map[normalized]

    def to_feature_dict(self) -> dict[str, object]:
        return self.model_dump()


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    detail: str | None = None

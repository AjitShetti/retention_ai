from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = Path(os.getenv("RETENTION_ARTIFACT_DIR", BASE_DIR / "artifacts"))
    model_filename: str = os.getenv("RETENTION_MODEL_FILENAME", "churn_model.joblib")
    metadata_filename: str = os.getenv("RETENTION_METADATA_FILENAME", "model_metadata.json")
    data_path: Path = Path(os.getenv("RETENTION_DATA_PATH", BASE_DIR / "notebooks" / "data" / "data.csv"))
    log_level: str = os.getenv("RETENTION_LOG_LEVEL", "INFO")
    model_name: str = os.getenv("RETENTION_MODEL_NAME", "retention-baseline")
    model_version: str = os.getenv("RETENTION_MODEL_VERSION", "0.1.0")

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifact_dir / self.metadata_filename


def get_settings() -> Settings:
    return Settings()

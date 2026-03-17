[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

# Retention AI (Churn Prediction)

Retention AI is a small, end-to-end **churn prediction** project: it trains a scikit-learn pipeline on the included Telco dataset and serves **single-record inference** through a FastAPI API.

## What you get

- **Training pipeline**: reproducible training via `python -m src.pipeline.train`
- **Artifacts**: `churn_model.joblib` + `model_metadata.json`
- **API**: `GET /health` and `POST /predict` (JSON in → JSON out)
- **CI**: GitHub Actions runs `pytest -q` on pushes/PRs to `main`

## System architecture diagram

### ASCII (quick scan)

```
notebooks/data/data.csv
        │
        ▼
  src.pipeline.train
  (prep + model fit)
        │
        ├──────────────► artifacts/churn_model.joblib
        └──────────────► artifacts/model_metadata.json
                                  │
                                  ▼
                            app.main (FastAPI)
                       ┌──────────┴──────────┐
                       ▼                     ▼
                  GET /health           POST /predict
                 status/metadata     prediction + probability
```

### Mermaid (renderable)

```mermaid
flowchart LR
  D[notebooks/data/data.csv] --> T[src.pipeline.train\ntrain + persist]
  T --> A[artifacts/churn_model.joblib]
  T --> M[artifacts/model_metadata.json]
  A --> API[app.main (FastAPI)]
  M --> API
  API --> H[GET /health]
  API --> P[POST /predict]
  P --> R[JSON response]
```

## How it works

During training, the dataset is validated and normalized, then a scikit-learn `Pipeline` is fitted with a `ColumnTransformer` (median imputation + scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features). The API enforces the same input contract and returns a churn prediction (`Yes`/`No`) with a probability score.

## Quickstart (local)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train

```bash
python -m src.pipeline.train
```

Artifacts are written by default to:
- `artifacts/churn_model.joblib`
- `artifacts/model_metadata.json`

### Serve

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### Health check

```bash
curl http://localhost:8000/health
```

### Predict

Example request body:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 55.2,
  "TotalCharges": 650.5
}
```

Curl example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":55.2,"TotalCharges":650.5}'
```

## Configuration

- `RETENTION_ARTIFACT_DIR`: directory containing model artifacts
- `RETENTION_DATA_PATH`: CSV path used by the training command
- `RETENTION_MODEL_NAME`: model name returned by the API
- `RETENTION_MODEL_VERSION`: model version returned by the API
- `RETENTION_LOG_LEVEL`: logging level such as `INFO` or `DEBUG`

## Tests

```bash
pytest -q
```

## Docker

Build:

```bash
docker build -t retention-ai .
```

Run:

```bash
docker run --rm -p 8000:8000 -e RETENTION_ARTIFACT_DIR=/app/artifacts retention-ai
```
